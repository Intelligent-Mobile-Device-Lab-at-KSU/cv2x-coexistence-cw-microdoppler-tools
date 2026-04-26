#!/usr/bin/env python3
"""
cv2x_cw_txrx.py -- Simultaneous C-V2X CW TX Replay + Narrow-Band RX Capture
=============================================================================
Replays a CW-injected C-V2X IQ file on one daughtercard (TX, 30.72 Msps)
while simultaneously capturing a narrow band centered on the CW tone's RF
frequency on the other daughtercard (RX, ~192 kSps).

The RX center frequency is auto-computed from the TX sidecar JSON:
    rx_freq = tx_center_freq + f_cw_hz + rx_offset

The --rx-offset (default 50 kHz) moves the CW tone away from DC in the RX
baseband to avoid DC offset contamination.

Hardware:
    USRP X310 with two SBX daughtercards:
      - Radio#0 / subdev A:0 -> TX (replay CW-injected file at 30.72 Msps)
      - Radio#1 / subdev B:0 -> RX (narrow-band capture at ~192 kSps)
    GPSDO provides shared timing reference.

Example:
    python cv2x_cw_txrx.py \\
        --tx-file cv2x_iq_cw.cf32 \\
        --rx-file cv2x_rx_doppler.cf32 \\
        --args "serial=33767A5,master_clock_rate=184.32e6" \\
        --tx-freq 5.915e9 --tx-gain 15 \\
        --rx-gain 30 --rx-rate 192e3 \\
        --duration 60 --gpsdo --loop
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

from gnuradio import gr, blocks

# ============================================================================
# 3GPP DFN timing constants  (TS 36.211 sidelink scrambling)
# ============================================================================
EPOCH_OFFSET_S = 2208988800   # seconds from 1900-01-01 to Unix epoch 1970-01-01
DFN_CYCLE_MS   = 10240        # 1024 frames x 10 subframes = 10.24 s
DFN_CYCLE_S    = DFN_CYCLE_MS / 1000.0

TX_SAMP_RATE = 30720000.0     # C-V2X 20 MHz channel at 30.72 Msps
DEFAULT_RX_RATE = 192000.0    # 184.32e6 / 960 = 192 kSps (clean decimation)


# ============================================================================
# GPSDO helpers  (copied from cv2x_capture_replay.py)
# ============================================================================
_QUERY_GPSDO_PATHS = [
    os.path.expanduser(r"~\radioconda\Library\lib\uhd\utils"
                       r"\query_gpsdo_sensors.exe"),
    r"C:\Program Files (x86)\National Instruments"
    r"\NI-USRP\utilities\query_gpsdo_sensors.exe",
]


def _gpsdo_preflight(dev_args, timeout=90):
    """Run query_gpsdo_sensors before touching GNU Radio."""
    exe = None
    for p in _QUERY_GPSDO_PATHS:
        if os.path.isfile(p):
            exe = p
            break
    if exe is None:
        print("[gpsdo] query_gpsdo_sensors not found -- skipping pre-flight.")
        return False

    cmd = [exe]
    if dev_args:
        cmd.append(f"--args={dev_args}")

    for attempt in range(1, 3):
        print(f"[gpsdo] Pre-flight attempt {attempt}/2 (timeout {timeout} s) ...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout)
            output = (result.stdout + result.stderr).strip()

            if result.returncode != 0:
                if ("No devices found" in output or
                        "LookupError" in output) and attempt < 2:
                    print(f"[gpsdo]   Device not ready (FPGA loading?). "
                          f"Retrying in 10 s ...")
                    time.sleep(10)
                    continue
                if output:
                    for line in output.splitlines():
                        print(f"[gpsdo]   {line}")
                return False

            if output:
                for line in output.splitlines():
                    print(f"[gpsdo]   {line}")

            ref_locked = "usrp locked to reference" in output.lower()
            if ref_locked:
                print("[gpsdo] 10 MHz ref LOCKED")
            return ref_locked

        except subprocess.TimeoutExpired:
            print(f"[gpsdo] Timed out after {timeout} s")
            if attempt < 2:
                continue
            return False
        except OSError as exc:
            print(f"[gpsdo] Could not run query_gpsdo_sensors: {exc}")
            return False
    return False


def _gpsdo_setup(usrp_block, label="usrp", timeout=90):
    """Wait for 10 MHz ref lock and sync USRP time to GPS PPS/UTC."""
    from gnuradio import uhd

    usrp_block.set_clock_source("gpsdo")
    usrp_block.set_time_source("gpsdo")

    print(f"[gpsdo:{label}] Waiting for ref_locked (up to {timeout} s) ...")
    ref_locked = False
    for _ in range(timeout):
        try:
            ref_locked = usrp_block.get_mboard_sensor("ref_locked").to_bool()
        except Exception:
            ref_locked = False
        if ref_locked:
            break
        time.sleep(1.0)

    if not ref_locked:
        print(f"[gpsdo:{label}] WARNING -- ref did not lock in {timeout} s.")
        return

    print(f"[gpsdo:{label}] 10 MHz reference LOCKED.")
    try:
        gps_locked = usrp_block.get_mboard_sensor("gps_locked").to_bool()
        print(f"[gpsdo:{label}] GPS nav lock: "
              f"{'YES' if gps_locked else 'no (not required)'}")
    except Exception:
        pass

    gps_time = usrp_block.get_mboard_sensor("gps_time").to_int()
    next_pps = gps_time + 1
    usrp_block.set_time_next_pps(uhd.time_spec(float(next_pps)))
    time.sleep(1.1)
    print(f"[gpsdo:{label}] USRP time set to {next_pps} (UTC via GPS PPS).")


# ============================================================================
# DFN cycle helpers
# ============================================================================
def _gps_to_cycle_offset_ms(gps_utc_secs):
    ms_since_1900 = int((gps_utc_secs + EPOCH_OFFSET_S) * 1000)
    return ms_since_1900 % DFN_CYCLE_MS


def _cycle_offset_to_dfn_sf(offset_ms):
    return (offset_ms // 10) % 1024, offset_ms % 10


def _compute_dfn_aligned_start(usrp_block, capture_offset_ms, guard_ms=200):
    from gnuradio import uhd
    now_secs    = usrp_block.get_time_now().get_real_secs()
    now_ms_1900 = int((now_secs + EPOCH_OFFSET_S) * 1000)
    now_offset  = now_ms_1900 % DFN_CYCLE_MS

    wait_ms = (capture_offset_ms - now_offset) % DFN_CYCLE_MS
    if wait_ms < guard_ms:
        wait_ms += DFN_CYCLE_MS

    target_ms_1900 = now_ms_1900 + wait_ms
    target_whole_s = target_ms_1900 // 1000 - EPOCH_OFFSET_S
    target_frac_ms = target_ms_1900 % 1000
    target_secs    = float(target_whole_s) + target_frac_ms / 1000.0

    verify_offset = target_ms_1900 % DFN_CYCLE_MS
    target_dfn, target_sf = _cycle_offset_to_dfn_sf(capture_offset_ms)

    print(f"[dfn] Current cycle offset : {now_offset} ms  "
          f"(DFN={now_offset // 10 % 1024}, SF={now_offset % 10})")
    print(f"[dfn] Target  cycle offset : {capture_offset_ms} ms  "
          f"(DFN={target_dfn}, SF={target_sf})  [verify={verify_offset}]")
    print(f"[dfn] Waiting {wait_ms} ms ({wait_ms / 1000.0:.3f} s) "
          f"for DFN-aligned TX start")

    if verify_offset != capture_offset_ms:
        print(f"[dfn] ERROR: Offset mismatch!")

    return uhd.time_spec(target_secs)


# ============================================================================
# Sidecar I/O
# ============================================================================
def _sidecar_path(iq_file):
    base, _ = os.path.splitext(iq_file)
    return base + ".json"


def read_sidecar(iq_file):
    path = _sidecar_path(iq_file)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_sidecar(iq_file, data):
    path = _sidecar_path(iq_file)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[sidecar] Wrote {path}")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tx-file", required=True,
                        help="CW-injected IQ file to replay (.cf32)")
    parser.add_argument("--rx-file", required=True,
                        help="Output narrow-band RX capture (.cf32)")
    parser.add_argument("--args", "-a", type=str,
                        default="serial=33767A5,master_clock_rate=184.32e6",
                        help="UHD device args")
    parser.add_argument("--tx-freq", type=float, default=5.915e9,
                        help="TX center frequency (default: 5.915 GHz)")
    parser.add_argument("--tx-gain", type=float, default=15,
                        help="TX gain in dB (default: 15)")
    parser.add_argument("--rx-freq", type=float, default=None,
                        help="RX center frequency (auto-computed from sidecar "
                             "if omitted)")
    parser.add_argument("--rx-gain", type=float, default=30,
                        help="RX gain in dB (default: 30)")
    parser.add_argument("--rx-rate", type=float, default=DEFAULT_RX_RATE,
                        help="RX sample rate in Hz (default: 192 kSps). "
                             "192e3 divides cleanly from 184.32 MHz.")
    parser.add_argument("--rx-offset", type=float, default=50e3,
                        help="LO offset to avoid DC in RX baseband (default: "
                             "50 kHz). CW appears at -rx_offset in baseband.")
    parser.add_argument("--duration", "-d", type=float, default=60.0,
                        help="RX capture duration in seconds (default: 60)")
    parser.add_argument("--gpsdo", action="store_true",
                        help="Enable GPSDO sync (DFN-aligned TX + PPS timing)")
    parser.add_argument("--gpsdo-timeout", type=int, default=90,
                        help="GPSDO lock timeout in seconds (default: 90)")
    parser.add_argument("--loop", dest="loop", action="store_true",
                        default=True, help="Loop the TX file (default: loop)")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Play TX file once then stop")
    parser.add_argument("--tx-subdev", default="A:0",
                        help="TX subdevice spec (default: A:0)")
    parser.add_argument("--rx-subdev", default="B:0",
                        help="RX subdevice spec (default: B:0)")
    return parser.parse_args()


# ============================================================================
# Flowgraph
# ============================================================================
class CV2XCwTxRx(gr.top_block):
    """Simultaneous TX replay + narrow-band RX capture."""

    def __init__(self, args):
        gr.top_block.__init__(self, "C-V2X CW TX/RX")
        from gnuradio import uhd

        # -- Read TX sidecar -------------------------------------------------
        self.tx_sidecar = read_sidecar(args.tx_file)
        if self.tx_sidecar is None:
            print(f"WARNING: No sidecar found for {args.tx_file}")
            self.tx_sidecar = {}

        cw_info = self.tx_sidecar.get("cw_inject", {})
        f_cw_hz = cw_info.get("f_cw_hz", 0.0)
        tx_center_freq = self.tx_sidecar.get("center_freq", args.tx_freq)

        # Compute RX frequency
        if args.rx_freq is not None:
            rx_freq = args.rx_freq
        else:
            cw_rf = tx_center_freq + f_cw_hz
            rx_freq = cw_rf + args.rx_offset
            print(f"[auto] CW RF frequency:  {cw_rf / 1e9:.6f} GHz "
                  f"(center {tx_center_freq/1e9:.3f} GHz + "
                  f"CW {f_cw_hz/1e3:.1f} kHz)")
            print(f"[auto] RX tuned to:      {rx_freq / 1e9:.6f} GHz "
                  f"(+{args.rx_offset/1e3:.0f} kHz offset)")
            print(f"[auto] CW in RX baseband: {-args.rx_offset/1e3:.0f} kHz")

        self.cw_baseband_hz = -(args.rx_offset)
        self.f_cw_hz = f_cw_hz
        self.cw_rf_hz = tx_center_freq + f_cw_hz

        # -- STEP 1: Open RX first (claims NI-RPC session) ------------------
        print("\n[rx] Opening RX on subdev", args.rx_subdev, "...")
        self.rx_src = uhd.usrp_source(
            args.args,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )
        self.rx_src.set_subdev_spec(args.rx_subdev)
        self.rx_src.set_samp_rate(args.rx_rate)
        self.rx_src.set_center_freq(rx_freq)
        self.rx_src.set_gain(args.rx_gain)
        self.rx_src.set_antenna("RX2")
        self.rx_src.set_auto_iq_balance(True, 0)
        actual_rx_rate = self.rx_src.get_samp_rate()
        print(f"[rx] Rate: {actual_rx_rate:.0f} sps "
              f"(requested {args.rx_rate:.0f})")
        print(f"[rx] Freq: {rx_freq/1e9:.6f} GHz  Gain: {args.rx_gain} dB")

        print("[rx] Waiting 2 s for X310 init to settle...")
        time.sleep(2.0)

        # -- STEP 2: Open TX (reuses session) --------------------------------
        print(f"\n[tx] Opening TX on subdev {args.tx_subdev} ...")
        self.tx_sink = uhd.usrp_sink(
            args.args,
            uhd.stream_args(
                cpu_format="fc32",
                args="send_buff_size=9998336",
                channels=[0]),
        )
        self.tx_sink.set_subdev_spec(args.tx_subdev)
        self.tx_sink.set_samp_rate(TX_SAMP_RATE)
        self.tx_sink.set_center_freq(args.tx_freq)
        self.tx_sink.set_gain(args.tx_gain)
        self.tx_sink.set_antenna("TX/RX")
        actual_tx_rate = self.tx_sink.get_samp_rate()
        print(f"[tx] Rate: {actual_tx_rate:.0f} sps  "
              f"Freq: {args.tx_freq/1e9:.3f} GHz  Gain: {args.tx_gain} dB")
        print("[tx] Waiting 2 s for USRP init ...")
        time.sleep(2.0)

        # -- GPSDO setup ---------------------------------------------------------
        # Do full GPSDO setup on tx_sink (same pattern as the working
        # cv2x_capture_replay.py --replay).  Since both blocks share the
        # same mboard, the set_time_next_pps() call propagates to both.
        # Then explicitly set clock/time source on rx_src as well.
        self._gpsdo = args.gpsdo
        if args.gpsdo:
            print()
            _gpsdo_setup(self.tx_sink, "tx", args.gpsdo_timeout)
            # RX shares same mboard -- ensure sources are set
            self.rx_src.set_clock_source("gpsdo")
            self.rx_src.set_time_source("gpsdo")

        # -- TX signal path: file_source -> tx_sink --------------------------
        self.tx_file_src = blocks.file_source(
            gr.sizeof_gr_complex, args.tx_file, repeat=args.loop)
        self.connect(self.tx_file_src, self.tx_sink)

        # -- RX signal path: rx_src -> file_sink ------------------------------
        self.rx_file_sink = blocks.file_sink(
            gr.sizeof_gr_complex, args.rx_file, append=False)
        self.rx_file_sink.set_unbuffered(False)
        self.connect(self.rx_src, self.rx_file_sink)

        # Store for sidecar writing
        self._args = args
        self._actual_rx_rate = actual_rx_rate
        self._rx_freq = rx_freq
        self._rx_start_utc = None

        # TX file info
        tx_file_size = os.path.getsize(args.tx_file)
        tx_num_samples = tx_file_size // 8  # complex64 = 8 bytes
        self._tx_duration_s = tx_num_samples / TX_SAMP_RATE
        print(f"\n[tx] File: {args.tx_file}  "
              f"({tx_num_samples:,} samples, {self._tx_duration_s:.3f} s)")
        if args.loop:
            print(f"[tx] Looping enabled")

    def schedule_and_start(self):
        from gnuradio import uhd

        if self._gpsdo:
            # Schedule RX on subframe boundary
            now = self.rx_src.get_time_now().get_real_secs()
            rx_target = math.ceil((now + 0.5) * 1000.0) / 1000.0
            self.rx_src.set_start_time(uhd.time_spec(rx_target))
            self._rx_start_utc = rx_target
            print(f"\n[gpsdo:rx] RX scheduled at t = {rx_target:.6f} s")

            # DFN-aligned TX start
            sidecar = self.tx_sidecar
            if sidecar and "cycle_offset_ms" in sidecar:
                capture_offset_ms = sidecar["cycle_offset_ms"]
                target_ts = _compute_dfn_aligned_start(
                    self.tx_sink, capture_offset_ms)
                self.tx_sink.set_start_time(target_ts)
                print(f"[gpsdo:tx] TX DFN-aligned start scheduled")
            else:
                print("[gpsdo:tx] No DFN sidecar -- TX starts immediately")
        else:
            self._rx_start_utc = time.time()

        self.start()
        print("\n[info] Flowgraph started. TX + RX running...")

    def write_rx_sidecar(self):
        """Write RX capture sidecar with timing and CW info."""
        cw_info = self.tx_sidecar.get("cw_inject", {})
        rx_meta = {
            "rx_center_freq": self._rx_freq,
            "rx_rate": self._actual_rx_rate,
            "rx_offset_hz": self._args.rx_offset,
            "cw_baseband_hz": self.cw_baseband_hz,
            "f_cw_hz": self.f_cw_hz,
            "cw_rf_hz": self.cw_rf_hz,
            "rx_start_utc": self._rx_start_utc,
            "duration_s": self._args.duration,
            "active_subframes": cw_info.get("active_subframes", []),
            "num_active": cw_info.get("num_active", 0),
            "num_total": cw_info.get("num_total", 0),
            "tx_file_duration_s": self._tx_duration_s,
            "tx_sidecar_ref": _sidecar_path(self._args.tx_file),
        }
        write_sidecar(self._args.rx_file, rx_meta)


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # Validate
    if not os.path.isfile(args.tx_file):
        print(f"ERROR: TX file not found: {args.tx_file}")
        return 1

    print("=" * 60)
    print("  C-V2X CW TX/RX -- Simultaneous Replay + Capture")
    print("=" * 60)

    # GPSDO pre-flight
    if args.gpsdo:
        _gpsdo_preflight(args.args, args.gpsdo_timeout)

    # Build and start flowgraph
    tb = CV2XCwTxRx(args)
    tb.schedule_and_start()

    # Run for specified duration
    try:
        remaining = args.duration
        print(f"[info] Capturing for {args.duration:.1f} s "
              f"(Ctrl+C to stop early) ...\n")
        while remaining > 0:
            sleep_time = min(remaining, 5.0)
            time.sleep(sleep_time)
            remaining -= sleep_time
            if remaining > 0:
                print(f"  {remaining:.0f} s remaining ...", end="\r")
        print(f"\n[info] Duration reached ({args.duration:.1f} s).")
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")

    tb.stop()
    tb.wait()

    # Write RX sidecar
    tb.write_rx_sidecar()

    # Summary
    rx_file_size = os.path.getsize(args.rx_file) if os.path.isfile(args.rx_file) else 0
    print(f"\n[done] RX file: {args.rx_file}  ({rx_file_size / 1e6:.2f} MB)")
    print(f"[done] Next: python cv2x_microdoppler_extract.py "
          f"--input {args.rx_file} --plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
