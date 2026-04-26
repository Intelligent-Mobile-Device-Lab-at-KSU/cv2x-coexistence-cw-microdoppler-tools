#!/usr/bin/env python3
"""
cv2x_capture_replay.py -- C-V2X Sidelink Capture & Replay
==========================================================
Capture a 3GPP Rel-14 C-V2X sidelink signal (20 MHz, 5.9 GHz band) from a
commercial radio into a raw IQ file, then replay that capture through a USRP
TX interface.

Two mutually exclusive modes:
  --capture   RX from USRP -> IQ file on disk
  --replay    IQ file -> TX through USRP

DFN-aligned replay (--gpsdo):
  C-V2X sidelink PHY scrambling repeats on a 10.24 s cycle (DFN 0-1023,
  10 subframes each).  When --gpsdo is used:
    * Capture: writes a JSON sidecar (.json) with the DFN/subframe of the
      first sample.  Duration is auto-snapped to 10.24 s multiples.
    * Replay: reads the sidecar and schedules TX start at the exact same
      DFN/subframe coordinate, so the receiver descrambles correctly.

Display:  PyQt5 freq-sink + waterfall (disable with --headless).

-----
# Capture 60 s (auto-snapped to 61.44 s = 6 DFN cycles):
python cv2x_capture_replay.py --capture --gpsdo --headless \\
    --rx-freq 5.915e9 --rx-gain 30 \\
    --rx-args "serial=33767A5,master_clock_rate=184.32e6" \\
    --capture-file cv2x_iq.cf32 --duration 60 -r 30.72e6

# Replay in a loop, DFN-aligned via GPSDO:
python cv2x_capture_replay.py --replay --gpsdo --headless \\
    --tx-freq 5.915e9 --tx-gain 15 \\
    --tx-args "serial=33767A5,master_clock_rate=184.32e6" \\
    --replay-file cv2x_iq.cf32 --loop -r 30.72e6

# Replay once, headless, no GPS sync:
python cv2x_capture_replay.py --replay --headless \\
    --tx-freq 5.915e9 --tx-gain 15 \\
    --tx-args "serial=33767A5,master_clock_rate=184.32e6" \\
    --replay-file cv2x_iq.cf32 --no-loop -r 30.72e6
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

# ---------------------------------------------------------------------------
# Blackman-Harris window constant (portable across GNU Radio versions)
# ---------------------------------------------------------------------------
try:
    from gnuradio.fft import window as grwindow
    WIN_BLACKMAN_HARRIS = grwindow.WIN_BLACKMAN_hARRIS
except (ImportError, AttributeError):
    try:
        from gnuradio.filter import firdes
        WIN_BLACKMAN_HARRIS = firdes.WIN_BLACKMAN_hARRIS
    except (ImportError, AttributeError):
        WIN_BLACKMAN_HARRIS = 5


# ============================================================================
# 3GPP DFN timing constants  (TS 36.211 sidelink scrambling)
# ============================================================================
EPOCH_OFFSET_S = 2208988800   # seconds from 1900-01-01 to Unix epoch 1970-01-01
DFN_CYCLE_MS   = 10240        # 1024 frames x 10 subframes = 10.24 s
DFN_CYCLE_S    = DFN_CYCLE_MS / 1000.0


# ============================================================================
# GPSDO helpers
# ============================================================================
_QUERY_GPSDO_PATHS = [
    os.path.expanduser(r"~\radioconda\Library\lib\uhd\utils"
                       r"\query_gpsdo_sensors.exe"),
    r"C:\Program Files (x86)\National Instruments"
    r"\NI-USRP\utilities\query_gpsdo_sensors.exe",
]


def _gpsdo_preflight(dev_args, timeout=90):
    """Run query_gpsdo_sensors before touching GNU Radio.

    Handles the NI-RIO FPGA-load race: the first invocation after boot
    often fails with "No devices found" while the bitfile is still being
    programmed.  We retry once after a short delay to cover that case.

    Returns True if GPS is locked, False otherwise.  Prints sensor output
    so the operator can see satellite / fix status.
    """
    exe = None
    for p in _QUERY_GPSDO_PATHS:
        if os.path.isfile(p):
            exe = p
            break
    if exe is None:
        print("[gpsdo] query_gpsdo_sensors not found in any known location:\n"
              + "\n".join(f"        {p}" for p in _QUERY_GPSDO_PATHS)
              + "\n[gpsdo] Skipping pre-flight check.")
        return False

    cmd = [exe]
    if dev_args:
        cmd.append(f"--args={dev_args}")

    max_attempts = 2          # first attempt may fail while FPGA loads
    retry_delay  = 10         # seconds to wait before retry

    for attempt in range(1, max_attempts + 1):
        print(f"[gpsdo] Pre-flight attempt {attempt}/{max_attempts} "
              f"(timeout {timeout} s) ...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout)
            output = (result.stdout + result.stderr).strip()

            if result.returncode != 0:
                # Check for the FPGA-not-ready / device-not-found error
                if ("No devices found" in output or
                        "LookupError" in output or
                        "KeyError" in output):
                    if attempt < max_attempts:
                        print(f"[gpsdo]   Device not ready (FPGA loading?). "
                              f"Retrying in {retry_delay} s ...")
                        time.sleep(retry_delay)
                        continue
                # Final attempt or different error -- print everything
                if output:
                    for line in output.splitlines():
                        print(f"[gpsdo]   {line}")
                print(f"[gpsdo] query_gpsdo_sensors exited with code "
                      f"{result.returncode}")
                return False

            # Success -- print output
            if output:
                for line in output.splitlines():
                    print(f"[gpsdo]   {line}")

            # Check for 10 MHz ref lock (we do NOT require GPS nav lock)
            ref_locked = "usrp locked to reference" in output.lower()
            if ref_locked:
                print("[gpsdo] Pre-flight: 10 MHz ref LOCKED "
                      "(GPS nav lock not required)")
            else:
                print("[gpsdo] Pre-flight: 10 MHz ref NOT locked "
                      "(proceeding -- _gpsdo_setup will retry)")
            return ref_locked

        except subprocess.TimeoutExpired:
            print(f"[gpsdo] query_gpsdo_sensors timed out after {timeout} s")
            if attempt < max_attempts:
                continue
            return False
        except OSError as exc:
            print(f"[gpsdo] Could not run query_gpsdo_sensors: {exc}")
            return False

    return False


def _gpsdo_setup(usrp_block, label="usrp", timeout=90):
    """Wait for 10 MHz ref lock and sync USRP time to GPS PPS/UTC.

    We only need the 10 MHz reference to be disciplined and PPS to be
    valid -- a full GPS navigation fix (position lock) is NOT required.

    *usrp_block* is either a uhd.usrp_source or uhd.usrp_sink.
    *timeout* is max seconds to wait for ref lock.
    """
    from gnuradio import uhd

    # Clock / time source should already be set via device args, but
    # set them explicitly in case the caller forgot.
    usrp_block.set_clock_source("gpsdo")
    usrp_block.set_time_source("gpsdo")

    # -- wait for 10 MHz reference lock (NOT GPS nav lock) ----------------
    print(f"[gpsdo:{label}] Waiting for ref_locked (up to {timeout} s) ...")
    ref_locked = False
    for i in range(timeout):
        try:
            ref_locked = usrp_block.get_mboard_sensor("ref_locked").to_bool()
        except Exception:
            ref_locked = False
        if ref_locked:
            break
        time.sleep(1.0)

    if not ref_locked:
        print(f"[gpsdo:{label}] WARNING -- 10 MHz ref did not lock in "
              f"{timeout} s.  Proceeding with free-running clock.")
        return

    print(f"[gpsdo:{label}] 10 MHz reference LOCKED.")

    # -- report GPS nav lock status (informational only) ------------------
    try:
        gps_locked = usrp_block.get_mboard_sensor("gps_locked").to_bool()
        print(f"[gpsdo:{label}] GPS nav lock: "
              f"{'YES' if gps_locked else 'no (not required)'}")
    except Exception:
        pass

    # -- sync USRP time to GPS UTC via PPS --------------------------------
    # PPS is valid whenever the GPSDO is warmed up, even without nav lock.
    gps_time = usrp_block.get_mboard_sensor("gps_time").to_int()
    next_pps = gps_time + 1
    usrp_block.set_time_next_pps(uhd.time_spec(float(next_pps)))
    # Sleep past the PPS edge so the time is latched.
    time.sleep(1.1)
    print(f"[gpsdo:{label}] USRP time set to {next_pps} (UTC via GPS PPS).")


def _schedule_tx_on_subframe(usrp_sink):
    """Insert a tx_time stream tag so the first TX burst is subframe-aligned.

    Unlike usrp_source, usrp_sink does NOT support set_start_time().
    Instead, we attach a tx_time tag at sample offset 0 via the UHD
    sink's 'set_start_time' only if it internally converts to a tag --
    but GNU Radio's UHD sink actually needs a tx_sob / tx_time tag on
    the stream.  The simplest reliable approach: just set the USRP time
    so that t=0 on the device corresponds to a subframe boundary, then
    stream immediately (no timed command).

    Returns the target UTC time as a float.
    """
    now = usrp_sink.get_time_now().get_real_secs()
    target = math.ceil((now + 0.2) * 1000.0) / 1000.0   # next ms boundary
    print(f"[gpsdo:tx] USRP time is {now:.6f} s  "
          f"(next subframe boundary: {target:.6f} s)")
    print(f"[gpsdo:tx] TX will stream immediately -- "
          f"clock is GPS-disciplined, no timed start needed.")
    return target


def _schedule_rx_on_subframe(usrp_source):
    """Schedule the first RX sample on the next 1 ms subframe boundary."""
    from gnuradio import uhd

    now = usrp_source.get_time_now().get_real_secs()
    target = math.ceil((now + 0.5) * 1000.0) / 1000.0
    usrp_source.set_start_time(uhd.time_spec(target))
    print(f"[gpsdo:rx] RX scheduled at t = {target:.6f} s  "
          f"(subframe-aligned)")
    return target


# ============================================================================
# DFN cycle helpers  (3GPP Rel-14 sidelink, TS 36.211)
# ============================================================================
def _gps_to_cycle_offset_ms(gps_utc_secs):
    """Convert GPS/UTC time (Unix epoch seconds) to position (0-10239)
    in the 10.24 s DFN cycle.

    Reference epoch is Jan 1, 1900 00:00:00 UTC per 3GPP TS 36.211.
    """
    ms_since_1900 = int((gps_utc_secs + EPOCH_OFFSET_S) * 1000)
    return ms_since_1900 % DFN_CYCLE_MS


def _cycle_offset_to_dfn_sf(offset_ms):
    """Convert a DFN cycle offset (0-10239) to a (DFN, subframe) tuple."""
    dfn = (offset_ms // 10) % 1024
    sf  = offset_ms % 10
    return dfn, sf


def _compute_dfn_aligned_start(usrp_block, capture_offset_ms, guard_ms=200):
    """Compute the next UTC time when *capture_offset_ms* recurs in the
    DFN cycle and return a uhd.time_spec for that instant.

    The returned time is snapped to an exact millisecond boundary so the
    first TX sample aligns precisely to a 1 ms subframe edge.

    *guard_ms* ensures enough lead time so the timed command is not late.
    """
    from gnuradio import uhd

    now_secs   = usrp_block.get_time_now().get_real_secs()

    # Snap "now" to the current integer-ms to avoid sub-ms drift.
    # _gps_to_cycle_offset_ms already truncates to int ms internally,
    # so we must work in integer ms for consistency.
    now_ms_1900 = int((now_secs + EPOCH_OFFSET_S) * 1000)
    now_offset  = now_ms_1900 % DFN_CYCLE_MS

    wait_ms = (capture_offset_ms - now_offset) % DFN_CYCLE_MS
    if wait_ms < guard_ms:
        wait_ms += DFN_CYCLE_MS          # skip to next cycle

    # Compute target as an exact integer millisecond since 1900,
    # then convert back to a clean UTC float.  Split into whole
    # seconds + fractional ms to avoid floating-point precision loss
    # on the large (trillions) ms-since-1900 value.
    target_ms_1900 = now_ms_1900 + wait_ms
    target_whole_s = target_ms_1900 // 1000 - EPOCH_OFFSET_S
    target_frac_ms = target_ms_1900 % 1000
    target_secs    = float(target_whole_s) + target_frac_ms / 1000.0

    # Verify the target lands on the correct cycle offset
    verify_offset = target_ms_1900 % DFN_CYCLE_MS
    target_dfn, target_sf = _cycle_offset_to_dfn_sf(capture_offset_ms)

    print(f"[dfn] Current cycle offset : {now_offset} ms  "
          f"(DFN={now_offset // 10 % 1024}, SF={now_offset % 10})")
    print(f"[dfn] Target  cycle offset : {capture_offset_ms} ms  "
          f"(DFN={target_dfn}, SF={target_sf})  "
          f"[verify={verify_offset}]")
    print(f"[dfn] Waiting {wait_ms} ms ({wait_ms / 1000.0:.3f} s) "
          f"for DFN-aligned TX start at t = {target_secs:.6f} s")

    if verify_offset != capture_offset_ms:
        print(f"[dfn] ERROR: Offset mismatch! Expected {capture_offset_ms}, "
              f"got {verify_offset}")

    return uhd.time_spec(target_secs)


# ============================================================================
# Capture metadata sidecar  (.json alongside the .cf32)
# ============================================================================
def _sidecar_path(iq_file_path):
    """Return the JSON sidecar path for a given IQ file."""
    base, _ = os.path.splitext(iq_file_path)
    return base + ".json"


def _write_capture_sidecar(iq_file_path, capture_time_utc, cycle_offset_ms,
                           sample_rate, center_freq, duration_s):
    """Write a JSON sidecar file with DFN cycle metadata."""
    dfn, sf = _cycle_offset_to_dfn_sf(cycle_offset_ms)
    meta = {
        "capture_time_utc": capture_time_utc,
        "cycle_offset_ms":  cycle_offset_ms,
        "dfn":              dfn,
        "subframe":         sf,
        "sample_rate":      sample_rate,
        "center_freq":      center_freq,
        "duration_s":       duration_s,
    }
    path = _sidecar_path(iq_file_path)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dfn] Sidecar written : {path}")
    print(f"[dfn]   DFN={dfn}, SF={sf}, cycle_offset={cycle_offset_ms} ms")
    return path


def _read_capture_sidecar(iq_file_path):
    """Read the JSON sidecar file.  Returns dict or None if not found."""
    path = _sidecar_path(iq_file_path)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ============================================================================
# GUI flowgraph  (gr.top_block + Qt.QWidget)
# ============================================================================
class CV2XCaptureReplay(gr.top_block):
    """GUI-enabled capture / replay flowgraph."""

    def __init__(self, args, qt_module, sip_module):
        gr.top_block.__init__(self, "C-V2X Sidelink Capture/Replay")
        Qt = qt_module
        sip = sip_module

        self._widget = Qt.QWidget()
        self._widget.setWindowTitle("C-V2X Sidelink Capture/Replay")
        layout = Qt.QVBoxLayout(self._widget)

        samp_rate = args.rate
        fft_size = args.fft_size
        update_time = args.update_time

        if args.capture:
            self._build_capture(args, samp_rate, fft_size, update_time,
                                layout, Qt, sip)
        else:
            self._build_replay(args, samp_rate, fft_size, update_time,
                               layout, Qt, sip)

    # -- capture --------------------------------------------------------------
    def _build_capture(self, args, samp_rate, fft_size, update_time,
                       layout, Qt, sip):
        from gnuradio import uhd, qtgui

        freq = args.rx_freq
        gain = args.rx_gain
        dev_args = args.rx_args

        print(f"\n[capture] Sample rate : {samp_rate/1e6:.2f} Msps")
        print(f"[capture] Frequency  : {freq/1e9:.4f} GHz")
        print(f"[capture] Gain       : {gain} dB")
        print(f"[capture] Device     : '{dev_args}'")
        print(f"[capture] Output     : {args.capture_file}\n")

        # -- UHD source -------------------------------------------------------
        self.rx_src = uhd.usrp_source(
            dev_args,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )
        self.rx_src.set_samp_rate(samp_rate)
        self.rx_src.set_center_freq(freq)
        self.rx_src.set_gain(gain)
        self.rx_src.set_antenna("RX2")
        self.rx_src.set_auto_iq_balance(True, 0)
        print("[capture] Waiting 2 s for USRP init ...")
        time.sleep(2.0)

        if args.gpsdo:
            _gpsdo_setup(self.rx_src, "rx", args.gpsdo_timeout)
            self._rx_needs_schedule = True
            self._capture_args = args
            self._capture_samp_rate = samp_rate
        else:
            self._rx_needs_schedule = False

        # -- optional head (duration limit) -----------------------------------
        src_block = self.rx_src
        if args.duration is not None:
            n_samples = int(samp_rate * args.duration)
            self.head = blocks.head(gr.sizeof_gr_complex, n_samples)
            self.connect(self.rx_src, self.head)
            src_block = self.head
            print(f"[capture] Duration   : {args.duration:.2f} s  "
                  f"({n_samples} samples)")

        # -- file sink --------------------------------------------------------
        if os.path.exists(args.capture_file):
            print(f"[warn] File '{args.capture_file}' exists -- "
                  "will be overwritten.")
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex, args.capture_file, False)
        self.file_sink.set_unbuffered(False)
        self.connect(src_block, self.file_sink)

        # -- GUI sinks --------------------------------------------------------
        spec_label = (f"RX Spectrum  |  {freq/1e9:.4f} GHz  "
                      f"|  {samp_rate/1e6:.2f} Msps")
        self.rx_fft = qtgui.freq_sink_c(
            fft_size, WIN_BLACKMAN_HARRIS, 0, samp_rate, spec_label)
        self.rx_fft.set_update_time(update_time)
        self.rx_fft.set_y_axis(-120, 0)
        self.rx_fft.enable_autoscale(True)
        self.rx_fft.enable_grid(True)
        layout.addWidget(sip.wrapinstance(self.rx_fft.qwidget(), Qt.QWidget))
        self.connect(src_block, self.rx_fft)

        wfall_label = (f"RX Waterfall  |  {freq/1e9:.4f} GHz")
        self.rx_wfall = qtgui.waterfall_sink_c(
            fft_size, WIN_BLACKMAN_HARRIS, 0, samp_rate, wfall_label)
        self.rx_wfall.set_update_time(update_time)
        self.rx_wfall.set_intensity_range(-120, 0)
        layout.addWidget(
            sip.wrapinstance(self.rx_wfall.qwidget(), Qt.QWidget))
        self.connect(src_block, self.rx_wfall)

    # -- replay ---------------------------------------------------------------
    def _build_replay(self, args, samp_rate, fft_size, update_time,
                      layout, Qt, sip):
        from gnuradio import uhd, qtgui

        freq = args.tx_freq
        gain = args.tx_gain
        dev_args = args.tx_args
        loop = args.loop

        file_size = os.path.getsize(args.replay_file)
        file_dur = file_size / (gr.sizeof_gr_complex * samp_rate)

        print(f"\n[replay] Sample rate : {samp_rate/1e6:.2f} Msps")
        print(f"[replay] Frequency   : {freq/1e9:.4f} GHz")
        print(f"[replay] Gain        : {gain} dB")
        print(f"[replay] Device      : '{dev_args}'")
        print(f"[replay] Input       : {args.replay_file}  "
              f"({file_size/1e6:.1f} MB, ~{file_dur:.2f} s)")
        print(f"[replay] Loop        : {loop}")

        # DFN cycle duration warning for looped replay
        if loop:
            tol = 0.001
            remainder = file_dur % DFN_CYCLE_S
            if remainder > tol and (DFN_CYCLE_S - remainder) > tol:
                print(f"[dfn] WARNING: File duration ({file_dur:.3f} s) "
                      f"is not a multiple of {DFN_CYCLE_S} s.\n"
                      f"      Looped replay will have DFN discontinuities "
                      f"at file boundaries.")
        print()

        # -- file source ------------------------------------------------------
        self.file_src = blocks.file_source(
            gr.sizeof_gr_complex, args.replay_file, loop)

        # -- UHD sink ---------------------------------------------------------
        self.tx_sink = uhd.usrp_sink(
            dev_args,
            uhd.stream_args(cpu_format="fc32",
                            args="send_buff_size=9998336",
                            channels=[0]),
        )
        self.tx_sink.set_samp_rate(samp_rate)
        self.tx_sink.set_center_freq(freq)
        self.tx_sink.set_gain(gain)
        self.tx_sink.set_antenna("TX/RX")
        print("[replay] Waiting 2 s for USRP init ...")
        time.sleep(2.0)

        if args.gpsdo:
            _gpsdo_setup(self.tx_sink, "tx", args.gpsdo_timeout)
            self._tx_needs_schedule = True
            # Load DFN sidecar for DFN-aligned TX
            self._dfn_sidecar = _read_capture_sidecar(args.replay_file)
            if self._dfn_sidecar:
                dfn = self._dfn_sidecar["dfn"]
                sf = self._dfn_sidecar["subframe"]
                co = self._dfn_sidecar["cycle_offset_ms"]
                print(f"[dfn] Sidecar loaded: DFN={dfn}, SF={sf}, "
                      f"cycle_offset={co} ms")
            else:
                print(f"[dfn] WARNING: No sidecar file found for "
                      f"'{args.replay_file}'.\n"
                      f"      TX will NOT be DFN-aligned "
                      f"(subframe-aligned only).")
        else:
            self._tx_needs_schedule = False
            self._dfn_sidecar = None

        self.connect(self.file_src, self.tx_sink)

        # -- GUI sinks (tap off file source via tee) --------------------------
        # Use a null sink as a rate limiter -- GUI sinks are for monitoring
        # only and must not back-pressure the TX path.
        self.tx_tee = blocks.copy(gr.sizeof_gr_complex)
        self.tx_keep = blocks.keep_one_in_n(gr.sizeof_gr_complex,
                                            max(1, int(samp_rate / 30000)))
        self.connect(self.file_src, self.tx_tee, self.tx_keep)

        spec_label = (f"TX Spectrum  |  {freq/1e9:.4f} GHz  "
                      f"|  {samp_rate/1e6:.2f} Msps")
        self.tx_fft = qtgui.freq_sink_c(
            fft_size, WIN_BLACKMAN_HARRIS, 0, samp_rate, spec_label)
        self.tx_fft.set_update_time(update_time)
        self.tx_fft.set_y_axis(-120, 0)
        self.tx_fft.enable_autoscale(True)
        self.tx_fft.enable_grid(True)
        layout.addWidget(sip.wrapinstance(self.tx_fft.qwidget(), Qt.QWidget))
        self.connect(self.tx_keep, self.tx_fft)

        wfall_label = (f"TX Waterfall  |  {freq/1e9:.4f} GHz")
        self.tx_wfall = qtgui.waterfall_sink_c(
            fft_size, WIN_BLACKMAN_HARRIS, 0, samp_rate, wfall_label)
        self.tx_wfall.set_update_time(update_time)
        self.tx_wfall.set_intensity_range(-120, 0)
        layout.addWidget(
            sip.wrapinstance(self.tx_wfall.qwidget(), Qt.QWidget))
        self.connect(self.tx_keep, self.tx_wfall)

    # -- public helpers -------------------------------------------------------
    def schedule_and_start(self):
        """Schedule timed streams (if needed) and start the flowgraph.

        Timed TX/RX must be scheduled immediately before start() so the
        target time is not already past when the flowgraph begins.
        """
        if getattr(self, '_rx_needs_schedule', False):
            rx_start_utc = _schedule_rx_on_subframe(self.rx_src)
            # Write DFN sidecar for later replay alignment
            cycle_offset = _gps_to_cycle_offset_ms(rx_start_utc)
            dfn, sf = _cycle_offset_to_dfn_sf(cycle_offset)
            print(f"[dfn] Capture starts at DFN={dfn}, SF={sf}  "
                  f"(cycle_offset={cycle_offset} ms)")
            args = self._capture_args
            _write_capture_sidecar(
                args.capture_file,
                capture_time_utc=rx_start_utc,
                cycle_offset_ms=cycle_offset,
                sample_rate=self._capture_samp_rate,
                center_freq=args.rx_freq,
                duration_s=args.duration if args.duration is not None else 0.0,
            )
        if getattr(self, '_tx_needs_schedule', False):
            sidecar = getattr(self, '_dfn_sidecar', None)
            if sidecar and 'cycle_offset_ms' in sidecar:
                target_ts = _compute_dfn_aligned_start(
                    self.tx_sink, sidecar['cycle_offset_ms'])
                self.tx_sink.set_start_time(target_ts)
            else:
                _schedule_tx_on_subframe(self.tx_sink)
        self.start()

    @property
    def widget(self):
        return self._widget

    def show(self):
        self._widget.show()


# ============================================================================
# Headless flowgraph  (gr.top_block only -- no Qt)
# ============================================================================
class CV2XCaptureReplayHeadless(gr.top_block):
    """Headless capture / replay flowgraph (no GUI dependencies)."""

    def __init__(self, args):
        gr.top_block.__init__(self, "C-V2X Sidelink Capture/Replay")

        samp_rate = args.rate

        if args.capture:
            self._build_capture(args, samp_rate)
        else:
            self._build_replay(args, samp_rate)

    def _build_capture(self, args, samp_rate):
        from gnuradio import uhd

        freq = args.rx_freq
        gain = args.rx_gain
        dev_args = args.rx_args

        print(f"\n[capture] Sample rate : {samp_rate/1e6:.2f} Msps")
        print(f"[capture] Frequency  : {freq/1e9:.4f} GHz")
        print(f"[capture] Gain       : {gain} dB")
        print(f"[capture] Device     : '{dev_args}'")
        print(f"[capture] Output     : {args.capture_file}")
        print(f"[capture] Headless   : yes\n")

        self.rx_src = uhd.usrp_source(
            dev_args,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )
        self.rx_src.set_samp_rate(samp_rate)
        self.rx_src.set_center_freq(freq)
        self.rx_src.set_gain(gain)
        self.rx_src.set_antenna("RX2")
        self.rx_src.set_auto_iq_balance(True, 0)
        print("[capture] Waiting 2 s for USRP init ...")
        time.sleep(2.0)

        if args.gpsdo:
            _gpsdo_setup(self.rx_src, "rx", args.gpsdo_timeout)
            self._rx_needs_schedule = True
            self._capture_args = args
            self._capture_samp_rate = samp_rate
        else:
            self._rx_needs_schedule = False

        src_block = self.rx_src
        if args.duration is not None:
            n_samples = int(samp_rate * args.duration)
            self.head = blocks.head(gr.sizeof_gr_complex, n_samples)
            self.connect(self.rx_src, self.head)
            src_block = self.head
            print(f"[capture] Duration   : {args.duration:.2f} s  "
                  f"({n_samples} samples)")

        if os.path.exists(args.capture_file):
            print(f"[warn] File '{args.capture_file}' exists -- "
                  "will be overwritten.")
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex, args.capture_file, False)
        self.file_sink.set_unbuffered(False)
        self.connect(src_block, self.file_sink)

    def _build_replay(self, args, samp_rate):
        from gnuradio import uhd

        freq = args.tx_freq
        gain = args.tx_gain
        dev_args = args.tx_args
        loop = args.loop

        file_size = os.path.getsize(args.replay_file)
        file_dur = file_size / (gr.sizeof_gr_complex * samp_rate)

        print(f"\n[replay] Sample rate : {samp_rate/1e6:.2f} Msps")
        print(f"[replay] Frequency   : {freq/1e9:.4f} GHz")
        print(f"[replay] Gain        : {gain} dB")
        print(f"[replay] Device      : '{dev_args}'")
        print(f"[replay] Input       : {args.replay_file}  "
              f"({file_size/1e6:.1f} MB, ~{file_dur:.2f} s)")
        print(f"[replay] Loop        : {loop}")
        print(f"[replay] Headless    : yes")

        # DFN cycle duration warning for looped replay
        if loop:
            tol = 0.001
            remainder = file_dur % DFN_CYCLE_S
            if remainder > tol and (DFN_CYCLE_S - remainder) > tol:
                print(f"[dfn] WARNING: File duration ({file_dur:.3f} s) "
                      f"is not a multiple of {DFN_CYCLE_S} s.\n"
                      f"      Looped replay will have DFN discontinuities "
                      f"at file boundaries.")
        print()

        self.file_src = blocks.file_source(
            gr.sizeof_gr_complex, args.replay_file, loop)

        self.tx_sink = uhd.usrp_sink(
            dev_args,
            uhd.stream_args(cpu_format="fc32",
                            args="send_buff_size=9998336",
                            channels=[0]),
        )
        self.tx_sink.set_samp_rate(samp_rate)
        self.tx_sink.set_center_freq(freq)
        self.tx_sink.set_gain(gain)
        self.tx_sink.set_antenna("TX/RX")
        print("[replay] Waiting 2 s for USRP init ...")
        time.sleep(2.0)

        if args.gpsdo:
            _gpsdo_setup(self.tx_sink, "tx", args.gpsdo_timeout)
            self._tx_needs_schedule = True
            # Load DFN sidecar for DFN-aligned TX
            self._dfn_sidecar = _read_capture_sidecar(args.replay_file)
            if self._dfn_sidecar:
                dfn = self._dfn_sidecar["dfn"]
                sf = self._dfn_sidecar["subframe"]
                co = self._dfn_sidecar["cycle_offset_ms"]
                print(f"[dfn] Sidecar loaded: DFN={dfn}, SF={sf}, "
                      f"cycle_offset={co} ms")
            else:
                print(f"[dfn] WARNING: No sidecar file found for "
                      f"'{args.replay_file}'.\n"
                      f"      TX will NOT be DFN-aligned "
                      f"(subframe-aligned only).")
        else:
            self._tx_needs_schedule = False
            self._dfn_sidecar = None

        self.connect(self.file_src, self.tx_sink)

    def schedule_and_start(self):
        """Schedule timed streams (if needed) and start the flowgraph."""
        if getattr(self, '_rx_needs_schedule', False):
            rx_start_utc = _schedule_rx_on_subframe(self.rx_src)
            cycle_offset = _gps_to_cycle_offset_ms(rx_start_utc)
            dfn, sf = _cycle_offset_to_dfn_sf(cycle_offset)
            print(f"[dfn] Capture starts at DFN={dfn}, SF={sf}  "
                  f"(cycle_offset={cycle_offset} ms)")
            args = self._capture_args
            _write_capture_sidecar(
                args.capture_file,
                capture_time_utc=rx_start_utc,
                cycle_offset_ms=cycle_offset,
                sample_rate=self._capture_samp_rate,
                center_freq=args.rx_freq,
                duration_s=args.duration if args.duration is not None else 0.0,
            )
        if getattr(self, '_tx_needs_schedule', False):
            sidecar = getattr(self, '_dfn_sidecar', None)
            if sidecar and 'cycle_offset_ms' in sidecar:
                target_ts = _compute_dfn_aligned_start(
                    self.tx_sink, sidecar['cycle_offset_ms'])
                self.tx_sink.set_start_time(target_ts)
            else:
                _schedule_tx_on_subframe(self.tx_sink)
        self.start()


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="C-V2X sidelink capture & replay (20 MHz, 5.9 GHz band)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true",
                      help="Capture mode: RX from USRP -> IQ file")
    mode.add_argument("--replay", action="store_true",
                      help="Replay mode: IQ file -> TX through USRP")

    # -- common ---------------------------------------------------------------
    p.add_argument("-r", "--rate", type=float, default=25e6, metavar="HZ",
                   help="Sample rate (Hz). Default 25 Msps gives clean "
                        "decimation (200 MHz / 8) on X310 and covers the "
                        "20 MHz C-V2X channel.")
    p.add_argument("--headless", action="store_true",
                   help="Disable GUI (for SSH / no-display use)")
    p.add_argument("--fft-size", type=int, default=2048, metavar="N",
                   help="FFT size for GUI spectrum/waterfall")
    p.add_argument("--update-time", type=float, default=0.05, metavar="S",
                   help="GUI display refresh interval (s)")
    p.add_argument("--gpsdo", action="store_true",
                   help="Enable GPSDO sync.  Capture: writes DFN metadata "
                        "sidecar (.json) and auto-snaps duration to 10.24 s "
                        "multiples.  Replay: DFN-aligns TX start for "
                        "correct descrambling by commercial radios.")
    p.add_argument("--gpsdo-timeout", type=int, default=90, metavar="SEC",
                   help="Max seconds to wait for GPS lock before proceeding")

    # -- capture --------------------------------------------------------------
    cap = p.add_argument_group("capture options")
    cap.add_argument("--rx-freq", type=float, default=5.915e9, metavar="HZ",
                     help="RX center frequency (Hz)")
    cap.add_argument("--rx-gain", type=float, default=30, metavar="dB",
                     help="RX gain (dB)")
    cap.add_argument("--rx-args", type=str, default="", metavar="ARGS",
                     help="UHD device args for RX")
    cap.add_argument("--capture-file", type=str, default="capture.cf32",
                     metavar="FILE",
                     help="Output IQ file (raw complex64)")
    cap.add_argument("--duration", type=float, default=None, metavar="SEC",
                     help="Capture duration in seconds (default: infinite). "
                          "With --gpsdo, auto-snapped to nearest 10.24 s "
                          "multiple (one full DFN cycle) for seamless "
                          "looped replay.")

    # -- replay ---------------------------------------------------------------
    rep = p.add_argument_group("replay options")
    rep.add_argument("--tx-freq", type=float, default=5.915e9, metavar="HZ",
                     help="TX center frequency (Hz)")
    rep.add_argument("--tx-gain", type=float, default=15, metavar="dB",
                     help="TX gain (dB)")
    rep.add_argument("--tx-args", type=str, default="", metavar="ARGS",
                     help="UHD device args for TX")
    rep.add_argument("--replay-file", type=str, default=None, metavar="FILE",
                     help="Input IQ file for replay (raw complex64)")
    loop_grp = rep.add_mutually_exclusive_group()
    loop_grp.add_argument("--loop", action="store_true", default=True,
                          help="Loop file playback continuously")
    loop_grp.add_argument("--no-loop", action="store_false", dest="loop",
                          help="Play file once then stop")

    return p.parse_args()


def validate_args(args):
    """Sanity-check arguments before building the flowgraph."""
    if args.replay:
        if args.replay_file is None:
            print("[error] --replay-file is required in replay mode.")
            sys.exit(1)
        if not os.path.isfile(args.replay_file):
            print(f"[error] Replay file not found: {args.replay_file}")
            sys.exit(1)
        fsize = os.path.getsize(args.replay_file)
        if fsize == 0:
            print(f"[error] Replay file is empty: {args.replay_file}")
            sys.exit(1)
        if fsize % gr.sizeof_gr_complex != 0:
            print(f"[warn] File size ({fsize} bytes) is not a multiple of "
                  f"{gr.sizeof_gr_complex} -- file may be truncated.")
        print(f"[info] Replay file: {args.replay_file}  "
              f"({fsize/1e6:.1f} MB, "
              f"~{fsize / (gr.sizeof_gr_complex * args.rate):.2f} s "
              f"@ {args.rate/1e6:.2f} Msps)")

    if args.capture and args.duration is not None and args.duration <= 0:
        print("[error] --duration must be positive.")
        sys.exit(1)

    # Auto-snap capture duration to nearest full DFN cycle (10.24 s multiple)
    if args.capture and args.gpsdo and args.duration is not None:
        cycles = math.ceil(args.duration / DFN_CYCLE_S)
        snapped = cycles * DFN_CYCLE_S
        if abs(snapped - args.duration) > 0.001:
            print(f"[dfn] Adjusting capture duration {args.duration:.2f} s "
                  f"-> {snapped:.2f} s ({cycles} full DFN cycle(s))")
            args.duration = snapped

    if args.capture and args.gpsdo and args.duration is None:
        print("[dfn] NOTE: No --duration set.  For seamless looped "
              "DFN-aligned replay, use --duration N (auto-snapped to "
              "10.24 s multiples).")

    if args.replay and args.duration is not None:
        print("[warn] --duration is ignored in replay mode.")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    validate_args(args)

    # Pre-flight GPSDO check (runs *before* GNU Radio opens the device)
    if args.gpsdo:
        dev = args.rx_args if args.capture else args.tx_args
        _gpsdo_preflight(dev, timeout=args.gpsdo_timeout)

    if args.headless:
        tb = CV2XCaptureReplayHeadless(args)
        tb.schedule_and_start()
        print("[info] Flowgraph running. Press Ctrl+C to stop.")
        try:
            if args.capture and args.duration is not None:
                tb.wait()
                print("[info] Capture complete.")
            elif args.replay and not args.loop:
                tb.wait()
                print("[info] Replay complete.")
            else:
                while True:
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[info] Interrupted.")
        tb.stop()
        tb.wait()
    else:
        from PyQt5 import Qt
        import sip
        from gnuradio import qtgui  # noqa: F401 -- ensure qtgui is loaded

        qapp = Qt.QApplication(sys.argv)
        tb = CV2XCaptureReplay(args, Qt, sip)

        # Allow Ctrl+C to kill the process instead of being eaten by Qt.
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        tb.show()
        tb.schedule_and_start()

        # For non-looping replay, quit the app when the flowgraph finishes.
        if args.replay and not args.loop:
            import threading

            def _wait_and_quit():
                tb.wait()
                print("[info] Replay complete.")
                qapp.quit()

            t = threading.Thread(target=_wait_and_quit, daemon=True)
            t.start()

        # For duration-limited capture, quit when done.
        if args.capture and args.duration is not None:
            import threading

            def _wait_and_quit():
                tb.wait()
                print("[info] Capture complete.")
                qapp.quit()

            t = threading.Thread(target=_wait_and_quit, daemon=True)
            t.start()

        qapp.exec_()
        tb.stop()
        tb.wait()


if __name__ == "__main__":
    main()
