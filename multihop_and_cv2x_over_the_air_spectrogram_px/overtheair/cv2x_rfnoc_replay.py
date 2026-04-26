#!/usr/bin/env python3
"""
cv2x_rfnoc_replay.py -- C-V2X Sidelink TX via RFNoC Replay Block
=================================================================
Uploads a raw IQ file into the X310's FPGA DRAM via the RFNoC Replay Block,
then plays it back directly through the Radio block with zero host involvement
during TX.  This eliminates underflows and guarantees sample-accurate 1 ms
subframe timing required by 3GPP Rel-14 C-V2X sidelink.

Two phases (can be combined in one invocation):
  --upload   Load IQ file (cf32) from disk into Replay DRAM
  --play     Play from DRAM -> DUC -> Radio TX

Optional GPSDO synchronisation (--gpsdo) aligns the first TX sample to the
1 ms subframe grid used by C-V2X MACs (UTC-synchronised via GNSS).

Dependencies: uhd (UHD 4.x with RFNoC), numpy.  No GNU Radio required.

-----
# Upload + play in a loop with GPSDO sync:
python cv2x_rfnoc_replay.py --upload --play --gpsdo \\
    --file cv2x_17s.cf32 -r 30.72e6 \\
    --tx-freq 5.915e9 --tx-gain 15 \\
    --args "serial=33767A5,master_clock_rate=184.32e6" --loop

# Upload only (preload DRAM for later play):
python cv2x_rfnoc_replay.py --upload \\
    --file cv2x_17s.cf32 -r 30.72e6 \\
    --args "serial=33767A5,master_clock_rate=184.32e6"

# Play only (DRAM already loaded in same power cycle):
python cv2x_rfnoc_replay.py --play --gpsdo \\
    --file cv2x_17s.cf32 -r 30.72e6 \\
    --tx-freq 5.915e9 --tx-gain 15 \\
    --args "serial=33767A5,master_clock_rate=184.32e6" --loop
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

# Wire format: sc16 = 4 bytes per sample on FPGA
BYTES_PER_SC16 = 4
# Host format: fc32 = 8 bytes per sample in .cf32 files
BYTES_PER_FC32 = 8
# Upload chunk size (samples) -- ~1 ms worth at 30.72 Msps
UPLOAD_CHUNK = 32768


# ============================================================================
# GPSDO helpers  (preflight reused from cv2x_capture_replay.py)
# ============================================================================
_QUERY_GPSDO_PATHS = [
    os.path.expanduser(r"~\radioconda\Library\lib\uhd\utils"
                       r"\query_gpsdo_sensors.exe"),
    r"C:\Program Files (x86)\National Instruments"
    r"\NI-USRP\utilities\query_gpsdo_sensors.exe",
]


def _gpsdo_preflight(dev_args, timeout=90):
    """Run query_gpsdo_sensors before touching UHD.

    Handles the NI-RIO FPGA-load race: the first invocation after boot
    often fails with "No devices found" while the bitfile is still being
    programmed.  We retry once after a short delay to cover that case.
    """
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
        print(f"[gpsdo] Pre-flight attempt {attempt}/2 "
              f"(timeout {timeout} s) ...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout)
            output = (result.stdout + result.stderr).strip()

            if result.returncode != 0:
                if ("No devices found" in output or
                        "LookupError" in output or
                        "KeyError" in output):
                    if attempt < 2:
                        print("[gpsdo]   Device not ready (FPGA loading?). "
                              "Retrying in 10 s ...")
                        time.sleep(10)
                        continue
                if output:
                    for line in output.splitlines():
                        print(f"[gpsdo]   {line}")
                print(f"[gpsdo] query_gpsdo_sensors exited with code "
                      f"{result.returncode}")
                return False

            if output:
                for line in output.splitlines():
                    print(f"[gpsdo]   {line}")
            ref_locked = "usrp locked to reference" in output.lower()
            if ref_locked:
                print("[gpsdo] Pre-flight: 10 MHz ref LOCKED")
            else:
                print("[gpsdo] Pre-flight: 10 MHz ref NOT locked "
                      "(will retry in _gpsdo_setup)")
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


def _gpsdo_setup_rfnoc(mbc, label="mb0", timeout=90):
    """Wait for 10 MHz ref lock and sync USRP time to GPS PPS/UTC.

    Operates on an MBController (RFNoC API) instead of a GNU Radio UHD block.
    Only requires 10 MHz reference discipline + PPS -- GPS nav lock is NOT
    required.
    """
    import uhd

    mbc.set_clock_source("gpsdo")
    mbc.set_time_source("gpsdo")

    print(f"[gpsdo:{label}] Waiting for ref_locked (up to {timeout} s) ...")
    ref_locked = False
    for _ in range(timeout):
        try:
            ref_locked = mbc.get_sensor("ref_locked").to_bool()
        except Exception:
            ref_locked = False
        if ref_locked:
            break
        time.sleep(1.0)

    if not ref_locked:
        print(f"[gpsdo:{label}] WARNING -- 10 MHz ref did not lock in "
              f"{timeout} s.  Proceeding with free-running clock.")
        return False

    print(f"[gpsdo:{label}] 10 MHz reference LOCKED.")

    # Informational only -- GPS nav lock is NOT required
    try:
        gps_locked = mbc.get_sensor("gps_locked").to_bool()
        print(f"[gpsdo:{label}] GPS nav lock: "
              f"{'YES' if gps_locked else 'no (not required)'}")
    except Exception:
        pass

    # Sync USRP time to GPS UTC via PPS
    gps_time = mbc.get_sensor("gps_time").to_int()
    next_pps = gps_time + 1
    tk = mbc.get_timekeeper(0)
    tk.set_time_next_pps(uhd.types.TimeSpec(float(next_pps)))
    time.sleep(1.1)  # wait past PPS edge
    now = tk.get_time_now().get_real_secs()
    print(f"[gpsdo:{label}] USRP time set to {next_pps} (UTC via GPS PPS). "
          f"Current: {now:.6f} s")
    return True


def _compute_subframe_start(tk, guard_ms=5):
    """Return a TimeSpec aligned to the next 1 ms subframe boundary.

    Adds guard_ms milliseconds to account for command latency.
    """
    import uhd

    now = tk.get_time_now().get_real_secs()
    target = math.ceil((now + guard_ms / 1000.0) * 1000.0) / 1000.0
    print(f"[gpsdo:tx] USRP time is {now:.6f} s  "
          f"(subframe-aligned start: {target:.6f} s)")
    return uhd.types.TimeSpec(target)


def _align_to_word(nbytes, word_size):
    """Round nbytes DOWN to the nearest multiple of word_size."""
    return (nbytes // word_size) * word_size


# ============================================================================
# Upload: host file -> Replay DRAM
# ============================================================================
def upload_to_replay(args):
    """Stream IQ data from a .cf32 file into the Replay block's DRAM.

    Returns (playback_size_bytes, replay_block_id_str).
    """
    import uhd

    print("\n[upload] Opening RfnocGraph ...")
    graph = uhd.rfnoc.RfnocGraph(args.args)

    # -- find Replay block ----------------------------------------------------
    replay_blocks = graph.find_blocks("Replay")
    if not replay_blocks:
        print("[error] No Replay block found on FPGA.  The HG image may not "
              "include one.\n        You may need a custom FPGA image with "
              "the Replay block enabled.")
        sys.exit(1)
    replay_blk_id = replay_blocks[0]
    replay = uhd.rfnoc.ReplayBlockControl(graph.get_block(replay_blk_id))

    mem_size = replay.get_mem_size()
    word_size = replay.get_word_size()
    print(f"[upload] Replay block : {replay_blk_id}")
    print(f"[upload] DRAM         : {mem_size / (1024**2):.1f} MB  "
          f"(word size: {word_size} B)")

    # -- compute sizes --------------------------------------------------------
    file_size = os.path.getsize(args.file)
    n_samples = file_size // BYTES_PER_FC32
    sc16_size = n_samples * BYTES_PER_SC16
    sc16_size = _align_to_word(sc16_size, word_size)

    if sc16_size > mem_size:
        max_sc16 = _align_to_word(mem_size, word_size)
        max_samples = max_sc16 // BYTES_PER_SC16
        max_fc32_bytes = max_samples * BYTES_PER_FC32
        max_dur = max_samples / args.rate
        print(f"\n[error] File requires {sc16_size / (1024**2):.1f} MB as "
              f"sc16, but Replay DRAM is only {mem_size / (1024**2):.1f} MB.")
        print(f"        Max duration at {args.rate/1e6:.2f} Msps: "
              f"{max_dur:.2f} s  ({max_samples} samples)")
        print(f"        Truncate the file to fit:")
        trunc_file = args.file.replace(".cf32", f"_{max_dur:.0f}s.cf32")
        print(f'          python -c "open(\'{trunc_file}\',\'wb\').write('
              f'open(\'{args.file}\',\'rb\').read({max_fc32_bytes}))"')
        sys.exit(1)

    file_dur = n_samples / args.rate
    print(f"[upload] File         : {args.file}  "
          f"({file_size / (1024**2):.1f} MB fc32, "
          f"{sc16_size / (1024**2):.1f} MB sc16, "
          f"~{file_dur:.2f} s)")

    # -- create tx_streamer -> Replay -----------------------------------------
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    tx_streamer = graph.create_tx_streamer(1, stream_args)
    graph.connect(tx_streamer, 0, replay_blk_id, 0)
    graph.commit()

    # -- arm recording --------------------------------------------------------
    replay.set_record_type("sc16", 0)
    replay.record(0, sc16_size, 0)
    print(f"[upload] Recording armed: offset=0, size={sc16_size} B")

    # -- stream file data -----------------------------------------------------
    tx_md = uhd.types.TXMetadata()
    tx_md.has_time_spec = False

    total_sent = 0
    last_report = 0
    report_interval = max(1, n_samples // 10)  # every ~10%

    print(f"[upload] Streaming {n_samples} samples to Replay DRAM ...")
    t0 = time.monotonic()

    with open(args.file, "rb") as fh:
        while total_sent < n_samples:
            remaining = n_samples - total_sent
            chunk_n = min(UPLOAD_CHUNK, remaining)
            data = np.fromfile(fh, dtype=np.complex64, count=chunk_n)
            if len(data) == 0:
                break

            # Mark end-of-burst on final chunk
            if total_sent + len(data) >= n_samples:
                tx_md.end_of_burst = True

            sent = tx_streamer.send(data, tx_md, 5.0)
            total_sent += sent

            if total_sent - last_report >= report_interval:
                pct = 100.0 * total_sent / n_samples
                print(f"[upload]   {pct:5.1f}%  "
                      f"({total_sent}/{n_samples} samples)")
                last_report = total_sent

    elapsed = time.monotonic() - t0
    print(f"[upload] Sent {total_sent} samples in {elapsed:.1f} s  "
          f"({total_sent * BYTES_PER_FC32 / elapsed / (1024**2):.1f} MB/s)")

    # -- wait for DRAM to fill ------------------------------------------------
    print("[upload] Waiting for Replay DRAM to fill ...")
    timeout_t = time.monotonic() + 15.0
    last_fullness = 0
    stall_t = time.monotonic()

    while True:
        fullness = replay.get_record_fullness(0)
        if fullness >= sc16_size:
            break
        if fullness != last_fullness:
            last_fullness = fullness
            stall_t = time.monotonic()
        if time.monotonic() - stall_t > 10.0:
            print(f"[error] Upload stalled at {fullness}/{sc16_size} bytes.")
            sys.exit(1)
        if time.monotonic() > timeout_t:
            print(f"[error] Timeout waiting for DRAM fill "
                  f"({fullness}/{sc16_size} bytes).")
            sys.exit(1)
        time.sleep(0.1)

    print(f"[upload] DRAM loaded: {replay.get_record_fullness(0)} / "
          f"{sc16_size} bytes.  Done.")

    replay_id_str = str(replay_blk_id)
    del graph  # release RfnocGraph session
    return sc16_size, replay_id_str


# ============================================================================
# Play: Replay DRAM -> Radio TX
# ============================================================================
def play_from_replay(args, playback_size):
    """Connect Replay -> DUC -> Radio and start playback."""
    import uhd

    print("\n[play] Opening RfnocGraph ...")
    graph = uhd.rfnoc.RfnocGraph(args.args)

    # -- find blocks ----------------------------------------------------------
    replay_blocks = graph.find_blocks("Replay")
    if not replay_blocks:
        print("[error] No Replay block found on FPGA.")
        sys.exit(1)
    replay_blk_id = replay_blocks[0]
    replay = uhd.rfnoc.ReplayBlockControl(graph.get_block(replay_blk_id))

    radio_blocks = graph.find_blocks("Radio")
    if not radio_blocks:
        print("[error] No Radio block found on FPGA.")
        sys.exit(1)
    radio_blk_id = radio_blocks[0]
    radio = uhd.rfnoc.RadioControl(graph.get_block(radio_blk_id))

    print(f"[play] Replay : {replay_blk_id}")
    print(f"[play] Radio  : {radio_blk_id}")

    # -- connect Replay -> [DUC ->] Radio ------------------------------------
    try:
        chain = uhd.rfnoc.connect_through_blocks(
            graph,
            str(replay_blk_id), 0,
            str(radio_blk_id), 0)
        block_names = [str(e.dst_blockid) for e in chain]
        print(f"[play] RFNoC chain: {' -> '.join(block_names)}")
    except Exception as exc:
        print(f"[play] connect_through_blocks failed: {exc}")
        print("[play] Trying direct Replay -> Radio connection ...")
        graph.connect(replay_blk_id, 0, radio_blk_id, 0)

    # -- configure Radio ------------------------------------------------------
    radio.set_tx_frequency(args.tx_freq, 0)
    radio.set_tx_gain(args.tx_gain, 0)
    radio.set_tx_antenna(args.antenna, 0)

    # Set rate -- try DUC first, fall back to Radio
    duc_blocks = graph.find_blocks("DUC")
    if duc_blocks:
        duc = uhd.rfnoc.DucBlockControl(graph.get_block(duc_blocks[0]))
        duc.set_input_rate(args.rate, 0)
        print(f"[play] DUC input rate: {args.rate/1e6:.2f} Msps")
    else:
        radio.set_rate(args.rate)

    actual_rate = radio.get_rate()
    print(f"[play] Radio rate   : {actual_rate/1e6:.2f} Msps")
    print(f"[play] TX frequency : {radio.get_tx_frequency(0)/1e9:.4f} GHz")
    print(f"[play] TX gain      : {radio.get_tx_gain(0)} dB")
    print(f"[play] TX antenna   : {radio.get_tx_antenna(0)}")

    # -- commit graph ---------------------------------------------------------
    graph.commit()

    # -- configure Replay playback --------------------------------------------
    replay.set_play_type("sc16", 0)

    play_dur = playback_size / (BYTES_PER_SC16 * args.rate)
    print(f"\n[play] Playback size : {playback_size} bytes  "
          f"(~{play_dur:.2f} s)")
    print(f"[play] Loop          : {args.loop}")

    # -- GPSDO setup ----------------------------------------------------------
    time_spec = uhd.types.TimeSpec(0.0)
    if args.gpsdo:
        mbc = graph.get_mb_controller()
        _gpsdo_setup_rfnoc(mbc, label="tx", timeout=args.gpsdo_timeout)
        tk = mbc.get_timekeeper(0)
        time_spec = _compute_subframe_start(tk, guard_ms=5)
        print(f"[play] Timed start   : {time_spec.get_real_secs():.6f} s")
    else:
        print("[play] Timed start   : immediate (no GPSDO)")

    # -- start playback -------------------------------------------------------
    global _replay_ctrl, _graph_ref
    _replay_ctrl = replay
    _graph_ref = graph

    replay.play(0, playback_size, 0, time_spec, args.loop)
    print("[play] Playback started.  Press Ctrl+C to stop.\n")

    try:
        if args.loop:
            while True:
                time.sleep(1.0)
        else:
            time.sleep(play_dur + 0.5)
            print("[play] Single-shot playback complete.")
    except KeyboardInterrupt:
        print("\n[play] Interrupted.")
    finally:
        replay.stop(0)
        print("[play] Replay stopped.")


# ============================================================================
# Signal handler
# ============================================================================
_replay_ctrl = None
_graph_ref = None


def _sigint_handler(sig, frame):
    if _replay_ctrl is not None:
        print("\n[info] Stopping replay ...")
        try:
            _replay_ctrl.stop(0)
        except Exception:
            pass
    sys.exit(0)


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="C-V2X sidelink TX via RFNoC Replay Block "
                    "(zero host-side streaming)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--upload", action="store_true",
                   help="Upload IQ file to Replay DRAM")
    p.add_argument("--play", action="store_true",
                   help="Play from Replay DRAM to Radio TX")

    p.add_argument("--file", type=str, default=None, metavar="FILE",
                   help="IQ file (raw complex64 / .cf32)")
    p.add_argument("-r", "--rate", type=float, default=30.72e6, metavar="HZ",
                   help="Sample rate (Hz)")
    p.add_argument("--args", type=str,
                   default="serial=33767A5,master_clock_rate=184.32e6",
                   metavar="ARGS",
                   help="UHD device args")

    p.add_argument("--tx-freq", type=float, default=5.915e9, metavar="HZ",
                   help="TX center frequency (Hz)")
    p.add_argument("--tx-gain", type=float, default=15, metavar="dB",
                   help="TX gain (dB)")
    p.add_argument("--antenna", type=str, default="TX/RX",
                   help="TX antenna port")

    p.add_argument("--gpsdo", action="store_true",
                   help="Enable GPSDO for subframe-aligned timed TX")
    p.add_argument("--gpsdo-timeout", type=int, default=90, metavar="SEC",
                   help="Max seconds to wait for 10 MHz ref lock")

    loop_grp = p.add_mutually_exclusive_group()
    loop_grp.add_argument("--loop", action="store_true", default=True,
                          help="Continuous loop playback")
    loop_grp.add_argument("--no-loop", action="store_false", dest="loop",
                          help="Single-shot playback")

    return p.parse_args()


def validate_args(args):
    if not args.upload and not args.play:
        print("[error] Specify --upload, --play, or both.")
        sys.exit(1)

    if args.upload:
        if args.file is None:
            print("[error] --file is required for --upload.")
            sys.exit(1)
        if not os.path.isfile(args.file):
            print(f"[error] File not found: {args.file}")
            sys.exit(1)
        fsize = os.path.getsize(args.file)
        if fsize == 0:
            print(f"[error] File is empty: {args.file}")
            sys.exit(1)
        if fsize % BYTES_PER_FC32 != 0:
            print(f"[warn] File size ({fsize} B) is not a multiple of "
                  f"{BYTES_PER_FC32} -- may be truncated.")
        n_samp = fsize // BYTES_PER_FC32
        dur = n_samp / args.rate
        print(f"[info] File: {args.file}  ({fsize / (1024**2):.1f} MB, "
              f"{n_samp} samples, ~{dur:.2f} s @ {args.rate/1e6:.2f} Msps)")

    if args.play and not args.upload and args.file is None:
        print("[warn] --play without --upload and no --file: will assume "
              "DRAM was loaded in a prior session.  Specify --file to "
              "compute playback size.")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    validate_args(args)

    signal.signal(signal.SIGINT, _sigint_handler)

    if args.gpsdo:
        _gpsdo_preflight(args.args, timeout=args.gpsdo_timeout)

    playback_size = None

    # -- Upload phase ---------------------------------------------------------
    if args.upload:
        playback_size, _ = upload_to_replay(args)

    # -- Play phase -----------------------------------------------------------
    if args.play:
        if playback_size is None:
            # Compute expected playback size from file if available
            if args.file and os.path.isfile(args.file):
                fsize = os.path.getsize(args.file)
                n_samples = fsize // BYTES_PER_FC32
                # We need to know word_size; open a quick graph to check
                import uhd
                g = uhd.rfnoc.RfnocGraph(args.args)
                rb = g.find_blocks("Replay")
                if not rb:
                    print("[error] No Replay block found on FPGA.")
                    sys.exit(1)
                r = uhd.rfnoc.ReplayBlockControl(g.get_block(rb[0]))
                ws = r.get_word_size()
                playback_size = _align_to_word(
                    n_samples * BYTES_PER_SC16, ws)
                del g
            else:
                print("[warn] No file specified; playing entire DRAM.")
                import uhd
                g = uhd.rfnoc.RfnocGraph(args.args)
                rb = g.find_blocks("Replay")
                if not rb:
                    print("[error] No Replay block found on FPGA.")
                    sys.exit(1)
                r = uhd.rfnoc.ReplayBlockControl(g.get_block(rb[0]))
                playback_size = r.get_mem_size()
                del g

        play_from_replay(args, playback_size)

    print("[done]")


if __name__ == "__main__":
    main()
