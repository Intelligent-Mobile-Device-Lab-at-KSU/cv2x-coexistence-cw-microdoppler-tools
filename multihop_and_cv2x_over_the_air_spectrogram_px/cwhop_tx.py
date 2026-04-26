#!/usr/bin/env python3
"""
cwhop_tx.py -- Config-Driven Hopped CW Transmitter
====================================================
Reads cwhopping_config.json (or a user-specified config), builds a
phase-continuous frequency-hopping CW sensing signal, optionally combines it
with SC-FDMA coexistence interference, writes the result to a .cf32 file,
and optionally streams it to UHD hardware via GNU Radio while simultaneously
capturing on the RX port.

The hop schedule is realised by a global NCO that NEVER resets its phase --
this is the critical invariant that makes the de-hopping reconstruct the
micro-Doppler butterfly pattern without spectral artefacts.

Usage examples
--------------
  # Generate only (no hardware):
  python cwhop_tx.py --config cwhopping_config.json --output my_capture \\
      --duration 60 --generate-only

  # Generate + TX/RX with GPSDO:
  python cwhop_tx.py --config cwhopping_config.json --output my_capture \\
      --duration 60 --gpsdo --gpsdo-timeout 90 \\
      --args "serial=33767A5,master_clock_rate=184.32e6" \\
      --tx-gain 15 --rx-gain 30 --center-freq 5.9e9 \\
      --tx-subdev "A:0" --rx-subdev "A:0"

Output files
------------
  <output>.cf32               Baseband IQ (complex64, interleaved I/Q)
  <output>_schedule.json      Realized hop schedule + SC-FDMA log (sidecar)
  <output>_rx.cf32            RX capture (when not --generate-only)
  <output>_rx.json            RX sidecar pointing back to TX sidecar
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Fixed C-V2X 20 MHz PHY constants (3GPP Rel 14, never configurable)
# ---------------------------------------------------------------------------
FS_HZ = 30_720_000          # Native sample rate, Hz
SAMPLES_PER_SF = 30_720     # Samples per 1 ms subframe  (= Fs * 1e-3)
N_FFT_CV2X = 2048           # OFDM FFT size (14 symbols x 1 slot)
TOTAL_SUBCH = 20            # 20 subchannels in 20 MHz C-V2X
BINS_PER_SUBCH = 1199 / TOTAL_SUBCH   # ~60 sensing bins per subchannel

# CP lengths for one 1ms subframe (2 slots, 7 symbols each):
#   slot 0: [160, 144, 144, 144, 144, 144, 144]
#   slot 1: [160, 144, 144, 144, 144, 144, 144]
# Total = 2*(160 + 6*144) + 12*0  ... let's compute explicitly:
#   2*(160+2048) + 12*(144+2048) = 4416 + 26304 = 30720  ✓
_CP_LENS = [160, 144, 144, 144, 144, 144, 144,
            160, 144, 144, 144, 144, 144, 144]

# Block size for streaming generation (1 second = 30.72 M samples).
# Keeps peak memory near 246 MB (2 x float32 x 30.72M) rather than holding
# the entire capture in RAM.
BLOCK_SIZE = 30_720_000


# ============================================================================
# GPSDO helpers  (copied from test_hopper_tx.py)
# ============================================================================
_QUERY_GPSDO_PATHS = [
    os.path.expanduser(r"~\radioconda\Library\lib\uhd\utils"
                       r"\query_gpsdo_sensors.exe"),
    r"C:\Program Files (x86)\National Instruments"
    r"\NI-USRP\utilities\query_gpsdo_sensors.exe",
]


def _gpsdo_preflight(dev_args, timeout=90):
    """Run query_gpsdo_sensors before touching GNU Radio."""
    import subprocess
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
                        "LookupError" in output) and attempt < 2:
                    print("[gpsdo]   Device not ready (FPGA loading?). "
                          "Retrying in 10 s ...")
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

    print(f"[gpsdo:{label}] Waiting for ref_locked "
          f"(up to {timeout} s) ...")
    ref_locked = False
    for _ in range(timeout):
        try:
            ref_locked = usrp_block.get_mboard_sensor(
                "ref_locked").to_bool()
        except Exception:
            ref_locked = False
        if ref_locked:
            break
        time.sleep(1.0)

    if not ref_locked:
        print(f"[gpsdo:{label}] WARNING -- ref did not lock "
              f"in {timeout} s.")
        return

    print(f"[gpsdo:{label}] 10 MHz reference LOCKED.")
    try:
        gps_locked = usrp_block.get_mboard_sensor("gps_locked").to_bool()
        print(f"[gpsdo:{label}] GPS nav lock: "
              f"{'YES' if gps_locked else 'no (not required)'}")
    except Exception:
        pass

    try:
        gps_time = usrp_block.get_mboard_sensor("gps_time").to_int()
        next_pps = gps_time + 1
        usrp_block.set_time_next_pps(uhd.time_spec(float(next_pps)))
        time.sleep(1.1)
        print(f"[gpsdo:{label}] USRP time set to {next_pps} "
              f"(UTC via GPS PPS).")
    except RuntimeError:
        print(f"[gpsdo:{label}] gps_time sensor not available, "
              f"syncing to next PPS with system time.")
        next_pps = math.ceil(time.time()) + 1
        usrp_block.set_time_next_pps(uhd.time_spec(float(next_pps)))
        time.sleep(1.1)
        print(f"[gpsdo:{label}] USRP time set to {next_pps} "
              f"(system clock + PPS).")


# ============================================================================
# Taper envelope  (inline from make_taper_envelope.py)
# ============================================================================
def _make_taper_envelope(n_dwell: int,
                         taper_shape: str = 'raised_cosine',
                         taper_pct: float = 0.10,
                         min_amp: float = 0.0) -> np.ndarray:
    """Generate amplitude envelope for one dwell period.

    Returns a float64 array of shape (n_dwell,) with values in [min_amp, 1.0].
    The envelope ramps from min_amp up to 1.0 over the first taper_pct fraction
    of samples, holds at 1.0 in the middle, and mirrors the ramp back down at
    the end.  During gap regions the caller is responsible for zeroing.
    """
    from scipy.signal import windows as _windows

    env = np.ones(n_dwell, dtype=np.float64)

    if taper_shape == 'none' or taper_pct <= 0:
        return env

    n_ramp = int(np.floor(taper_pct * n_dwell))
    if n_ramp < 2:
        return env

    shape = taper_shape.lower()
    if shape == 'raised_cosine':
        n = np.arange(n_ramp) / (n_ramp - 1)
        ramp = 0.5 * (1 - np.cos(np.pi * n))
    elif shape == 'hann':
        w = _windows.hann(2 * n_ramp, sym=False)
        ramp = w[:n_ramp]
    elif shape == 'blackman':
        w = _windows.blackman(2 * n_ramp, sym=False)
        ramp = w[:n_ramp]
    elif shape == 'linear':
        ramp = np.linspace(0.0, 1.0, n_ramp)
    else:
        raise ValueError(f"Unknown taper_shape: '{taper_shape}'. "
                         f"Choose from: none, linear, raised_cosine, hann, blackman")

    ramp = min_amp + (1.0 - min_amp) * ramp
    env[:n_ramp] = ramp
    env[-n_ramp:] = ramp[::-1]
    return env


# ============================================================================
# Config loading
# ============================================================================
def load_config(config_path: str) -> dict:
    """Load and validate cwhopping_config.json."""
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    # Validate key sections exist
    for section in ("transmitter", "hardware"):
        if section not in cfg:
            sys.exit(f"ERROR: Config missing required section '{section}'")

    return cfg


# ============================================================================
# Hop schedule construction
# ============================================================================
def build_hop_schedule(cfg: dict, duration_s: float) -> list:
    """Construct the realised hop schedule from config parameters.

    Returns a list of dicts, each representing one dwell:
        {dwell_idx, start_sample, end_sample, freq_hz, start_ms, end_ms}

    The algorithm mirrors generate_hopped_cw.py exactly:
      - N_dwell is rounded to the nearest multiple of SAMPLES_PER_SF so that
        all boundaries land on 1 ms subframe edges.
      - Gap per dwell = max(0, round(gap_ms + U(-jitter, +jitter))) ms.
      - hop_sparsity: at each dwell boundary a Bernoulli draw decides whether
        the frequency index advances.

    NOTE: All sample positions are multiples of 30720 because dwell_ms and
    gap_ms are integer milliseconds and Fs=30.72 MHz -> 30720 samples/ms.
    """
    tx = cfg["transmitter"]
    hw = cfg["hardware"]
    Fs = hw["Fs_hz"]

    dwell_ms = tx["dwell_ms"]
    hop_freqs = np.asarray(tx["hop_freqs_hz"], dtype=np.float64)
    hop_sparsity = float(tx["hop_sparsity"])
    gap_ms = float(tx["gap_ms"])
    gap_jitter_ms = float(tx["gap_jitter_ms"])

    N_total = round(Fs * duration_s)
    N_dwell = round(Fs * dwell_ms / 1000)

    # Enforce subframe alignment: N_dwell must be a multiple of SAMPLES_PER_SF
    if N_dwell % SAMPLES_PER_SF != 0:
        N_dwell = ((N_dwell + SAMPLES_PER_SF - 1) // SAMPLES_PER_SF) * SAMPLES_PER_SF
        actual_dwell_ms = N_dwell / Fs * 1000
        print(f"[schedule] NOTE: dwell_ms={dwell_ms} rounded to "
              f"{actual_dwell_ms:.3f} ms for subframe alignment")

    num_freqs = len(hop_freqs)
    freq_idx = 0
    sample_pos = 0
    schedule = []
    max_dwells = int(np.ceil(N_total / max(N_dwell, 1))) + 10  # safety margin

    for d in range(max_dwells):
        s_start = sample_pos
        if s_start >= N_total:
            break

        s_end = min(s_start + N_dwell, N_total) - 1

        # Sparsity: advance freq_idx with probability hop_sparsity
        # (never advance on first dwell so we always start at freq_idx=0)
        if d > 0 and np.random.rand() <= hop_sparsity:
            freq_idx = (freq_idx + 1) % num_freqs

        schedule.append({
            "dwell_idx":    d,
            "start_sample": int(s_start),
            "end_sample":   int(s_end),
            "freq_hz":      float(hop_freqs[freq_idx]),
            "start_ms":     round(s_start / Fs * 1000, 6),
            "end_ms":       round(s_end   / Fs * 1000, 6),
        })

        # Per-dwell randomised gap
        if gap_jitter_ms > 0:
            jitter = np.random.uniform(-gap_jitter_ms, gap_jitter_ms)
            gap_this_ms = max(0, round(gap_ms + jitter))
        else:
            gap_this_ms = round(gap_ms)

        N_gap = round(Fs * gap_this_ms / 1000)
        sample_pos = s_end + 1 + N_gap

    return schedule, N_total


# ============================================================================
# Frequency / envelope timeline builder
# ============================================================================
def _build_timeline(schedule: list, N_total: int) -> list:
    """Convert hop schedule to a flat list of (s_start, s_end, freq, is_dwell).

    Gap regions carry the preceding dwell frequency so the NCO phase keeps
    advancing correctly (spec section 4.3: 'global_phase STILL advances at
    the last dwell's frequency').  The is_dwell flag lets the envelope builder
    zero out gap regions.
    """
    timeline = []
    cursor = 0
    last_freq = schedule[0]["freq_hz"] if schedule else 0.0

    for dw in schedule:
        s_start = dw["start_sample"]
        s_end   = dw["end_sample"]

        # Gap before this dwell (if any)
        if cursor < s_start:
            timeline.append((cursor, s_start - 1, last_freq, False))

        timeline.append((s_start, s_end, dw["freq_hz"], True))
        last_freq = dw["freq_hz"]
        cursor = s_end + 1

    # Trailing gap after the last dwell
    if cursor < N_total:
        timeline.append((cursor, N_total - 1, last_freq, False))

    return timeline


# ============================================================================
# Taper cache
# ============================================================================
class _TaperCache:
    """Cache taper envelopes by dwell length to avoid redundant computation."""

    def __init__(self, taper_shape, taper_pct, min_amp):
        self._shape = taper_shape
        self._pct   = taper_pct
        self._min   = min_amp
        self._cache = {}

    def get(self, n_dwell: int) -> np.ndarray:
        if n_dwell not in self._cache:
            self._cache[n_dwell] = _make_taper_envelope(
                n_dwell, self._shape, self._pct, self._min)
        return self._cache[n_dwell]


# ============================================================================
# SC-FDMA subframe generation
# ============================================================================
def _generate_scfdma_subframe(
        sf_idx: int,
        cw_freq_hz: float,
        scfdma_cfg: dict,
        A_cw: float,
        Fs: float,
) -> tuple:
    """Generate one 1ms SC-FDMA subframe (30720 samples) and return
    (subframe_cf32, log_entry).

    The subframe contains QPSK symbols mapped to a random contiguous subchannel
    allocation using the standard C-V2X 20 MHz OFDM numerology:
      N_fft = 2048, 14 symbols/subframe (7 per slot), variable CP lengths.

    Subcarrier mapping (data SCs 0..1199 into FFT bins):
      SC 0..599     -> bins 1..600
      SC 600..1199  -> bins N_fft-600..N_fft-1
    (DC bin 0 is unused; mirrored placement for 2-sided spectrum.)

    Power is normalised so mean(|subframe|^2) equals the target P_data.
    """
    TOTAL_SC = 1200    # 100 PRBs x 12 SC
    SC_PER_SUBCH = TOTAL_SC // TOTAL_SUBCH   # = 60

    # Scenario-based allocation placement
    colocated = scfdma_cfg["colocated_tx"]
    inside    = scfdma_cfg["cw_inside_alloc"]
    num_subch_max = int(scfdma_cfg["num_subch_max"])
    data_power_dB = float(scfdma_cfg["data_power_dB"])
    A_cw_sq = A_cw ** 2

    # CW bin -> subchannel index
    bin_idx = round(cw_freq_hz / 7500.0)
    cw_subch = min(int(bin_idx * TOTAL_SUBCH / 1199), TOTAL_SUBCH - 1)

    # Choose allocation size L
    L = np.random.randint(1, num_subch_max + 1)

    if colocated and inside:
        # CW bin is inside allocation: place so cw_subch is covered
        max_start = max(0, min(TOTAL_SUBCH - L, cw_subch))
        start_subch = np.random.randint(
            max(0, cw_subch - L + 1),
            min(TOTAL_SUBCH - L + 1, cw_subch + 1))
    elif colocated and not inside:
        # CW bin must be OUTSIDE allocation
        # Place allocation entirely below or above cw_subch
        candidates = []
        # Allocation entirely below cw_subch
        if cw_subch >= L:
            candidates.append(np.random.randint(0, cw_subch - L + 1))
        # Allocation entirely above cw_subch
        if cw_subch + L < TOTAL_SUBCH:
            candidates.append(np.random.randint(cw_subch + 1,
                                                 TOTAL_SUBCH - L + 1))
        if not candidates:
            # Edge case: no valid placement; fall back to random
            start_subch = np.random.randint(0, max(TOTAL_SUBCH - L + 1, 1))
        else:
            start_subch = candidates[np.random.randint(len(candidates))]
    else:
        # Other vehicle: fully random placement
        start_subch = np.random.randint(0, max(TOTAL_SUBCH - L + 1, 1))

    # Subchannel -> subcarrier range
    sc_start = start_subch * SC_PER_SUBCH      # first subcarrier index
    sc_end   = sc_start + L * SC_PER_SUBCH     # one past last subcarrier

    # Check CW overlap
    alloc_subch_set = set(range(start_subch, start_subch + L))
    overlaps_cw = cw_subch in alloc_subch_set

    # Build OFDM subframe sample by sample ─────────────────────────────────
    subframe = np.zeros(SAMPLES_PER_SF, dtype=np.complex128)
    n_sc_alloc = L * SC_PER_SUBCH
    ptr = 0

    for sym_idx, cp_len in enumerate(_CP_LENS):
        # Random QPSK on allocated subcarriers
        qpsk_vals = (np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2.0))
        symbols = qpsk_vals[np.random.randint(0, 4, n_sc_alloc)]

        # Map to FFT frequency bins
        freq_domain = np.zeros(N_FFT_CV2X, dtype=np.complex128)
        sc_indices = np.arange(sc_start, sc_end)  # length = n_sc_alloc

        # Subcarrier mapping: lower half -> bins 1..600; upper half -> N_fft-600..N_fft-1
        for sc_i, sym in zip(sc_indices, symbols):
            if sc_i < TOTAL_SC // 2:      # SC 0..599
                freq_domain[sc_i + 1] = sym
            else:                          # SC 600..1199
                freq_domain[N_FFT_CV2X - (TOTAL_SC - sc_i)] = sym

        # IFFT and add CP
        td = np.fft.ifft(freq_domain) * N_FFT_CV2X   # normalise for IFFT
        cp = td[-cp_len:]
        sym_samples = np.concatenate([cp, td])

        subframe[ptr:ptr + cp_len + N_FFT_CV2X] = sym_samples
        ptr += cp_len + N_FFT_CV2X

    # Normalise to target power
    P_data = A_cw_sq * 10.0 ** (data_power_dB / 10.0)
    actual_power = np.mean(np.abs(subframe) ** 2)
    if actual_power > 0:
        subframe *= np.sqrt(P_data / actual_power)

    log_entry = {
        "subframe_idx":  int(sf_idx),
        "start_sample":  int(sf_idx * SAMPLES_PER_SF),
        "num_subch":     int(L),
        "start_subch":   int(start_subch),
        "overlaps_cw":   bool(overlaps_cw),
    }

    return subframe.astype(np.complex64), log_entry


# ============================================================================
# Streaming CW generation
# ============================================================================
def generate_and_write(
        output_cf32: str,
        schedule: list,
        N_total: int,
        cfg: dict,
        scfdma_log_out: list,
) -> None:
    """Generate phase-continuous hopped CW in 1-second blocks and write to file.

    Phase continuity invariant
    --------------------------
    global_phase is a float64 scalar that NEVER resets.  Within each block it
    is propagated via cumulative sum (vectorised, avoids a Python per-sample
    loop), then wrapped modulo 2π to prevent float64 overflow on long captures.

    Block processing
    ----------------
    Each BLOCK_SIZE-sample block goes through:
      1. freq_block[n]: frequency assigned to each sample (float64 array)
         built by scanning the pre-computed timeline for overlapping segments.
      2. env_block[n]:  amplitude envelope -- 1.0 inside dwells (with ramps),
         0.0 in gap regions.
      3. Phase accumulation (vectorised):
           phase_incr = 2π * freq_block / Fs
           cumphase   = cumsum(phase_incr)
           phases[n]  = global_phase + cumphase[n-1]   (for n>0), global_phase for n=0
           global_phase += cumphase[-1]; wrap to [0, 2π)
      4. sig_block = A_cw * env_block * exp(j * phases)  -> cast to complex64
      5. SC-FDMA: for each 1ms subframe in the block, Bernoulli draw and add
         SC-FDMA waveform if enabled.
    """
    tx   = cfg["transmitter"]
    hw   = cfg["hardware"]
    Fs   = hw["Fs_hz"]
    A_cw = float(tx["A_cw"])

    scfdma_cfg = cfg.get("scfdma", {})
    scfdma_on  = scfdma_cfg.get("enabled", False)
    data_activity = float(scfdma_cfg.get("data_activity", 0.5))

    taper = _TaperCache(
        taper_shape=tx["taper_shape"],
        taper_pct=float(tx["taper_pct"]),
        min_amp=float(tx["min_amplitude"]),
    )

    timeline = _build_timeline(schedule, N_total)
    n_blocks = int(np.ceil(N_total / BLOCK_SIZE))

    global_phase = 0.0   # float64, never reset

    t_start_gen = time.time()

    with open(output_cf32, "wb") as fout:
        for blk in range(n_blocks):
            block_start = blk * BLOCK_SIZE
            block_end   = min(block_start + BLOCK_SIZE, N_total)
            n_blk       = block_end - block_start

            # ── Build per-sample frequency and envelope arrays ────────────
            freq_block = np.zeros(n_blk, dtype=np.float64)
            env_block  = np.zeros(n_blk, dtype=np.float64)

            for (seg_s, seg_e, freq, is_dwell) in timeline:
                # Check overlap with [block_start, block_end)
                ol_s = max(seg_s, block_start) - block_start
                ol_e = min(seg_e + 1, block_end) - block_start
                if ol_s >= ol_e:
                    continue

                freq_block[ol_s:ol_e] = freq

                if is_dwell:
                    # Retrieve or compute taper for this dwell's full length
                    n_dwell_full = seg_e - seg_s + 1
                    env_full = taper.get(n_dwell_full)

                    # Offset into the dwell where this block segment starts
                    dwell_offset_s = max(seg_s, block_start) - seg_s
                    dwell_offset_e = dwell_offset_s + (ol_e - ol_s)
                    env_block[ol_s:ol_e] = env_full[dwell_offset_s:dwell_offset_e]
                # else: gap region -- env stays 0.0

            # ── Vectorised phase accumulation ─────────────────────────────
            # phase_incr[n] = 2π * freq[n] / Fs
            # phases[n] = global_phase + sum(phase_incr[0..n-1])
            # This is equivalent to the sample-by-sample NCO but ~1000x faster.
            phase_incr = (2.0 * np.pi / Fs) * freq_block
            cumphase   = np.cumsum(phase_incr)
            # phases[0] = global_phase (phase before first new increment)
            # phases[n] = global_phase + cumphase[n-1]  for n >= 1
            phases = global_phase + np.concatenate([[0.0], cumphase[:-1]])

            # Advance global_phase by the total accumulated in this block,
            # then wrap to [0, 2π) to prevent float64 mantissa erosion on
            # very long captures (hours).
            global_phase += cumphase[-1]
            global_phase %= (2.0 * np.pi)

            # ── Modulate and apply envelope ───────────────────────────────
            sig_block = (A_cw * env_block * np.exp(1j * phases)).astype(np.complex64)

            # ── SC-FDMA overlay (when enabled) ────────────────────────────
            if scfdma_on:
                # Iterate over 1ms subframes that overlap this block
                sf_first = block_start // SAMPLES_PER_SF
                sf_last  = (block_end - 1) // SAMPLES_PER_SF

                for sf_idx in range(sf_first, sf_last + 1):
                    sf_abs_start = sf_idx * SAMPLES_PER_SF
                    sf_abs_end   = sf_abs_start + SAMPLES_PER_SF

                    # Bernoulli: skip with probability (1 - data_activity)
                    if np.random.rand() > data_activity:
                        continue

                    # Find CW frequency at this subframe
                    # (use the timeline: first dwell/gap that covers sf_abs_start)
                    cw_freq = 0.0
                    for (seg_s, seg_e, freq, is_dwell) in timeline:
                        if seg_s <= sf_abs_start <= seg_e:
                            cw_freq = freq
                            break
                    if cw_freq == 0.0:
                        continue

                    subframe_cf32, log_entry = _generate_scfdma_subframe(
                        sf_idx=sf_idx,
                        cw_freq_hz=cw_freq,
                        scfdma_cfg=scfdma_cfg,
                        A_cw=A_cw,
                        Fs=Fs,
                    )
                    scfdma_log_out.append(log_entry)

                    # Inject SC-FDMA subframe into the block buffer.
                    # The subframe may straddle block boundaries, so clip.
                    blk_sf_s = sf_abs_start - block_start
                    blk_sf_e = sf_abs_end   - block_start
                    blk_sf_s_clip = max(blk_sf_s, 0)
                    blk_sf_e_clip = min(blk_sf_e, n_blk)
                    sf_slice_s = blk_sf_s_clip - blk_sf_s
                    sf_slice_e = blk_sf_e_clip - blk_sf_s

                    sig_block[blk_sf_s_clip:blk_sf_e_clip] += \
                        subframe_cf32[sf_slice_s:sf_slice_e]

            # ── Write block ───────────────────────────────────────────────
            sig_block.tofile(fout)

            # ── Progress report ───────────────────────────────────────────
            elapsed = time.time() - t_start_gen
            frac_done = (blk + 1) / n_blocks
            if frac_done > 0:
                eta = elapsed / frac_done * (1.0 - frac_done)
            else:
                eta = 0.0
            print(f"  [gen] Block {blk+1}/{n_blocks}  "
                  f"{frac_done*100:.1f}%  "
                  f"ETA {eta:.1f} s", end="\r")

    print()  # newline after \r progress


# ============================================================================
# Sidecar I/O
# ============================================================================
def _sidecar_path(base: str) -> str:
    """Return <base>_schedule.json path."""
    return base + "_schedule.json"


def write_tx_sidecar(sidecar_path: str,
                     config_path: str,
                     Fs: int,
                     N_total: int,
                     schedule: list,
                     scfdma_log: list = None) -> None:
    """Write (or update) the TX sidecar JSON file."""
    data = {
        "version":     1,
        "config_file": config_path,
        "Fs_hz":       Fs,
        "N_total":     int(N_total),
        "hop_schedule": schedule,
    }
    if scfdma_log is not None:
        data["scfdma_log"] = scfdma_log

    with open(sidecar_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[sidecar] Wrote {sidecar_path}")


# ============================================================================
# GNU Radio flowgraph
# ============================================================================
class CWHopTxRx:
    """Simultaneous TX replay from file and RX capture to file via UHD."""

    def __init__(self, args, tx_file: str, rx_file: str):
        from gnuradio import gr, blocks, uhd

        self.tb = gr.top_block("CWHop TX/RX")
        self._args = args

        Fs = FS_HZ  # both TX and RX run at the native C-V2X rate

        # -- Open RX first (claims NI-RPC session) ---------------------------
        print(f"\n[rx] Opening RX on subdev {args.rx_subdev} ...")
        self.rx_src = uhd.usrp_source(
            args.args,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )
        self.rx_src.set_subdev_spec(args.rx_subdev)
        self.rx_src.set_samp_rate(Fs)
        self.rx_src.set_center_freq(args.center_freq)
        self.rx_src.set_gain(args.rx_gain)
        self.rx_src.set_antenna("RX2")
        self.rx_src.set_auto_iq_balance(True, 0)
        actual_rx_rate = self.rx_src.get_samp_rate()
        print(f"[rx] Rate: {actual_rx_rate:.0f} sps")
        print(f"[rx] Freq: {args.center_freq/1e9:.6f} GHz  "
              f"Gain: {args.rx_gain} dB")

        print("[rx] Waiting 2 s for USRP init to settle...")
        time.sleep(2.0)

        # -- Open TX ---------------------------------------------------------
        print(f"\n[tx] Opening TX on subdev {args.tx_subdev} ...")
        self.tx_sink = uhd.usrp_sink(
            args.args,
            uhd.stream_args(cpu_format="fc32", channels=[0]),
        )
        self.tx_sink.set_subdev_spec(args.tx_subdev)
        self.tx_sink.set_samp_rate(Fs)
        self.tx_sink.set_center_freq(args.center_freq)
        self.tx_sink.set_gain(args.tx_gain)
        self.tx_sink.set_antenna("TX/RX")
        actual_tx_rate = self.tx_sink.get_samp_rate()
        print(f"[tx] Rate: {actual_tx_rate:.0f} sps")
        print(f"[tx] Freq: {args.center_freq/1e9:.6f} GHz  "
              f"Gain: {args.tx_gain} dB")
        if abs(actual_tx_rate - Fs) > 1.0:
            print(f"[tx] WARNING: actual rate {actual_tx_rate:.0f} differs "
                  f"from requested {Fs:.0f}.")

        print("[tx] Waiting 2 s for USRP init ...")
        time.sleep(2.0)

        # -- GPSDO -----------------------------------------------------------
        self._gpsdo = args.gpsdo
        if args.gpsdo:
            print()
            _gpsdo_setup(self.tx_sink, "tx", args.gpsdo_timeout)
            self.rx_src.set_clock_source("gpsdo")
            self.rx_src.set_time_source("gpsdo")

        # -- TX path: file_source (repeat=False) -> tx_sink -------------------
        # repeat=False: TX plays the file exactly once.  The capture duration
        # controls how long the RX records.
        self.tx_file_src = blocks.file_source(
            gr.sizeof_gr_complex, tx_file, repeat=False)
        self.tb.connect(self.tx_file_src, self.tx_sink)

        # -- RX path: rx_src -> file_sink -------------------------------------
        self.rx_file_sink = blocks.file_sink(
            gr.sizeof_gr_complex, rx_file, append=False)
        self.rx_file_sink.set_unbuffered(False)
        self.tb.connect(self.rx_src, self.rx_file_sink)

        self._actual_rx_rate = actual_rx_rate
        self._rx_start_utc   = None
        self._rx_file        = rx_file

    def start(self):
        from gnuradio import uhd

        if self._gpsdo:
            now = self.rx_src.get_time_now().get_real_secs()
            rx_target = math.ceil((now + 0.5) * 1000.0) / 1000.0
            # Schedule BOTH TX and RX to start at the same GPS time.
            # Without this, TX starts immediately at tb.start() while RX
            # waits for rx_target, creating a 0.5-1.5 s timing offset that
            # shifts every dwell boundary in the RX file relative to the
            # sidecar -- causing de-hopping to apply the wrong frequency
            # and producing ghost images in the spectrogram.
            self.rx_src.set_start_time(uhd.time_spec(rx_target))
            self.tx_sink.set_start_time(uhd.time_spec(rx_target))
            self._rx_start_utc = rx_target
            print(f"\n[gpsdo] TX + RX both scheduled at t = {rx_target:.6f} s")
        else:
            self._rx_start_utc = time.time()

        self.tb.start()
        print("\n[info] Flowgraph started.  TX + RX running...")

    def stop(self):
        self.tb.stop()
        self.tb.wait()

    def write_rx_sidecar(self, tx_sidecar_path: str) -> str:
        """Write a minimal RX sidecar that references the TX sidecar."""
        rx_sidecar_path = os.path.splitext(self._rx_file)[0] + ".json"
        rx_meta = {
            "version":          1,
            "rx_center_freq_hz": self._args.center_freq,
            "rx_rate_hz":       self._actual_rx_rate,
            "rx_start_utc":     self._rx_start_utc,
            "duration_s":       self._args.duration,
            "tx_sidecar_ref":   os.path.abspath(tx_sidecar_path),
        }
        with open(rx_sidecar_path, "w") as fh:
            json.dump(rx_meta, fh, indent=2)
        print(f"[sidecar] Wrote RX sidecar: {rx_sidecar_path}")
        return rx_sidecar_path


# ============================================================================
# CLI argument parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", default="cwhopping_config.json",
        help="Path to cwhopping_config.json (default: cwhopping_config.json)")
    parser.add_argument(
        "--output", required=True, metavar="BASE",
        help="Output base name (no extension).  Writes <BASE>.cf32 and "
             "<BASE>_schedule.json.")
    parser.add_argument(
        "--duration", "-d", type=float, default=60.0,
        help="Capture duration in seconds (default: 60)")
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Generate IQ file only; skip GNU Radio TX/RX")
    parser.add_argument(
        "--gpsdo", action="store_true",
        help="Enable GPSDO clock/time synchronisation")
    parser.add_argument(
        "--gpsdo-timeout", type=int, default=90,
        help="GPSDO lock timeout in seconds (default: 90)")
    parser.add_argument(
        "--args", "-a",
        default="serial=33767A5,master_clock_rate=184.32e6",
        help="UHD device args string")
    parser.add_argument(
        "--tx-gain", type=float, default=15.0,
        help="TX gain in dB (default: 15)")
    parser.add_argument(
        "--rx-gain", type=float, default=30.0,
        help="RX gain in dB (default: 30)")
    parser.add_argument(
        "--center-freq", type=float, default=None,
        help="RF centre frequency in Hz (default: from config)")
    parser.add_argument(
        "--tx-subdev", default="A:0",
        help="TX subdevice spec (default: A:0)")
    parser.add_argument(
        "--rx-subdev", default="A:0",
        help="RX subdevice spec (default: A:0)")

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    print("=" * 62)
    print("  CWHop TX -- Config-Driven Hopped CW Transmitter")
    print("=" * 62)

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    hw  = cfg["hardware"]
    tx  = cfg["transmitter"]

    Fs           = int(hw["Fs_hz"])
    center_freq  = float(args.center_freq or hw["center_freq_hz"])

    print(f"\n[config] {config_path}")
    print(f"  Fs:           {Fs/1e6:.3f} MHz")
    print(f"  Center freq:  {center_freq/1e9:.6f} GHz")
    print(f"  Dwell:        {tx['dwell_ms']} ms")
    print(f"  Hop freqs:    {tx['hop_freqs_hz']} Hz")
    print(f"  Hop sparsity: {tx['hop_sparsity']:.3f}")
    print(f"  Gap:          {tx['gap_ms']} ms +/- {tx['gap_jitter_ms']} ms jitter")
    print(f"  Taper:        {tx['taper_shape']}, {tx['taper_pct']*100:.1f}%")
    print(f"  Duration:     {args.duration:.1f} s")

    scfdma_on = cfg.get("scfdma", {}).get("enabled", False)
    if scfdma_on:
        sc = cfg["scfdma"]
        print(f"  SC-FDMA:      ENABLED -- {sc.get('scenario', '')}")
        print(f"    data_activity={sc['data_activity']:.3f}  "
              f"num_subch_max={sc['num_subch_max']}  "
              f"data_power={sc['data_power_dB']} dB")
    else:
        print("  SC-FDMA:      disabled")

    # ── Derived paths ────────────────────────────────────────────────────────
    output_base  = args.output
    tx_cf32      = output_base + ".cf32"
    tx_sidecar   = _sidecar_path(output_base)
    rx_cf32      = output_base + "_rx.cf32"

    # ── Build hop schedule ────────────────────────────────────────────────────
    print(f"\n[schedule] Building hop schedule for {args.duration:.1f} s ...")
    schedule, N_total = build_hop_schedule(cfg, args.duration)
    print(f"[schedule] {len(schedule)} dwells, {N_total:,} total samples "
          f"({N_total/Fs:.3f} s)")

    # ── Write sidecar BEFORE generation (hop_schedule only) ──────────────────
    # Spec section 7.4: write at startup after building the schedule.
    # SC-FDMA log will be appended/updated after generation.
    write_tx_sidecar(
        sidecar_path=tx_sidecar,
        config_path=config_path,
        Fs=Fs,
        N_total=N_total,
        schedule=schedule,
        scfdma_log=[] if scfdma_on else None,
    )

    # ── Generate IQ file ──────────────────────────────────────────────────────
    file_size_mb = N_total * 8 / 1e6
    print(f"\n[gen] Generating {tx_cf32}")
    print(f"      {N_total:,} samples  ({file_size_mb:.1f} MB)  "
          f"{N_total/BLOCK_SIZE:.1f} blocks of {BLOCK_SIZE/1e6:.0f}M")

    scfdma_log: list = []
    t0 = time.time()
    generate_and_write(tx_cf32, schedule, N_total, cfg, scfdma_log)
    gen_elapsed = time.time() - t0
    print(f"[gen] Done in {gen_elapsed:.1f} s  "
          f"({file_size_mb/gen_elapsed:.1f} MB/s)")

    # ── Finalise sidecar with SC-FDMA log ─────────────────────────────────────
    if scfdma_on:
        write_tx_sidecar(
            sidecar_path=tx_sidecar,
            config_path=config_path,
            Fs=Fs,
            N_total=N_total,
            schedule=schedule,
            scfdma_log=scfdma_log,
        )
        print(f"[scfdma] Logged {len(scfdma_log)} active subframes")

    # ── Summary ──────────────────────────────────────────────────────────────
    actual_size = os.path.getsize(tx_cf32) if os.path.isfile(tx_cf32) else 0
    print(f"\n[summary]")
    print(f"  TX file:    {tx_cf32}  ({actual_size/1e6:.2f} MB)")
    print(f"  Sidecar:    {tx_sidecar}")
    print(f"  Dwells:     {len(schedule)}")
    print(f"  Duration:   {N_total/Fs:.3f} s")

    if args.generate_only:
        print("\n[done] --generate-only: skipping TX/RX hardware.")
        print(f"       Next: python cwhop_rx.py --input {rx_cf32} "
              f"--sidecar {tx_sidecar} --plot")
        return 0

    # ── GNU Radio TX/RX ──────────────────────────────────────────────────────
    if args.gpsdo:
        _gpsdo_preflight(args.args, args.gpsdo_timeout)

    # Override center_freq in args so HW setup picks it up
    args.center_freq = center_freq

    txrx = CWHopTxRx(args, tx_file=tx_cf32, rx_file=rx_cf32)
    txrx.start()

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

    txrx.stop()
    rx_sidecar_path = txrx.write_rx_sidecar(tx_sidecar)

    rx_size = os.path.getsize(rx_cf32) if os.path.isfile(rx_cf32) else 0
    print(f"\n[done] RX file:    {rx_cf32}  ({rx_size/1e6:.2f} MB)")
    print(f"[done] RX sidecar: {rx_sidecar_path}")
    print(f"[done] Next: python cwhop_rx.py --input {rx_cf32} "
          f"--sidecar {tx_sidecar} --plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
