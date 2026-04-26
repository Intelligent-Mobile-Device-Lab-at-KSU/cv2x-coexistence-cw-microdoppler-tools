#!/usr/bin/env python3
"""
cwhop_rx.py -- Config-Driven Hopped CW RX Processor
=====================================================
Reads a raw .cf32 capture and its TX sidecar schedule, de-hops the
frequency-hopping CW tone, decimates from 30.72 MHz to 10 kHz in six
cascaded FIR stages, and computes an STFT spectrogram for micro-Doppler
extraction.

The de-hopped signal places all dwells at a common baseband frequency (DC),
so the spectrogram shows micro-Doppler content (arm-swing butterfly) at
0 Hz +/- f_doppler.

Usage examples
--------------
  # Basic -- auto-find sidecar and config:
  python cwhop_rx.py --input my_capture_rx.cf32 --plot

  # Explicit paths + save results:
  python cwhop_rx.py \\
      --input      my_capture_rx.cf32 \\
      --sidecar    my_capture_schedule.json \\
      --config     cwhopping_config.json \\
      --output-prefix my_results \\
      --doppler-range 400 \\
      --dynamic-range 60 \\
      --plot --save

Design notes
------------
Memory efficiency
  The file is accessed via np.memmap so only requested blocks touch RAM.
  Each 1-second block (30.72M complex64 samples = ~246 MB) is de-hopped and
  immediately decimated 3072x to 10000 samples (~80 KB) before accumulation.

Block-boundary artefacts
  scipy.signal.decimate uses a FIR anti-alias filter of effective order
  ~30x the decimation factor.  To avoid end-effects at block boundaries we
  read a margin of MARGIN_SAMPS extra samples on each side, decimate the
  padded block, then trim the margin from the decimated output.

Phase accumulation in de-hopping
  The de-hopping phasor exp(-j*2π*Δf*t) uses GLOBAL sample time
  t[n] = n/Fs (not per-block local time).  This is critical: resetting t
  to zero at each block boundary would break the phase relationship
  established by the TX NCO.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.signal import decimate, spectrogram
from scipy.signal import windows as sig_windows


# ---------------------------------------------------------------------------
# Fixed C-V2X 20 MHz PHY constants
# ---------------------------------------------------------------------------
FS_HZ         = 30_720_000   # Native sample rate, Hz
SAMPLES_PER_SF = 30_720       # Samples per 1 ms subframe

# Target output sample rate after decimation
FS_OUT_HZ = 10_000.0  # 10 kHz

# Cascaded decimation stages: 4^5 * 3 = 3072 total
# 30.72 MHz -> 7.68 MHz -> 1.92 MHz -> 480 kHz -> 120 kHz -> 30 kHz -> 10 kHz
DECIM_STAGES = [4, 4, 4, 4, 4, 3]

# Margin in RAW samples added around each block before decimation, then
# trimmed after.  This absorbs FIR filter transients at block edges.
# scipy.signal.decimate uses an 8th-order Chebyshev IIR or a 30*q FIR; we
# use 'fir' mode so effective length ~ 30*q.  Worst stage q=4 -> ~120 taps.
# 1000 samples is very conservative and adds only ~8 KB per block.
MARGIN_SAMPS = 1000

# 1-second block size at native rate
BLOCK_SIZE = 30_720_000


# ============================================================================
# Config / sidecar I/O
# ============================================================================
def _find_sidecar(input_path: str) -> str:
    """Derive sidecar path: strip _rx.cf32 or .cf32, append _schedule.json."""
    base = input_path
    for suffix in ("_rx.cf32", ".cf32"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + "_schedule.json"


def _find_config(input_path: str) -> str:
    """Look for cwhopping_config.json in the same directory as the input."""
    return os.path.join(os.path.dirname(os.path.abspath(input_path)),
                        "cwhopping_config.json")


def load_sidecar(sidecar_path: str) -> dict:
    """Load and return the TX sidecar JSON."""
    if not os.path.isfile(sidecar_path):
        sys.exit(f"ERROR: Sidecar not found: {sidecar_path}")
    with open(sidecar_path, "r") as fh:
        return json.load(fh)


def load_config(config_path: str) -> dict:
    """Load and return cwhopping_config.json."""
    if not os.path.isfile(config_path):
        sys.exit(f"ERROR: Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        return json.load(fh)


# ============================================================================
# STFT window helper
# ============================================================================
def _make_window(wtype: str, n: int, kaiser_beta: float = 6.0) -> np.ndarray:
    """Create a normalised STFT window array of length n."""
    wtype = wtype.lower()
    if wtype == 'rectangular':
        return np.ones(n)
    elif wtype == 'hann':
        return sig_windows.hann(n)
    elif wtype == 'blackman':
        return sig_windows.blackman(n)
    elif wtype == 'hamming':
        return sig_windows.hamming(n)
    elif wtype == 'kaiser':
        return sig_windows.kaiser(n, kaiser_beta)
    else:
        print(f"[warn] Unknown window_type '{wtype}', falling back to hann.")
        return sig_windows.hann(n)


# ============================================================================
# De-hopping  (block-wise, memory-efficient)
# ============================================================================
def dehop_block(raw_block: np.ndarray,
                block_start: int,
                hop_schedule: list,
                Fs: float) -> np.ndarray:
    """De-hop a single block of raw IQ samples.

    For each dwell that overlaps [block_start, block_start+len(raw_block)):
      1. Compute the frequency offset from f_out (first dwell frequency).
      2. Multiply the overlap region by exp(-j*2π*Δf*t) using GLOBAL time
         t[n] = n/Fs.  This cancels the hop frequency so all dwells end up
         at the same baseband frequency f_out.

    Gap regions (samples between dwells) are left as-is; they contain
    near-zero energy because the TX was off, so they contribute negligibly.

    After de-hopping, a final exp(-j*2π*f_out*t) shifts f_out to DC=0 Hz.
    This is the CFO correction step described in spec section 5.3.
    """
    n_blk  = len(raw_block)
    dehop  = raw_block.astype(np.complex128).copy()
    f_out  = hop_schedule[0]["freq_hz"]

    block_end = block_start + n_blk  # exclusive

    for dw in hop_schedule:
        s1 = dw["start_sample"]
        s2 = dw["end_sample"] + 1   # exclusive
        delta_f = dw["freq_hz"] - f_out

        # Find overlap with [block_start, block_end)
        ol_abs_s = max(s1, block_start)
        ol_abs_e = min(s2, block_end)
        if ol_abs_s >= ol_abs_e:
            continue

        ol_s = ol_abs_s - block_start   # local indices into dehop
        ol_e = ol_abs_e - block_start

        if abs(delta_f) > 0.1:
            # Use GLOBAL sample indices for the time vector.
            # This is critical: t[n] = n/Fs, not (n - block_start)/Fs.
            t_global = np.arange(ol_abs_s, ol_abs_e, dtype=np.float64) / Fs
            dehop[ol_s:ol_e] *= np.exp(-1j * 2.0 * np.pi * delta_f * t_global)

    # CFO correction: shift f_out -> DC = 0 Hz
    # Again, global time for phase continuity across blocks
    t_block_global = np.arange(block_start, block_end, dtype=np.float64) / Fs
    dehop *= np.exp(-1j * 2.0 * np.pi * f_out * t_block_global)

    return dehop.astype(np.complex64)


# ============================================================================
# Cascaded decimation
# ============================================================================
def decimate_cascaded(x: np.ndarray, stages: list,
                      verbose: bool = True) -> np.ndarray:
    """Decimate signal x through a cascade of FIR stages.

    Parameters
    ----------
    x : complex64 or complex128 array
    stages : list of int decimation factors (e.g. [4, 4, 4, 4, 4, 3])
    verbose : print intermediate rate at each stage

    Each stage uses scipy.signal.decimate with ftype='fir' and
    zero_phase=True.  The 'fir' option builds a Type I linear-phase FIR
    anti-alias filter via remez/firwin, giving excellent stopband rejection
    with no phase distortion.  zero_phase=True runs the filter forward and
    backward for true zero-phase response (doubles effective filter order).

    NOTE: scipy.signal.decimate does NOT support complex input natively
    (it checks dtype before calling lfilter).  We therefore split into real
    and imaginary parts, decimate each independently, and recombine.  This
    is mathematically equivalent to complex decimation.
    """
    Fs_current = FS_HZ
    out = x.astype(np.complex128)   # work in double for intermediate stages

    for stage_idx, q in enumerate(stages):
        # Split into I/Q, decimate each, recombine
        xi = decimate(out.real, q, ftype='fir', zero_phase=True)
        xq = decimate(out.imag, q, ftype='fir', zero_phase=True)
        out = xi + 1j * xq
        Fs_current /= q
        if verbose:
            print(f"  [decimate] Stage {stage_idx+1}/{len(stages)}: "
                  f"/{q}  ->  Fs = {Fs_current/1e3:.3f} kHz  "
                  f"({len(out):,} samples)")

    return out.astype(np.complex64)


def _decimate_block_with_margin(block_dehop: np.ndarray,
                                block_start: int,
                                n_decimated_per_block: int,
                                verbose: bool = False) -> np.ndarray:
    """Decimate one block with overlap margin to suppress edge artefacts.

    Parameters
    ----------
    block_dehop : de-hopped block (raw rate, complex64)
    block_start : sample offset of this block in the full signal
    n_decimated_per_block : expected number of output samples for this block
    verbose : enable per-stage print

    Strategy:
      - Prepend MARGIN_SAMPS zero-samples before block_start (or fewer at start).
      - Append  MARGIN_SAMPS zero-samples after block_end.
      - Decimate the padded block.
      - Trim the margin from the decimated output.

    Zero-padding is used (not raw signal) so the FIR anti-alias filter sees a
    DC-to-zero boundary rather than a DC-to-high-frequency boundary.  The
    latter caused large ringing artefacts because the raw signal at 7500-30000 Hz
    passed through the first four decimation stages (cutoffs 3.84 MHz, 960 kHz,
    240 kHz, 60 kHz) with negligible attenuation before being stopped at stage 5.

    The margin in the decimated domain is:
        margin_dec = ceil(MARGIN_SAMPS / total_decimation_factor)
    We trim this many samples from each end of the decimated padded block.
    """
    TOTAL_DECIM = 1
    for q in DECIM_STAGES:
        TOTAL_DECIM *= q    # = 3072

    n_blk  = len(block_dehop)

    # Zero-pad margins: pre-margin is clamped to [0, block_start] so we
    # don't request samples before the start of the file.
    pre_len  = min(MARGIN_SAMPS, block_start)
    post_len = MARGIN_SAMPS

    pre_samples  = np.zeros(pre_len,  dtype=np.complex128)
    post_samples = np.zeros(post_len, dtype=np.complex128)

    padded = np.concatenate([pre_samples,
                              block_dehop.astype(np.complex128),
                              post_samples])

    # Decimate the padded block through all stages
    out = padded
    Fs_cur = FS_HZ
    for q in DECIM_STAGES:
        xi = decimate(out.real, q, ftype='fir', zero_phase=True)
        xq = decimate(out.imag, q, ftype='fir', zero_phase=True)
        out = xi + 1j * xq
        Fs_cur /= q

    # Trim margin from the decimated output
    # Each margin sample at raw rate contributes 1/TOTAL_DECIM decimated samples
    trim_pre  = int(np.ceil(pre_len  / TOTAL_DECIM))
    trim_post = int(np.ceil(post_len / TOTAL_DECIM))

    out_trimmed = out[trim_pre: len(out) - trim_post if trim_post > 0 else len(out)]

    # Guard: if output is slightly longer/shorter than expected due to integer
    # rounding, pad with zeros or truncate so the caller gets a predictable size
    if len(out_trimmed) < n_decimated_per_block:
        pad = n_decimated_per_block - len(out_trimmed)
        out_trimmed = np.concatenate([out_trimmed,
                                      np.zeros(pad, dtype=np.complex128)])
    elif len(out_trimmed) > n_decimated_per_block:
        out_trimmed = out_trimmed[:n_decimated_per_block]

    return out_trimmed.astype(np.complex64)


# ============================================================================
# STFT
# ============================================================================
def compute_stft(signal_10k: np.ndarray,
                 Fs_out: float,
                 cfg: dict,
                 doppler_range_hz: float) -> tuple:
    """Compute STFT spectrogram on the 10 kHz de-hopped signal.

    Parameters
    ----------
    signal_10k     : complex decimated signal at Fs_out Hz
    Fs_out         : output sample rate (Hz), typically 10000
    cfg            : cwhopping_config dict (uses cfg['receiver'] section)
    doppler_range_hz : crop spectrogram to +/-this value (Hz); overrides config

    Returns
    -------
    S_dB   : (n_freq, n_time) array, dB
    S_lin  : (n_freq, n_time) array, linear magnitude
    f_axis : (n_freq,) frequency axis, Hz (Doppler), centred on 0
    t_axis : (n_time,) time axis, seconds
    """
    rx = cfg.get("receiver", {})

    window_ms      = float(rx.get("stft_window_ms", 100.0))
    window_type    = rx.get("window_type", "hann")
    overlap_pct    = float(rx.get("overlap_pct", 80.0))
    zero_pad_factor = int(rx.get("zero_pad_factor", 4))

    # If caller specified doppler_range_hz override, use it
    if doppler_range_hz is None or doppler_range_hz <= 0:
        doppler_range_hz = float(rx.get("doppler_range_hz", 400.0))

    win_samps = max(round(window_ms / 1000.0 * Fs_out), 16)
    noverlap  = round(overlap_pct / 100.0 * win_samps)
    # Zero-pad FFT size: next power of 2 >= zero_pad_factor * win_samps
    nfft = int(2 ** np.ceil(np.log2(zero_pad_factor * win_samps)))

    kaiser_beta = 6.0   # used only for kaiser window type
    window_arr  = _make_window(window_type, win_samps, kaiser_beta)

    print(f"\n[stft] Parameters:")
    print(f"  Fs_out:         {Fs_out:.0f} Hz")
    print(f"  Window:         {window_type}, {window_ms:.1f} ms "
          f"({win_samps} samples)")
    print(f"  Overlap:        {overlap_pct:.0f}% ({noverlap} samples)")
    print(f"  NFFT:           {nfft}  "
          f"(zero_pad_factor={zero_pad_factor})")
    print(f"  Freq resolution: {Fs_out/nfft:.3f} Hz/bin before crop")
    print(f"  Doppler crop:    +/-{doppler_range_hz:.0f} Hz")

    # scipy.signal.spectrogram returns frequencies in [0, Fs) when
    # return_onesided=False; fftshift centres it to [-Fs/2, Fs/2).
    f_spec, t_spec, Sxx = spectrogram(
        signal_10k,
        fs=Fs_out,
        window=window_arr,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        mode='complex',
    )

    # fftshift to centre spectrum around 0 Hz (Doppler = 0)
    Sxx    = np.fft.fftshift(Sxx, axes=0)
    f_spec = np.arange(-nfft // 2, nfft // 2) * Fs_out / nfft

    # Crop to Doppler range
    mask   = np.abs(f_spec) <= doppler_range_hz
    S_lin  = np.abs(Sxx[mask, :])
    f_axis = f_spec[mask]
    t_axis = t_spec

    S_dB = 20.0 * np.log10(S_lin + np.finfo(float).eps)

    return S_dB, S_lin, f_axis, t_axis


# ============================================================================
# Results saving and plotting
# ============================================================================
def save_results(prefix: str,
                 S_dB: np.ndarray,
                 S_lin: np.ndarray,
                 f_axis: np.ndarray,
                 t_axis: np.ndarray,
                 summary_meta: dict) -> None:
    """Save spectrogram arrays and summary JSON."""
    npz_path  = prefix + "_spectrogram.npz"
    json_path = prefix + "_summary.json"

    np.savez_compressed(npz_path,
                        S_dB=S_dB,
                        S_lin=S_lin,
                        f_axis=f_axis,
                        t_axis=t_axis)
    print(f"[save] Spectrogram: {npz_path}")

    with open(json_path, "w") as fh:
        json.dump(summary_meta, fh, indent=2)
    print(f"[save] Summary:     {json_path}")


def plot_spectrogram(prefix: str,
                     S_dB: np.ndarray,
                     f_axis: np.ndarray,
                     t_axis: np.ndarray,
                     dynamic_range_dB: float = 60.0) -> None:
    """Render and save a 2-D micro-Doppler spectrogram image.

    Colour axis is clipped to [peak - dynamic_range_dB, peak] so faint
    features are visible without being washed out by the DC spike or noise.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for headless environments
    import matplotlib.pyplot as plt

    peak_dB  = S_dB.max()
    vmin_dB  = peak_dB - dynamic_range_dB
    vmax_dB  = peak_dB

    fig, ax = plt.subplots(figsize=(12, 6))
    pcm = ax.pcolormesh(
        t_axis, f_axis, S_dB,
        shading="auto",
        cmap="inferno",
        vmin=vmin_dB,
        vmax=vmax_dB,
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Power (dB)", fontsize=11)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Doppler Frequency (Hz)", fontsize=12)
    ax.set_title("Micro-Doppler Spectrogram (De-hopped CW)", fontsize=13)
    ax.set_ylim(f_axis[0], f_axis[-1])

    plt.tight_layout()
    png_path = prefix + "_plot.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {png_path}")


# ============================================================================
# CLI argument parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i", required=True, metavar="FILE.cf32",
        help="Input .cf32 capture file (complex64 interleaved I/Q)")
    parser.add_argument(
        "--sidecar", default=None, metavar="SCHEDULE.json",
        help="TX sidecar JSON with hop_schedule.  "
             "If omitted, looks for <input base>_schedule.json.")
    parser.add_argument(
        "--config", default=None, metavar="CONFIG.json",
        help="cwhopping_config.json.  "
             "If omitted, looks in the same directory as --input.")
    parser.add_argument(
        "--output-prefix", default=None, metavar="PREFIX",
        help="Output file prefix for --save and --plot results.  "
             "Default: <input base>_results.")
    parser.add_argument(
        "--plot", action="store_true",
        help="Render spectrogram and save as <prefix>_plot.png")
    parser.add_argument(
        "--save", action="store_true",
        help="Save spectrogram arrays as <prefix>_spectrogram.npz "
             "and summary as <prefix>_summary.json")
    parser.add_argument(
        "--doppler-range", type=float, default=None,
        metavar="HZ",
        help="Crop spectrogram to +/-HZ (default: from config receiver section)")
    parser.add_argument(
        "--dynamic-range", type=float, default=60.0,
        metavar="dB",
        help="Colour axis dynamic range for plot in dB (default: 60)")

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    print("=" * 62)
    print("  CWHop RX -- De-Hop / Decimate / STFT Processor")
    print("=" * 62)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    input_path   = os.path.abspath(args.input)
    sidecar_path = os.path.abspath(args.sidecar
                                   if args.sidecar
                                   else _find_sidecar(input_path))
    config_path  = os.path.abspath(args.config
                                   if args.config
                                   else _find_config(input_path))

    if args.output_prefix:
        out_prefix = args.output_prefix
    else:
        base = input_path
        for suffix in ("_rx.cf32", ".cf32"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        out_prefix = base + "_results"

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.isfile(input_path):
        sys.exit(f"ERROR: Input file not found: {input_path}")

    file_bytes   = os.path.getsize(input_path)
    N_file       = file_bytes // 8    # complex64 = 8 bytes per sample
    duration_s   = N_file / FS_HZ

    print(f"\n[input]   {input_path}")
    print(f"          {file_bytes/1e6:.2f} MB  |  "
          f"{N_file:,} samples  |  {duration_s:.3f} s @ {FS_HZ/1e6:.3f} MHz")

    # ── Load sidecar and config ───────────────────────────────────────────────
    print(f"[sidecar] {sidecar_path}")
    sidecar = load_sidecar(sidecar_path)

    print(f"[config]  {config_path}")
    cfg = load_config(config_path)

    hop_schedule = sidecar.get("hop_schedule", [])
    if not hop_schedule:
        sys.exit("ERROR: hop_schedule is empty in sidecar file.")

    Fs_sidecar = sidecar.get("Fs_hz", FS_HZ)
    if Fs_sidecar != FS_HZ:
        print(f"[warn] Sidecar Fs={Fs_sidecar} differs from expected "
              f"{FS_HZ}.  Using sidecar value.")
    Fs = float(Fs_sidecar)

    print(f"[schedule] {len(hop_schedule)} dwells loaded")
    f_out = hop_schedule[0]["freq_hz"]
    print(f"  f_out (reference freq): {f_out:.1f} Hz")
    print(f"  First dwell: sample {hop_schedule[0]['start_sample']} "
          f"-- {hop_schedule[0]['end_sample']}  "
          f"@ {hop_schedule[0]['freq_hz']:.1f} Hz")
    print(f"  Last  dwell: sample {hop_schedule[-1]['start_sample']} "
          f"-- {hop_schedule[-1]['end_sample']}  "
          f"@ {hop_schedule[-1]['freq_hz']:.1f} Hz")

    # ── Open file via memmap (no full-file allocation) ─────────────────────────
    raw_mmap = np.memmap(input_path, dtype=np.complex64, mode='r')
    N_actual = len(raw_mmap)

    # Use the smaller of file length and sidecar-declared length
    N_process = min(N_actual, sidecar.get("N_total", N_actual))
    print(f"\n[memmap] File has {N_actual:,} samples; "
          f"processing {N_process:,} samples.")

    # ── Block-wise de-hop + decimate ──────────────────────────────────────────
    TOTAL_DECIM   = 1
    for q in DECIM_STAGES:
        TOTAL_DECIM *= q    # = 3072

    Fs_out = Fs / TOTAL_DECIM   # should be exactly 10000.0

    # Pre-allocate accumulation buffer at output rate
    N_out = int(np.ceil(N_process / TOTAL_DECIM))
    signal_10k = np.zeros(N_out, dtype=np.complex64)

    n_blocks    = int(np.ceil(N_process / BLOCK_SIZE))
    out_ptr     = 0   # write pointer into signal_10k

    print(f"\n[process] De-hopping + decimating in {n_blocks} block(s) "
          f"of {BLOCK_SIZE/1e6:.0f}M samples")
    print(f"  Decimation: {' x '.join(str(q) for q in DECIM_STAGES)} "
          f"= {TOTAL_DECIM}x   "
          f"{Fs/1e6:.3f} MHz -> {Fs_out:.0f} Hz")

    t0 = time.time()

    for blk in range(n_blocks):
        block_start = blk * BLOCK_SIZE
        block_end   = min(block_start + BLOCK_SIZE, N_process)
        n_blk       = block_end - block_start

        elapsed = time.time() - t0
        frac    = (blk + 1) / n_blocks
        eta     = elapsed / frac * (1.0 - frac) if frac > 0 else 0.0
        print(f"\n  Block {blk+1}/{n_blocks}  "
              f"[{block_start/1e6:.1f}M -- {block_end/1e6:.1f}M samples]  "
              f"ETA {eta:.1f} s")

        # ── Read raw block from memmap ────────────────────────────────────
        raw_block = np.array(raw_mmap[block_start:block_end],
                             dtype=np.complex64)

        # ── De-hop this block ─────────────────────────────────────────────
        print(f"    De-hopping ...", end=" ", flush=True)
        dehop_blk = dehop_block(raw_block, block_start, hop_schedule, Fs)
        print("done")

        # ── Decimate this block with margin ───────────────────────────────
        # Expected output length: floor(n_blk / TOTAL_DECIM)
        n_dec_expected = n_blk // TOTAL_DECIM

        print(f"    Decimating {n_blk:,} -> ~{n_dec_expected:,} samples ...")
        dec_block = _decimate_block_with_margin(
            block_dehop=dehop_blk,
            block_start=block_start,
            n_decimated_per_block=n_dec_expected,
            verbose=False,
        )

        # ── Accumulate into output buffer ─────────────────────────────────
        n_write = min(len(dec_block), N_out - out_ptr)
        signal_10k[out_ptr: out_ptr + n_write] = dec_block[:n_write]
        out_ptr += n_write

    # Trim unused trailing zeros if any blocks were shorter than BLOCK_SIZE
    signal_10k = signal_10k[:out_ptr]
    total_elapsed = time.time() - t0
    print(f"\n[process] Done in {total_elapsed:.1f} s")
    print(f"  De-hopped + decimated signal: {len(signal_10k):,} samples "
          f"@ {Fs_out:.0f} Hz  ({len(signal_10k)/Fs_out:.3f} s)")

    # ── Verbose decimation summary (staged rates) ─────────────────────────────
    print("\n[decimate] Staged rate summary:")
    Fs_cur = Fs
    for stage_idx, q in enumerate(DECIM_STAGES):
        Fs_cur /= q
        print(f"  Stage {stage_idx+1}: /{q}  ->  {Fs_cur/1e3:.3f} kHz")

    # ── STFT ──────────────────────────────────────────────────────────────────
    doppler_range = args.doppler_range or float(
        cfg.get("receiver", {}).get("doppler_range_hz", 400.0))

    S_dB, S_lin, f_axis, t_axis = compute_stft(
        signal_10k, Fs_out, cfg, doppler_range)

    print(f"\n[stft] Output shape: {S_dB.shape[1]} time frames x "
          f"{S_dB.shape[0]} Doppler bins")
    print(f"  Time resolution:  {np.diff(t_axis).mean()*1000:.2f} ms/frame")
    print(f"  Freq resolution:  {np.diff(f_axis).mean():.3f} Hz/bin")
    print(f"  Peak power:       {S_dB.max():.1f} dB")
    print(f"  Dynamic range:    "
          f"{S_dB.max() - np.percentile(S_dB, 10):.1f} dB")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save:
        rx_cfg = cfg.get("receiver", {})
        summary_meta = {
            "input_file":       input_path,
            "sidecar_file":     sidecar_path,
            "config_file":      config_path,
            "N_raw_samples":    int(N_process),
            "duration_s":       float(N_process / Fs),
            "Fs_hz":            float(Fs),
            "Fs_out_hz":        float(Fs_out),
            "decimation_stages": DECIM_STAGES,
            "total_decimation":  TOTAL_DECIM,
            "n_decimated_samples": int(len(signal_10k)),
            "stft_window_ms":   rx_cfg.get("stft_window_ms", 100.0),
            "window_type":      rx_cfg.get("window_type", "hann"),
            "overlap_pct":      rx_cfg.get("overlap_pct", 80.0),
            "zero_pad_factor":  rx_cfg.get("zero_pad_factor", 4),
            "doppler_range_hz": float(doppler_range),
            "spectrogram_shape": list(S_dB.shape),
            "n_time_frames":    int(S_dB.shape[1]),
            "n_doppler_bins":   int(S_dB.shape[0]),
            "peak_dB":          float(S_dB.max()),
        }
        save_results(out_prefix, S_dB, S_lin, f_axis, t_axis, summary_meta)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        plot_spectrogram(out_prefix, S_dB, f_axis, t_axis, args.dynamic_range)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n[summary]")
    print(f"  Input:         {input_path}  ({file_bytes/1e6:.2f} MB)")
    print(f"  Sidecar:       {sidecar_path}")
    print(f"  Dwells:        {len(hop_schedule)}")
    print(f"  Duration:      {N_process/Fs:.3f} s")
    print(f"  Fs_out:        {Fs_out:.0f} Hz  (after {TOTAL_DECIM}x decimation)")
    print(f"  STFT frames:   {S_dB.shape[1]}")
    print(f"  Doppler bins:  {S_dB.shape[0]}  "
          f"(+/-{doppler_range:.0f} Hz)")

    if not args.save and not args.plot:
        print("\n[hint] Add --save and/or --plot to write output files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
