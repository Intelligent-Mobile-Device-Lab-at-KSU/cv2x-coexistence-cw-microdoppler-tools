#!/usr/bin/env python3
"""
iq_spectrogram.py -- IQ spectrogram viewer

Display STFT spectrogram of any .cf32 or .bin IQ file.
Optionally overlay active subframe markers and CW frequency from a sidecar.

Examples:
    python iq_spectrogram.py cv2x_iq.cf32
    python iq_spectrogram.py cv2x_iq_cw.cf32 -r 30.72e6 --fft 1024 --step 256
    python iq_spectrogram.py cv2x_iq.cf32 --sidecar cv2x_iq_cw.json --duration 5
    python iq_spectrogram.py test_hopper_tx.cf32 -r 30.72e6 --start 0.016 --duration 0.01
    python iq_spectrogram.py test_hopper_rx.cf32 -r 192000 --fft 64 --step 16
"""

import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="STFT spectrogram viewer for IQ files (.cf32 / .bin)")
    p.add_argument("input", help="IQ file path (.cf32 or .bin, complex64)")
    p.add_argument("-r", "--rate", type=float, default=30.72e6,
                   help="Sample rate in Hz (default: 30.72e6)")
    p.add_argument("--fft", type=int, default=512,
                   help="FFT window size in samples (default: 512)")
    p.add_argument("--step", type=int, default=None,
                   help="Hop size in samples (default: fft/4)")
    p.add_argument("--window", choices=["hann", "hamming", "blackman", "rect"],
                   default="hann", help="Window function (default: hann)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start time in seconds (default: 0)")
    p.add_argument("--duration", type=float, default=None,
                   help="Duration in seconds (default: whole file)")
    p.add_argument("--decimate", type=int, default=1,
                   help="Decimation factor to speed up display (default: 1). "
                        "E.g. --decimate 16 at 30.72 Msps -> 1.92 Msps")
    p.add_argument("--max-frames", type=int, default=4000,
                   help="Max STFT time frames -- auto-increases step if "
                        "needed for speed (default: 4000)")
    p.add_argument("--vmin", type=float, default=None,
                   help="Colorbar min in dB (default: auto)")
    p.add_argument("--vmax", type=float, default=None,
                   help="Colorbar max in dB (default: auto)")
    p.add_argument("--cmap", default="viridis",
                   help="Matplotlib colormap (default: viridis)")
    p.add_argument("--save", type=str, default=None,
                   help="Save plot to file instead of showing")
    p.add_argument("--title", type=str, default=None,
                   help="Custom plot title")
    p.add_argument("--db-range", type=float, default=60,
                   help="Dynamic range in dB below peak (default: 60)")
    p.add_argument("--sidecar", type=str, default=None,
                   help="Sidecar JSON to overlay active subframes and CW")
    p.add_argument("--iq-correct", action="store_true", default=False,
                   help="Apply offline IQ imbalance correction to remove "
                        "ghost mirror image")
    return p.parse_args()


def iq_imbalance_correct(iq):
    """
    Estimate and correct IQ gain/phase imbalance.

    Uses the correlation method: in a balanced receiver, I and Q are
    uncorrelated.  Any correlation indicates phase imbalance, and any
    difference in variance indicates gain imbalance.

    Model:  received = true + alpha * conj(true)
    Estimate alpha from the data, then invert.
    """
    # Estimate image rejection coefficient alpha = E[z^2] / E[|z|^2]
    # (Assumes signal is roughly symmetric in spectrum)
    n = min(len(iq), 2_000_000)  # use up to 2M samples for estimate
    z = iq[:n].astype(np.complex128)
    alpha = np.mean(z ** 2) / np.mean(np.abs(z) ** 2)
    rejection_db = -20 * np.log10(np.abs(alpha) + 1e-30)
    print(f"IQ corr: |alpha|={np.abs(alpha):.4f} "
          f"({rejection_db:.1f} dB image rejection before correction)")

    if np.abs(alpha) < 1e-6:
        print("IQ corr: imbalance negligible, skipping")
        return iq

    # Correct: z_corrected = (z - alpha * conj(z)) / (1 - |alpha|^2)
    scale = 1.0 / (1.0 - np.abs(alpha) ** 2)
    iq_out = (iq.astype(np.complex128) - alpha * np.conj(iq)) * scale
    return iq_out.astype(np.complex64)


def load_iq(path, rate, start_s, duration_s, decimate,
            max_samples=200_000_000):
    """Load and optionally decimate a slice of the IQ file."""
    itemsize = np.dtype(np.complex64).itemsize
    file_bytes = os.path.getsize(path)
    total_samples = file_bytes // itemsize

    i_start = int(start_s * rate)
    if i_start >= total_samples:
        print(f"ERROR: start ({start_s:.3f} s = sample {i_start}) "
              f"is past end of file ({total_samples} samples)")
        sys.exit(1)

    if duration_s is not None:
        n_read = min(int(duration_s * rate), total_samples - i_start)
    else:
        n_read = total_samples - i_start

    n_read = min(n_read, max_samples)

    print(f"File:     {path}")
    print(f"Total:    {total_samples:,} samples "
          f"({total_samples / rate:.3f} s)")
    print(f"Loading:  {n_read:,} samples "
          f"({n_read / rate:.6f} s) from t={start_s:.6f} s")

    iq = np.fromfile(path, dtype=np.complex64, count=n_read,
                     offset=i_start * itemsize)

    if decimate > 1:
        # Low-pass filter before decimation to avoid aliasing
        from scipy.signal import decimate as sp_decimate
        # scipy.signal.decimate works on real; do I/Q separately
        n_orig = len(iq)
        iq = sp_decimate(iq.real, decimate).astype(np.float32) + \
             1j * sp_decimate(iq.imag, decimate).astype(np.float32)
        iq = iq.astype(np.complex64)
        new_rate = rate / decimate
        print(f"Decimated {decimate}x: {len(iq):,} samples, "
              f"effective rate {new_rate/1e6:.3f} MHz")
        return iq, new_rate

    return iq, rate


def compute_stft(iq, fft_size, step, window_name, eff_rate, max_frames):
    """Compute STFT magnitude in dB."""
    if window_name == "hann":
        win = np.hanning(fft_size)
    elif window_name == "hamming":
        win = np.hamming(fft_size)
    elif window_name == "blackman":
        win = np.blackman(fft_size)
    else:
        win = np.ones(fft_size)

    # Auto-increase step if too many frames
    n_frames = (len(iq) - fft_size) // step + 1
    if n_frames > max_frames:
        step = max(step, (len(iq) - fft_size) // max_frames)
        n_frames = (len(iq) - fft_size) // step + 1
        print(f"  (auto step -> {step} to cap at {max_frames} frames)")

    if n_frames < 1:
        print(f"ERROR: not enough samples ({len(iq)}) for one FFT "
              f"window ({fft_size})")
        sys.exit(1)

    freq_res = eff_rate / fft_size
    time_res = step / eff_rate
    print(f"FFT:      {fft_size} samples  "
          f"({1e6 * fft_size / eff_rate:.1f} us)")
    print(f"Step:     {step} samples  "
          f"({1e6 * step / eff_rate:.1f} us)")
    print(f"Frames:   {n_frames:,}")
    print(f"Freq res: {freq_res:.1f} Hz")
    print(f"Time res: {1e6 * time_res:.1f} us")

    # Chunked STFT to limit memory
    chunk = 2000
    results = []
    for c_start in range(0, n_frames, chunk):
        c_end = min(c_start + chunk, n_frames)
        c_n = c_end - c_start
        indices = (np.arange(fft_size)[None, :]
                   + (np.arange(c_start, c_end) * step)[:, None])
        frames = iq[indices] * win[None, :]
        spectra = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
        results.append(20.0 * np.log10(np.abs(spectra) + 1e-30))

    mag_db = np.concatenate(results, axis=0)
    return mag_db, n_frames, step


def load_sidecar(path):
    """Load sidecar JSON and extract overlay info."""
    with open(path, "r") as f:
        data = json.load(f)
    cw = data.get("cw_inject", {})
    active_sfs = cw.get("active_subframes", [])
    f_cw_hz = cw.get("f_cw_hz", None)
    num_total = cw.get("num_total", 0)
    return active_sfs, f_cw_hz, num_total


def plot_spectrogram(mag_db, eff_rate, fft_size, step, start_s, args,
                     sidecar_info=None):
    """Plot and optionally save the spectrogram."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    n_frames = mag_db.shape[0]

    # Time and frequency axes
    t_start = start_s
    t_end = start_s + (n_frames - 1) * step / eff_rate
    f_start = -eff_rate / 2
    f_end = eff_rate / 2

    # Auto scale
    vmax = args.vmax if args.vmax is not None else float(np.max(mag_db))
    vmin = args.vmin if args.vmin is not None else vmax - args.db_range

    # Choose sensible axis units
    duration = t_end - t_start
    if duration < 0.1:
        t_scale, t_unit = 1e3, "ms"
    else:
        t_scale, t_unit = 1, "s"

    if eff_rate > 1e6:
        f_scale, f_unit = 1e6, "MHz"
    elif eff_rate > 1e3:
        f_scale, f_unit = 1e3, "kHz"
    else:
        f_scale, f_unit = 1, "Hz"

    extent = [t_start * t_scale, t_end * t_scale,
              f_start / f_scale, f_end / f_scale]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mag_db.T, aspect="auto", origin="lower",
                   extent=extent, vmin=vmin, vmax=vmax,
                   cmap=args.cmap, interpolation="nearest")
    ax.set_xlabel(f"Time ({t_unit})")
    ax.set_ylabel(f"Frequency ({f_unit})")

    title = args.title or os.path.basename(args.input)
    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Power (dB)")

    # -- Sidecar overlay --
    if sidecar_info is not None:
        active_sfs, f_cw_hz, num_total = sidecar_info

        # CW frequency line (red dashed)
        if f_cw_hz is not None and abs(f_cw_hz) <= eff_rate / 2:
            ax.axhline(f_cw_hz / f_scale, color="red", ls="--",
                       linewidth=1.5, alpha=0.9, label=f"CW {f_cw_hz/1e3:.1f} kHz")

        # Active subframe markers (semi-transparent green bars at top)
        sf_dur_s = 0.001
        y_lo = extent[2]
        y_hi = extent[3]
        bar_height = (y_hi - y_lo) * 0.02  # thin bar at top

        patches = []
        for sf_idx in active_sfs:
            t_sf = sf_idx * sf_dur_s
            if t_sf < start_s or t_sf > start_s + duration:
                continue
            x = t_sf * t_scale
            w = sf_dur_s * t_scale
            patches.append(Rectangle((x, y_hi - bar_height), w, bar_height))

        if patches:
            pc = PatchCollection(patches, facecolor="lime", edgecolor="none",
                                 alpha=0.7)
            ax.add_collection(pc)
            ax.plot([], [], color="lime", linewidth=4, alpha=0.7,
                    label=f"Active SFs ({len(patches)} visible)")

        ax.legend(loc="upper right", fontsize=9,
                  facecolor="black", edgecolor="white",
                  labelcolor="white", framealpha=0.7)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    step = args.step if args.step is not None else args.fft // 4

    iq, eff_rate = load_iq(args.input, args.rate, args.start, args.duration,
                           args.decimate)

    if args.iq_correct:
        iq = iq_imbalance_correct(iq)

    mag_db, n_frames, step = compute_stft(
        iq, args.fft, step, args.window, eff_rate, args.max_frames)

    sidecar_info = None
    if args.sidecar:
        if not os.path.isfile(args.sidecar):
            print(f"WARNING: sidecar not found: {args.sidecar}")
        else:
            sidecar_info = load_sidecar(args.sidecar)
            active_sfs, f_cw_hz, num_total = sidecar_info
            print(f"Sidecar:  {len(active_sfs)} active SFs, "
                  f"CW={f_cw_hz/1e3:.1f} kHz" if f_cw_hz else
                  f"Sidecar:  {len(active_sfs)} active SFs")

    plot_spectrogram(mag_db, eff_rate, args.fft, step, args.start, args,
                     sidecar_info)
