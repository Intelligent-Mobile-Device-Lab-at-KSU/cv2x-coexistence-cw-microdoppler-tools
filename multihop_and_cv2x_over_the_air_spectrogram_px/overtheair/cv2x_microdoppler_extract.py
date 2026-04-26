#!/usr/bin/env python3
"""
cv2x_microdoppler_extract.py -- Offline Micro-Doppler Extraction
================================================================
Reads a narrow-band RX capture centered on a C-V2X virtual sensing CW tone
and extracts micro-Doppler via STFT spectrogram and phase tracking.

Input:  narrow-band IQ file (.cf32) from cv2x_cw_txrx.py
        + sidecar JSON from the CW-injected TX file (cv2x_iq_cw.json)

Processing chain:
  1. Find exact CW peak via coarse FFT
  2. Mix CW tone to DC
  3. Calibrate out static LO offset between TX/RX daughtercards
  4. Compute STFT spectrogram (Doppler vs time)
  5. Phase tracking at CW frequency for fine Doppler estimation
  6. Subframe gating to highlight active windows

Example:
    python cv2x_microdoppler_extract.py \\
        --input cv2x_rx_doppler.cf32 \\
        --sidecar cv2x_iq_cw.json \\
        --fft-size 19200 --overlap 0.9 \\
        --doppler-range 500 --plot --save
"""

import argparse
import json
import os
import sys
import numpy as np


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True,
                        help="RX capture file (.cf32)")
    parser.add_argument("--sidecar", "-s", default=None,
                        help="Path to TX sidecar JSON (cv2x_iq_cw.json). "
                             "Auto-derived from --input if omitted.")
    parser.add_argument("--rate", "-r", type=float, default=None,
                        help="RX sample rate in Hz (default: from RX sidecar)")
    parser.add_argument("--fft-size", type=int, default=None,
                        help="STFT window length in samples. Default: "
                             "rate/10 (100 ms window = 10 Hz resolution)")
    parser.add_argument("--overlap", type=float, default=0.9,
                        help="STFT overlap fraction (default: 0.9)")
    parser.add_argument("--window", default="blackmanharris",
                        choices=["blackmanharris", "hann", "hamming"],
                        help="STFT window function (default: blackmanharris)")
    parser.add_argument("--doppler-range", type=float, default=500.0,
                        help="Max Doppler frequency to display in Hz "
                             "(default: 500)")
    parser.add_argument("--cw-freq", type=float, default=None,
                        help="CW baseband frequency override in Hz "
                             "(default: from sidecar)")
    parser.add_argument("--lo-cal", action="store_true", default=True,
                        help="Calibrate out static LO offset (default: on)")
    parser.add_argument("--no-lo-cal", dest="lo_cal", action="store_false")
    parser.add_argument("--active-only", action="store_true", default=True,
                        help="Concatenate only active subframes for STFT "
                             "(default: on). Removes noise-only gaps and "
                             "uses phase-coherent CW samples only.")
    parser.add_argument("--no-active-only", dest="active_only",
                        action="store_false",
                        help="Use full continuous capture for STFT")
    parser.add_argument("--gate", action="store_true", default=True,
                        help="Enable subframe gating (default: on)")
    parser.add_argument("--no-gate", dest="gate", action="store_false")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Show diagnostic plots")
    parser.add_argument("--save", action="store_true",
                        help="Save results to .npz and .json files")
    parser.add_argument("--output-prefix", default="doppler_results",
                        help="Prefix for output files (default: doppler_results)")
    return parser.parse_args()


# ============================================================================
# Sidecar helpers
# ============================================================================
def _sidecar_path(iq_file):
    base, _ = os.path.splitext(iq_file)
    return base + ".json"


def _read_sidecar(path):
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ============================================================================
# Signal processing
# ============================================================================
def find_cw_peak(iq, rate, expected_freq=None, search_bw=10000):
    """
    Find the exact CW tone frequency via a long FFT.

    If expected_freq is given, search within +/-search_bw Hz of it.
    Returns the measured CW frequency in Hz.
    """
    # Use up to 2 seconds of data for good resolution
    n_samp = min(len(iq), int(rate * 2))
    spectrum = np.fft.fft(iq[:n_samp])
    freqs = np.fft.fftfreq(n_samp, 1.0 / rate)

    mag = np.abs(spectrum)

    if expected_freq is not None:
        # Restrict search to +/-search_bw around expected
        mask = np.abs(freqs - expected_freq) < search_bw
        if np.any(mask):
            idx_offset = np.where(mask)[0]
            peak_local = np.argmax(mag[mask])
            peak_idx = idx_offset[peak_local]
        else:
            peak_idx = np.argmax(mag)
    else:
        peak_idx = np.argmax(mag)

    f_measured = freqs[peak_idx]
    peak_power_db = 20 * np.log10(mag[peak_idx] / n_samp + 1e-30)
    print(f"  CW peak at {f_measured:.1f} Hz  "
          f"(power: {peak_power_db:.1f} dB)")
    return f_measured


def mix_to_dc(iq, f_cw, rate):
    """
    Frequency-shift the CW tone to DC.

    Uses sample-index-based phase computation to avoid float precision
    issues when multiplying large time values by frequency.
    """
    n = np.arange(len(iq), dtype=np.float64)
    phase_per_sample = -2.0 * np.pi * f_cw / rate
    return iq * np.exp(1j * phase_per_sample * n).astype(np.complex64)


def calibrate_lo_offset(iq, rate):
    """
    Estimate and remove static LO frequency offset.
    Returns (corrected_iq, offset_hz).
    """
    # Instantaneous frequency via phase derivative
    phase = np.unwrap(np.angle(iq))
    # Use blocks to avoid memory issues on large arrays
    block_size = min(len(iq), int(rate * 5))  # 5 second blocks
    inst_freqs = []
    for start in range(0, len(iq) - 1, block_size):
        end = min(start + block_size, len(iq) - 1)
        dphase = np.diff(phase[start:end + 1])
        inst_freq = dphase * rate / (2.0 * np.pi)
        inst_freqs.append(np.median(inst_freq))

    f_offset = float(np.median(inst_freqs))
    print(f"  LO offset: {f_offset:.2f} Hz")

    # Correct using sample-index-based phase (avoids float precision issues)
    n = np.arange(len(iq), dtype=np.float64)
    phase_per_sample = -2.0 * np.pi * f_offset / rate
    iq_corrected = iq * np.exp(1j * phase_per_sample * n).astype(
        np.complex64)
    return iq_corrected, f_offset


def compute_stft(iq, rate, nperseg, noverlap, window_name):
    """
    Compute STFT spectrogram.
    Returns (frequencies, times, power_db).
    """
    from scipy.signal import stft

    f, t, Zxx = stft(iq, fs=rate, window=window_name,
                     nperseg=nperseg, noverlap=noverlap,
                     return_onesided=False)
    # fftshift so DC is in the center
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)

    power_db = 20.0 * np.log10(np.abs(Zxx) + 1e-30)
    return f, t, power_db


def compute_cw_power_vs_time(iq, rate, window_ms=1.0):
    """
    Compute short-time power at DC (where CW tone has been shifted).
    Uses window_ms millisecond windows.
    """
    window_samps = max(1, int(rate * window_ms / 1000.0))
    n_windows = len(iq) // window_samps
    times = np.arange(n_windows) * window_ms / 1000.0
    powers = np.empty(n_windows)
    for i in range(n_windows):
        chunk = iq[i * window_samps:(i + 1) * window_samps]
        powers[i] = np.mean(np.abs(chunk) ** 2)
    powers_db = 10.0 * np.log10(powers + 1e-30)
    return times, powers_db


def phase_tracking(iq, rate, active_sf_times, sf_duration_s=0.001):
    """
    Extract CW complex amplitude at each active subframe time,
    track phase, and compute instantaneous Doppler.

    active_sf_times: array of start times (seconds) for each active subframe
                     in the RX capture timebase.
    sf_duration_s: subframe duration (1 ms).

    Returns (times, phases_unwrapped, doppler_hz).
    """
    sf_samps = int(round(rate * sf_duration_s))
    amplitudes = []
    valid_times = []

    for t_start in active_sf_times:
        i_start = int(round(t_start * rate))
        i_end = i_start + sf_samps
        if i_start < 0 or i_end > len(iq):
            continue
        chunk = iq[i_start:i_end]
        # Complex amplitude at DC (CW has been shifted to DC)
        A = np.mean(chunk)
        amplitudes.append(A)
        valid_times.append(t_start)

    if len(amplitudes) < 2:
        return np.array([]), np.array([]), np.array([])

    amplitudes = np.array(amplitudes)
    valid_times = np.array(valid_times)
    phases = np.unwrap(np.angle(amplitudes))

    # Instantaneous Doppler from phase differences
    dt = np.diff(valid_times)
    dphi = np.diff(phases)
    doppler_hz = dphi / (2.0 * np.pi * dt)

    return valid_times, phases, doppler_hz


def compute_active_sf_times(rx_sidecar, tx_sidecar, rx_duration_s):
    """
    Map active subframe indices from the TX file to RX capture times.

    The TX file loops. Active subframe indices are relative to the TX file.
    We need to map them to absolute RX capture times.

    Returns array of times (seconds) in the RX capture timebase.
    """
    cw_info = tx_sidecar.get("cw_inject", {})
    active_sfs = cw_info.get("active_subframes", [])
    num_total = cw_info.get("num_total", 0)

    if not active_sfs or num_total == 0:
        print("  WARNING: No active subframe info in sidecar")
        return np.array([])

    # TX file duration in ms (= num_total subframes)
    tx_duration_ms = num_total  # each subframe = 1 ms

    # Generate active subframe times for the full RX duration
    # TX file loops, so active subframes repeat every tx_duration_ms
    rx_duration_ms = rx_duration_s * 1000.0
    n_loops = int(np.ceil(rx_duration_ms / tx_duration_ms)) + 1

    all_times = []
    for loop_idx in range(n_loops):
        offset_ms = loop_idx * tx_duration_ms
        for sf_idx in active_sfs:
            t_ms = offset_ms + sf_idx
            if t_ms < rx_duration_ms:
                all_times.append(t_ms / 1000.0)

    return np.array(sorted(all_times))


def detect_cw_onoff(iq, rate, threshold_db=6.0, window_ms=1.0):
    """
    Blind detection of CW on/off pattern from the RX data.
    Returns array of start times (seconds) where CW power exceeds threshold.
    """
    times, powers_db = compute_cw_power_vs_time(iq, rate, window_ms)
    median_power = np.median(powers_db)
    active = powers_db > (median_power + threshold_db)
    active_times = times[active]
    print(f"  Blind detection: {len(active_times)} active windows "
          f"out of {len(times)} (threshold: {median_power:.1f} + "
          f"{threshold_db:.1f} dB)")
    return active_times


def concatenate_active_subframes(iq, rate, active_sf_times,
                                 sf_duration_s=0.001):
    """
    Extract and concatenate only the active subframe chunks from the IQ data.

    Since the CW injection maintained phase continuity across subframes
    (phase advances as if the tone were always on), the concatenated signal
    preserves phase coherence.  The Doppler information accumulated during
    the gaps is encoded in the phase jump between consecutive chunks.

    Returns:
        iq_concat:      concatenated complex IQ (active chunks only)
        original_times: array of original start times for each chunk
                        (for mapping back to real time)
        chunk_size:     samples per chunk
    """
    chunk_size = int(round(rate * sf_duration_s))
    chunks = []
    valid_times = []

    for t_start in active_sf_times:
        i_start = int(round(t_start * rate))
        i_end = i_start + chunk_size
        if i_start < 0 or i_end > len(iq):
            continue
        chunks.append(iq[i_start:i_end])
        valid_times.append(t_start)

    if not chunks:
        return np.array([], dtype=np.complex64), np.array([]), chunk_size

    iq_concat = np.concatenate(chunks)
    return iq_concat, np.array(valid_times), chunk_size


# ============================================================================
# Plotting
# ============================================================================
def make_plots(f_stft, t_stft, power_db, doppler_range,
               cw_times, cw_powers_db,
               phase_times, phases, doppler_hz,
               active_sf_times, output_prefix, rate):
    """Generate 4-panel diagnostic figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("C-V2X Micro-Doppler Extraction", fontsize=14)

    # 1. STFT Spectrogram
    ax = axes[0, 0]
    # Trim to doppler range
    freq_mask = np.abs(f_stft) <= doppler_range
    f_trim = f_stft[freq_mask]
    pwr_trim = power_db[freq_mask, :]

    # Auto scale: use percentile for dynamic range
    vmin = np.percentile(pwr_trim, 5)
    vmax = np.percentile(pwr_trim, 99)

    pcm = ax.pcolormesh(t_stft, f_trim, pwr_trim,
                        shading='nearest', cmap='jet',
                        vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label='Power (dB)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler Frequency (Hz)")
    ax.set_title("STFT Micro-Doppler Spectrogram")

    # Mark active subframe times
    if len(active_sf_times) > 0 and len(active_sf_times) < 200:
        for t_sf in active_sf_times:
            if t_sf <= t_stft[-1]:
                ax.axvline(t_sf, color='white', alpha=0.1, linewidth=0.3)

    # 2. CW Power vs Time
    ax = axes[0, 1]
    ax.plot(cw_times, cw_powers_db, linewidth=0.5, color='tab:blue',
            label='CW power')
    if len(active_sf_times) > 0:
        # Show expected active windows as stems
        median_pwr = np.median(cw_powers_db)
        peak_pwr = np.percentile(cw_powers_db, 95)
        sf_display = active_sf_times[active_sf_times <= cw_times[-1]]
        if len(sf_display) < 500:
            ax.vlines(sf_display, median_pwr, peak_pwr,
                      colors='green', alpha=0.3, linewidth=0.5,
                      label='Active subframes')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("CW Tone Power vs Time")
    ax.legend(fontsize=8)

    # 3. Phase Tracking
    ax = axes[1, 0]
    if len(phase_times) > 1:
        ax2 = ax.twinx()
        ax.plot(phase_times, phases, linewidth=0.5, color='tab:blue',
                label='Unwrapped phase')
        ax.set_ylabel("Phase (rad)", color='tab:blue')

        # Doppler on secondary axis
        t_doppler = 0.5 * (phase_times[:-1] + phase_times[1:])
        ax2.plot(t_doppler, doppler_hz, linewidth=0.5, color='tab:red',
                 alpha=0.7, label='Inst. Doppler')
        ax2.set_ylabel("Doppler (Hz)", color='tab:red')
        ax2.set_ylim(-doppler_range, doppler_range)

        ax.set_xlabel("Time (s)")
        ax.set_title("Phase Tracking (per active subframe)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient phase data",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Phase Tracking")

    # 4. Doppler Histogram
    ax = axes[1, 1]
    if len(doppler_hz) > 0:
        # Clip to range
        d_clip = doppler_hz[np.abs(doppler_hz) < doppler_range]
        if len(d_clip) > 0:
            ax.hist(d_clip, bins=100, color='tab:blue', alpha=0.7,
                    edgecolor='none')
            ax.axvline(0, color='red', ls='--', linewidth=1)
            ax.set_xlabel("Doppler Shift (Hz)")
            ax.set_ylabel("Count")
            mean_d = np.mean(d_clip)
            std_d = np.std(d_clip)
            ax.set_title(f"Doppler Distribution "
                         f"(mean={mean_d:.1f} Hz, std={std_d:.1f} Hz)")
        else:
            ax.text(0.5, 0.5, "All Doppler outside range",
                    ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No Doppler data",
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plot_file = f"{output_prefix}_plots.png"
    plt.savefig(plot_file, dpi=200)
    print(f"[plot] Saved {plot_file}")
    plt.show()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # -- Load data -----------------------------------------------------------
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    print("=" * 60)
    print("  C-V2X Micro-Doppler Extraction")
    print("=" * 60)

    # Read RX sidecar
    rx_sidecar = _read_sidecar(_sidecar_path(args.input))

    # Read TX sidecar
    if args.sidecar:
        tx_sidecar = _read_sidecar(args.sidecar)
    elif rx_sidecar and "tx_sidecar_ref" in rx_sidecar:
        tx_sidecar = _read_sidecar(rx_sidecar["tx_sidecar_ref"])
    else:
        # Try default name
        tx_sidecar = _read_sidecar("cv2x_iq_cw.json")

    # Determine sample rate
    if args.rate is not None:
        rate = args.rate
    elif rx_sidecar and "rx_rate" in rx_sidecar:
        rate = rx_sidecar["rx_rate"]
    else:
        rate = 192000.0
        print(f"  WARNING: No rate info found, assuming {rate:.0f} Hz")

    # Load IQ
    iq = np.fromfile(args.input, dtype=np.complex64)
    duration_s = len(iq) / rate
    file_size_mb = os.path.getsize(args.input) / 1e6

    print(f"\nInput:      {args.input}")
    print(f"File size:  {file_size_mb:.2f} MB  "
          f"({len(iq):,} samples, {duration_s:.3f} s)")
    print(f"Rate:       {rate:.0f} Hz")

    # Determine expected CW baseband frequency
    if args.cw_freq is not None:
        expected_cw = args.cw_freq
    elif rx_sidecar and "cw_baseband_hz" in rx_sidecar:
        expected_cw = rx_sidecar["cw_baseband_hz"]
    else:
        expected_cw = None
    if expected_cw is not None:
        print(f"Expected CW: {expected_cw:.0f} Hz (baseband)")

    # -- Step 1: Find exact CW peak -----------------------------------------
    print(f"\n--- Step 1: CW tone detection ---")
    f_cw = find_cw_peak(iq, rate, expected_freq=expected_cw)

    # -- Step 2: Mix to DC ---------------------------------------------------
    print(f"\n--- Step 2: Mix CW to DC ---")
    iq_shifted = mix_to_dc(iq, f_cw, rate)
    print(f"  Shifted by {-f_cw:.1f} Hz")

    # -- Step 3: LO offset calibration ---------------------------------------
    f_offset = 0.0
    if args.lo_cal:
        print(f"\n--- Step 3: LO offset calibration ---")
        iq_shifted, f_offset = calibrate_lo_offset(iq_shifted, rate)
        print(f"  Residual after calibration: ~0 Hz")

    # -- Step 4: Subframe timing (needed before STFT for active-only mode) ---
    print(f"\n--- Step 4: Subframe timing ---")
    active_sf_times = np.array([])

    if tx_sidecar:
        active_sf_times = compute_active_sf_times(
            rx_sidecar, tx_sidecar, duration_s)
        print(f"  Mapped {len(active_sf_times)} active subframe windows "
              f"to RX timebase")

    if len(active_sf_times) == 0:
        print("  Falling back to blind CW detection...")
        active_sf_times = detect_cw_onoff(iq_shifted, rate)

    # -- Step 5: CW power vs time --------------------------------------------
    print(f"\n--- Step 5: CW power tracking ---")
    cw_times, cw_powers_db = compute_cw_power_vs_time(iq_shifted, rate,
                                                       window_ms=1.0)
    median_pwr = np.median(cw_powers_db)
    peak_pwr = np.max(cw_powers_db)
    print(f"  Power range: {median_pwr:.1f} to {peak_pwr:.1f} dB")

    # -- Step 6: STFT spectrogram --------------------------------------------
    # Choose input signal: active-only concatenation or full capture
    if args.active_only and len(active_sf_times) > 0:
        print(f"\n--- Step 6: Active-only STFT spectrogram ---")
        iq_concat, concat_times, chunk_size = concatenate_active_subframes(
            iq_shifted, rate, active_sf_times)
        n_chunks = len(concat_times)
        concat_duration = len(iq_concat) / rate
        print(f"  Concatenated {n_chunks} active subframes "
              f"({chunk_size} samp/chunk)")
        print(f"  Concat length: {len(iq_concat):,} samples "
              f"({concat_duration * 1000:.1f} ms)")

        # For active-only mode, use a smaller default FFT that spans
        # multiple subframe chunks for good Doppler resolution.
        # Each chunk is ~192 samples at 192 kSps (1 ms).
        # A window of 50 chunks = 9600 samples = 50 ms of "active time"
        # -> freq resolution = 192000 / 9600 = 20 Hz
        if args.fft_size is not None:
            nperseg = args.fft_size
        else:
            # Default: span 50 subframes' worth of active data
            nperseg = min(chunk_size * 50, len(iq_concat))
        noverlap = int(nperseg * args.overlap)
        hop = nperseg - noverlap
        freq_res = rate / nperseg
        n_chunks_per_window = nperseg / chunk_size

        print(f"  FFT size:    {nperseg} samples "
              f"(~{n_chunks_per_window:.0f} subframes)")
        print(f"  Overlap:     {args.overlap * 100:.0f}%  "
              f"(hop = {hop} samples)")
        print(f"  Freq res:    {freq_res:.2f} Hz")

        stft_input = iq_concat

        # Build a time axis that maps STFT bins back to real time.
        # Each STFT bin at sample index i in the concatenated signal
        # corresponds to real time concat_times[i // chunk_size].
        # For the time axis, interpolate between chunk original times.
        def _concat_sample_to_real_time(sample_idx):
            chunk_idx = min(int(sample_idx / chunk_size), n_chunks - 1)
            return concat_times[chunk_idx]
    else:
        print(f"\n--- Step 6: Full-capture STFT spectrogram ---")
        if args.fft_size is not None:
            nperseg = args.fft_size
        else:
            nperseg = int(rate / 10)  # 100 ms window
        noverlap = int(nperseg * args.overlap)
        hop = nperseg - noverlap
        freq_res = rate / nperseg
        time_step = hop / rate

        print(f"  FFT size:    {nperseg} samples "
              f"({nperseg / rate * 1000:.1f} ms)")
        print(f"  Overlap:     {args.overlap * 100:.0f}%  "
              f"(hop = {hop} samples)")
        print(f"  Freq res:    {freq_res:.2f} Hz")
        print(f"  Time step:   {time_step * 1000:.1f} ms")
        stft_input = iq_shifted
        _concat_sample_to_real_time = None

    f_stft, t_stft, power_db = compute_stft(
        stft_input, rate, nperseg, noverlap, args.window)
    print(f"  Spectrogram: {power_db.shape[0]} freq bins x "
          f"{power_db.shape[1]} time bins")

    # Map STFT time axis back to real time for active-only mode
    if args.active_only and _concat_sample_to_real_time is not None:
        # t_stft is in "concatenated time" (seconds into concat signal).
        # Map each bin to the original real time.
        t_stft_real = np.array([
            _concat_sample_to_real_time(t * rate) for t in t_stft])
        print(f"  Real time span: {t_stft_real[0]:.3f} - "
              f"{t_stft_real[-1]:.3f} s")
    else:
        t_stft_real = t_stft

    # -- Step 7: Phase tracking ----------------------------------------------
    print(f"\n--- Step 7: Phase tracking ---")
    phase_times, phases, doppler_hz = phase_tracking(
        iq_shifted, rate, active_sf_times)

    if len(doppler_hz) > 0:
        d_valid = doppler_hz[np.abs(doppler_hz) < args.doppler_range]
        if len(d_valid) > 0:
            print(f"  Doppler stats: mean={np.mean(d_valid):.1f} Hz, "
                  f"std={np.std(d_valid):.1f} Hz, "
                  f"range=[{np.min(d_valid):.1f}, {np.max(d_valid):.1f}] Hz")
        else:
            print(f"  All Doppler estimates outside +/-{args.doppler_range} Hz")
    else:
        print("  No phase tracking data (insufficient active subframes)")

    # -- Save results --------------------------------------------------------
    if args.save:
        print(f"\n--- Saving results ---")
        # Spectrogram
        spec_file = f"{args.output_prefix}_spectrogram.npz"
        np.savez_compressed(spec_file,
                            frequencies=f_stft, times=t_stft,
                            power_db=power_db)
        print(f"  Saved {spec_file}")

        # Phase tracking
        phase_file = f"{args.output_prefix}_phase.npz"
        np.savez_compressed(phase_file,
                            times=phase_times, phases=phases,
                            doppler_hz=doppler_hz)
        print(f"  Saved {phase_file}")

        # Summary
        summary = {
            "input_file": args.input,
            "rate": rate,
            "duration_s": duration_s,
            "f_cw_measured": float(f_cw),
            "f_lo_offset": float(f_offset),
            "fft_size": nperseg,
            "overlap": args.overlap,
            "window": args.window,
            "freq_resolution_hz": freq_res,
            "time_step_s": hop / rate,
            "active_only": args.active_only,
            "num_active_sf": len(active_sf_times),
            "doppler_mean_hz": float(np.mean(doppler_hz)) if len(doppler_hz) > 0 else None,
            "doppler_std_hz": float(np.std(doppler_hz)) if len(doppler_hz) > 0 else None,
        }
        summary_file = f"{args.output_prefix}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved {summary_file}")

    # -- Plots ---------------------------------------------------------------
    if args.plot:
        print(f"\n--- Generating plots ---")
        make_plots(f_stft, t_stft_real, power_db, args.doppler_range,
                   cw_times, cw_powers_db,
                   phase_times, phases, doppler_hz,
                   active_sf_times, args.output_prefix, rate)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
