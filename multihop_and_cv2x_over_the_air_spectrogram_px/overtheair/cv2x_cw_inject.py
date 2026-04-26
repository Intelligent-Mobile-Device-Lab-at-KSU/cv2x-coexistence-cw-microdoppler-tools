#!/usr/bin/env python3
"""
C-V2X Virtual Sensing Subcarrier CW Injection

Reads a captured C-V2X IQ file (.cf32, 30.72 Msps), detects active subframes,
identifies occupied subcarriers via FFT, and injects a CW tone at (k+0.5)×15 kHz
— exactly between two occupied data subcarriers. The modified IQ can be replayed
so the commercial radio still decodes C-V2X data while a second RX channel
extracts micro-Doppler from the CW tone.

Example:
    python cv2x_cw_inject.py --input cv2x_iq.cf32 --output cv2x_iq_cw.cf32 --plot

Then replay:
    python cv2x_capture_replay.py --replay --gpsdo --headless \\
        --tx-freq 5.915e9 --tx-gain 15 \\
        --tx-args "serial=33767A5,master_clock_rate=184.32e6" \\
        --replay-file cv2x_iq_cw.cf32 --loop -r 30.72e6
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# 3GPP LTE OFDM constants (20 MHz channel, 30.72 Msps, normal CP)
# ---------------------------------------------------------------------------
FFT_SIZE = 2048
SUBCARRIER_SPACING_HZ = 15000
NUM_ACTIVE_SC = 1200
# Active subcarriers occupy FFT bins 424..1623 (centred on DC bin 1024)
ACTIVE_SC_START = (FFT_SIZE - NUM_ACTIVE_SC) // 2   # 424
ACTIVE_SC_END = ACTIVE_SC_START + NUM_ACTIVE_SC      # 1624
DC_BIN = FFT_SIZE // 2                               # 1024

SAMPLES_PER_SF = 30720          # 1 ms subframe at 30.72 Msps
SAMPLES_PER_SLOT = 15360
SYMBOLS_PER_SF = 14
CP_LONG = 160                   # first symbol of each slot
CP_SHORT = 144                  # symbols 1-6 of each slot

DEFAULT_SAMP_RATE = 30720000.0

# Pre-compute symbol offsets within a subframe (slot0 + slot1)
def _build_symbol_table():
    """Return list of (offset, cp_len) for 14 symbols in a subframe."""
    table = []
    for slot in range(2):
        slot_offset = slot * SAMPLES_PER_SLOT
        offset = slot_offset
        for sym in range(7):
            cp = CP_LONG if sym == 0 else CP_SHORT
            table.append((offset, cp))
            offset += cp + FFT_SIZE
    return table

SYMBOL_TABLE = _build_symbol_table()  # [(offset, cp_len), ...] × 14


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True,
                        help="Input .cf32 IQ file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output .cf32 IQ file with injected CW")
    parser.add_argument("--rate", "-r", type=float, default=DEFAULT_SAMP_RATE,
                        help="Sample rate in Hz (default: 30.72e6)")
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE,
                        help="FFT size (default: 2048)")
    parser.add_argument("--threshold", "-t", type=float, default=6.0,
                        help="Active subframe detection threshold in dB above "
                             "median power (default: 6)")
    parser.add_argument("--occupancy-threshold", type=float, default=10.0,
                        help="Subcarrier occupancy threshold in dB above noise "
                             "floor (default: 10)")
    parser.add_argument("--inject-power", type=float, default=-6.0,
                        help="CW injection power relative to data subcarrier "
                             "amplitude in dB (default: -6)")
    parser.add_argument("--subcarrier", default="auto",
                        help="Subcarrier FFT bin index or 'auto' for random "
                             "selection within occupied range (default: auto)")
    parser.add_argument("--per-subframe", action="store_true",
                        help="Re-randomize CW position per subframe (testing)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true",
                        help="Show diagnostic matplotlib plots")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-subframe detail")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# OFDM helpers
# ---------------------------------------------------------------------------
def extract_ofdm_symbols(subframe_iq, fft_size=FFT_SIZE):
    """
    Extract and FFT all 14 OFDM symbols from a 30720-sample subframe.
    Returns ndarray shape (14, fft_size) of frequency-domain samples.
    """
    fd = np.empty((SYMBOLS_PER_SF, fft_size), dtype=np.complex64)
    for sym_idx, (offset, cp_len) in enumerate(SYMBOL_TABLE):
        # The useful part starts after the CP
        start = offset + cp_len
        end = start + fft_size
        fd[sym_idx] = np.fft.fftshift(np.fft.fft(subframe_iq[start:end], fft_size))
    return fd


def detect_active_subframes(iq_data, num_subframes, threshold_db):
    """
    Return boolean array indicating which subframes are active (have C-V2X TX).
    Uses per-subframe mean power compared to median.
    """
    powers = np.empty(num_subframes)
    for n in range(num_subframes):
        sf = iq_data[n * SAMPLES_PER_SF:(n + 1) * SAMPLES_PER_SF]
        powers[n] = np.mean(np.abs(sf) ** 2)
    # Convert to dB
    powers_db = 10.0 * np.log10(powers + 1e-30)
    median_db = np.median(powers_db)
    active = powers_db > (median_db + threshold_db)
    return active, powers_db, median_db


def detect_occupied_prbs(iq_data, active_indices, inactive_indices,
                         occupancy_threshold_db, fft_size=FFT_SIZE,
                         n_probe=20, min_prbs=3):
    """
    Detect occupied PRBs using a differential approach: subtract the average
    spectrum of inactive (noise-only) subframes from active subframes to
    cancel DC leakage, ADC spurs, and other persistent artifacts.

    Returns:
        occupied_ranges: list of (start_sc, end_sc) in 0..1199 active SC space
        sc_power_db:     per-subcarrier power of active subframes (for plotting)
        diff_prb_db:     per-PRB differential power in dB (100 PRBs)
        data_sc_amp:     median amplitude of occupied data subcarriers
    """
    num_prbs = NUM_ACTIVE_SC // 12  # 100

    def _avg_spectrum(indices, n):
        """Average per-subcarrier power over n subframes."""
        spec = np.zeros(NUM_ACTIVE_SC, dtype=np.float64)
        count = min(n, len(indices))
        for idx in indices[:count]:
            sf = iq_data[idx * SAMPLES_PER_SF:(idx + 1) * SAMPLES_PER_SF]
            fd = extract_ofdm_symbols(sf, fft_size)
            spec += np.mean(np.abs(fd[:, ACTIVE_SC_START:ACTIVE_SC_END]) ** 2,
                            axis=0)
        return spec / max(count, 1)

    active_spec = _avg_spectrum(active_indices, n_probe)
    noise_spec = _avg_spectrum(inactive_indices, n_probe)
    sc_power_db = 10.0 * np.log10(active_spec + 1e-30)

    # Differential: active minus noise baseline (in linear domain)
    diff = np.maximum(active_spec - noise_spec, 1e-30)
    # PRB-level averaging (12 SC per PRB)
    prb_diff = diff.reshape(num_prbs, 12).mean(axis=1)
    diff_prb_db = 10.0 * np.log10(prb_diff + 1e-30)

    # Threshold: median + occupancy_threshold_db
    median_diff_db = np.median(diff_prb_db)
    occ_prb = diff_prb_db > (median_diff_db + occupancy_threshold_db)

    # Find contiguous PRB ranges, filter by min width
    raw_ranges = []
    in_range = False
    start = 0
    for i, o in enumerate(occ_prb):
        if o and not in_range:
            start = i
            in_range = True
        elif not o and in_range:
            raw_ranges.append((start, i - 1))
            in_range = False
    if in_range:
        raw_ranges.append((start, num_prbs - 1))
    # Keep only ranges ≥ min_prbs
    ranges = [(s, e) for s, e in raw_ranges if (e - s + 1) >= min_prbs]
    if raw_ranges and not ranges:
        ranges = [max(raw_ranges, key=lambda r: r[1] - r[0])]

    # Convert PRB ranges to subcarrier ranges
    sc_ranges = [(s * 12, e * 12 + 11) for s, e in ranges]

    # Compute median data subcarrier amplitude from the active spectrum
    # using only occupied PRBs
    occ_sc_indices = []
    for s, e in sc_ranges:
        occ_sc_indices.extend(range(s, e + 1))
    if occ_sc_indices:
        # Get per-subcarrier amplitudes from active subframes
        occ_amps = []
        for idx in active_indices[:n_probe]:
            sf = iq_data[idx * SAMPLES_PER_SF:(idx + 1) * SAMPLES_PER_SF]
            fd = extract_ofdm_symbols(sf, fft_size)
            ab = np.abs(fd[:, ACTIVE_SC_START:ACTIVE_SC_END])
            occ_amps.append(ab[:, occ_sc_indices])
        data_sc_amp = float(np.median(np.concatenate(occ_amps)))
    else:
        data_sc_amp = 0.0

    return sc_ranges, sc_power_db, diff_prb_db, data_sc_amp


# ---------------------------------------------------------------------------
# CW generation
# ---------------------------------------------------------------------------
def select_subcarrier(occupied_ranges, rng, subcarrier_arg):
    """
    Select a virtual sensing subcarrier position.

    If subcarrier_arg is 'auto', randomly pick an interior subcarrier from the
    largest occupied range.  Otherwise parse as an integer FFT bin index.

    Returns (k_active, k_bin, f_cw_hz):
        k_active: index in 0..1199 active SC space
        k_bin:    FFT bin index 0..2047
        f_cw_hz:  baseband CW frequency in Hz (at +0.5 SC offset)
    """
    if subcarrier_arg != "auto":
        k_bin = int(subcarrier_arg)
        k_active = k_bin - ACTIVE_SC_START
        f_cw_hz = (k_bin - DC_BIN + 0.5) * SUBCARRIER_SPACING_HZ
        return k_active, k_bin, f_cw_hz

    if not occupied_ranges:
        raise RuntimeError("No occupied subcarriers found — cannot auto-select.")

    # Pick from the largest contiguous range
    largest = max(occupied_ranges, key=lambda r: r[1] - r[0])
    # Need at least 2 subcarriers to place a tone between them
    if largest[1] - largest[0] < 1:
        raise RuntimeError(
            f"Largest occupied range is only 1 subcarrier wide "
            f"({largest[0]}–{largest[1]}). Need at least 2.")
    # Pick an interior subcarrier (not the last one, since tone goes between k and k+1)
    k_active = rng.integers(largest[0], largest[1])  # [start, end-1]
    k_bin = k_active + ACTIVE_SC_START
    # Place CW at +0.5 subcarrier offset (between k and k+1)
    f_cw_hz = (k_bin - DC_BIN + 0.5) * SUBCARRIER_SPACING_HZ
    return k_active, k_bin, f_cw_hz


def generate_cw_subframe(f_cw_hz, amplitude, subframe_idx, samp_rate):
    """
    Generate a CW tone for one subframe with phase continuity.
    """
    n_start = subframe_idx * SAMPLES_PER_SF
    t = np.arange(SAMPLES_PER_SF, dtype=np.float64) / samp_rate
    phase_offset = 2.0 * np.pi * f_cw_hz * n_start / samp_rate
    cw = amplitude * np.exp(1j * (2.0 * np.pi * f_cw_hz * t + phase_offset))
    return cw.astype(np.complex64)


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(powers_db, median_db, threshold_db, active_mask,
               sc_power_db, diff_prb_db, occupancy_threshold_db,
               occupied_ranges, k_active, f_cw_hz,
               orig_subframe, injected_subframe, samp_rate):
    """Generate diagnostic plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("C-V2X CW Injection Diagnostics", fontsize=14)

    # 1. Subframe power profile
    ax = axes[0, 0]
    colors = ['tab:red' if a else 'tab:blue' for a in active_mask]
    ax.bar(range(len(powers_db)), powers_db, color=colors, width=1.0, alpha=0.7)
    ax.axhline(median_db + threshold_db, color='green', ls='--',
               label=f'Threshold ({median_db + threshold_db:.1f} dB)')
    ax.axhline(median_db, color='gray', ls=':', label=f'Median ({median_db:.1f} dB)')
    ax.set_xlabel("Subframe index")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"Subframe Power ({np.sum(active_mask)} active / {len(active_mask)} total)")
    ax.legend(fontsize=8)

    # 2. PRB-level differential power (active - inactive)
    ax = axes[0, 1]
    num_prbs = len(diff_prb_db)
    prb_colors = []
    for p in range(num_prbs):
        sc_s, sc_e = p * 12, p * 12 + 11
        in_occ = any(r[0] <= sc_s and sc_e <= r[1] for r in occupied_ranges)
        prb_colors.append('tab:red' if in_occ else 'tab:blue')
    ax.bar(range(num_prbs), diff_prb_db, color=prb_colors, width=1.0, alpha=0.7)
    median_diff = np.median(diff_prb_db)
    ax.axhline(median_diff + occupancy_threshold_db, color='green', ls='--',
               label=f'Threshold ({median_diff + occupancy_threshold_db:.1f} dB)')
    ax.axvline(k_active // 12, color='orange', ls='-', linewidth=2,
               label=f'CW @ PRB {k_active // 12}')
    ax.set_xlabel("PRB index (0–99)")
    ax.set_ylabel("Differential Power (dB)")
    ax.set_title("PRB Occupancy (active − inactive)")
    ax.legend(fontsize=8)

    # 3. Before / after spectrum
    ax = axes[1, 0]
    if orig_subframe is not None and injected_subframe is not None:
        orig_fd = extract_ofdm_symbols(orig_subframe)
        inj_fd = extract_ofdm_symbols(injected_subframe)
        # Average power across symbols, show active bins
        orig_pwr = 10 * np.log10(
            np.mean(np.abs(orig_fd[:, ACTIVE_SC_START:ACTIVE_SC_END]) ** 2, axis=0) + 1e-30)
        inj_pwr = 10 * np.log10(
            np.mean(np.abs(inj_fd[:, ACTIVE_SC_START:ACTIVE_SC_END]) ** 2, axis=0) + 1e-30)
        sc_idx = np.arange(NUM_ACTIVE_SC)
        ax.plot(sc_idx, orig_pwr, linewidth=0.5, alpha=0.7, label='Original')
        ax.plot(sc_idx, inj_pwr, linewidth=0.5, alpha=0.7, label='With CW')
        ax.axvline(k_active, color='orange', ls='-', linewidth=2,
                   label=f'CW tone')
        ax.set_xlabel("Active subcarrier index")
        ax.set_ylabel("Power (dB)")
        ax.set_title("Before / After CW Injection")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)

    # 4. Time-domain amplitude around an active subframe boundary
    ax = axes[1, 1]
    if injected_subframe is not None:
        amp = np.abs(injected_subframe)
        t_ms = np.arange(len(amp)) / samp_rate * 1000.0
        ax.plot(t_ms, amp, linewidth=0.3, color='tab:blue')
        ax.set_xlabel("Time within subframe (ms)")
        ax.set_ylabel("|IQ|")
        ax.set_title("Time-Domain Amplitude (injected subframe)")
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plot_file = "cv2x_cw_inject_diag.png"
    plt.savefig(plot_file, dpi=150)
    print(f"[plot] Saved diagnostic plot to {plot_file}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # --- Load input ---
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    file_size = os.path.getsize(args.input)
    num_samples = file_size // np.dtype(np.complex64).itemsize
    num_subframes = num_samples // SAMPLES_PER_SF
    truncated = num_samples % SAMPLES_PER_SF
    duration_s = num_subframes * SAMPLES_PER_SF / args.rate

    print(f"Input:      {args.input}")
    print(f"File size:  {file_size / 1e6:.2f} MB  "
          f"({num_samples:,} samples, {num_subframes} subframes, "
          f"{duration_s:.3f} s)")
    if truncated:
        print(f"WARNING: {truncated} trailing samples don't fill a complete "
              f"subframe — they will be ignored.")

    input_data = np.memmap(args.input, dtype=np.complex64, mode='r',
                           shape=(num_subframes * SAMPLES_PER_SF,))

    # --- Step 1: Detect active subframes ---
    print(f"\n--- Step 1: Active subframe detection (threshold={args.threshold} dB) ---")
    active_mask, powers_db, median_db = detect_active_subframes(
        input_data, num_subframes, args.threshold)
    num_active = int(np.sum(active_mask))
    active_indices = np.where(active_mask)[0]
    print(f"Active subframes: {num_active} / {num_subframes} "
          f"({100.0 * num_active / num_subframes:.1f}%)")
    if num_active == 0:
        print("ERROR: No active subframes detected. Try lowering --threshold.")
        return 1

    # --- Step 2: Differential occupancy detection ---
    # Subtract noise-only spectrum (inactive subframes) from active to cancel
    # DC leakage, ADC spurs, and persistent artifacts.
    inactive_indices = np.where(~active_mask)[0]
    print(f"\n--- Step 2: Differential occupancy detection "
          f"(occ_threshold={args.occupancy_threshold} dB) ---")
    print(f"  Using {min(20, num_active)} active + "
          f"{min(20, len(inactive_indices))} inactive subframes for averaging")

    occupied_ranges, sc_power_db, diff_prb_db, data_amp = \
        detect_occupied_prbs(input_data, active_indices, inactive_indices,
                             args.occupancy_threshold, args.fft_size)

    if not occupied_ranges:
        print("ERROR: No occupied subcarriers detected. "
              "Try lowering --occupancy-threshold.")
        return 1

    for r in occupied_ranges:
        width = r[1] - r[0] + 1
        prbs = width // 12
        freq_lo = (r[0] + ACTIVE_SC_START - DC_BIN) * SUBCARRIER_SPACING_HZ
        freq_hi = (r[1] + ACTIVE_SC_START - DC_BIN) * SUBCARRIER_SPACING_HZ
        print(f"  Occupied: SC {r[0]}–{r[1]}  "
              f"({width} subcarriers, {prbs} PRBs, "
              f"{freq_lo/1e3:+.0f} to {freq_hi/1e3:+.0f} kHz)")

    # --- Step 3: Select virtual sensing subcarrier ---
    print(f"\n--- Step 3: Virtual sensing subcarrier selection ---")
    k_active, k_bin, f_cw_hz = select_subcarrier(
        occupied_ranges, rng, args.subcarrier)
    print(f"  Selected SC index: {k_active} (FFT bin {k_bin})")
    print(f"  CW frequency:     {f_cw_hz:.1f} Hz  ({f_cw_hz/1e3:.2f} kHz)")
    print(f"  Placed between data SC {k_active} and {k_active + 1}")

    # --- Amplitude scaling ---
    # data_amp is FFT-domain amplitude (scaled by FFT_SIZE due to unnormalized
    # DFT).  Convert to time-domain amplitude before applying inject_power.
    data_amp_td = data_amp / FFT_SIZE
    inject_amp = data_amp_td * 10.0 ** (args.inject_power / 20.0)
    print(f"  Median data SC amplitude (FFT): {data_amp:.6f}")
    print(f"  Median data SC amplitude (TD):  {data_amp_td:.6f}")
    print(f"  Inject amplitude:               {inject_amp:.8f} "
          f"({args.inject_power:+.1f} dB rel. to 1 subcarrier)")

    # --- Step 5: Copy file and inject CW ---
    print(f"\n--- Step 4: Injecting CW into output file ---")
    print(f"Output: {args.output}")
    shutil.copy2(args.input, args.output)
    output_data = np.memmap(args.output, dtype=np.complex64, mode='r+',
                            shape=(num_subframes * SAMPLES_PER_SF,))

    # Save one original subframe for plotting before modification
    first_active_idx = active_indices[0]
    orig_subframe = np.array(
        input_data[first_active_idx * SAMPLES_PER_SF:
                   (first_active_idx + 1) * SAMPLES_PER_SF])

    clip_count = 0
    for i, sf_idx in enumerate(active_indices):
        if args.per_subframe and args.subcarrier == "auto":
            k_active, k_bin, f_cw_hz = select_subcarrier(
                occupied_ranges, rng, "auto")
            if args.verbose:
                print(f"  SF {sf_idx}: CW @ SC {k_active} ({f_cw_hz:.0f} Hz)")

        cw = generate_cw_subframe(f_cw_hz, inject_amp, sf_idx, args.rate)
        start = sf_idx * SAMPLES_PER_SF
        end = start + SAMPLES_PER_SF
        output_data[start:end] += cw

        # Clip check
        peak = np.max(np.abs(output_data[start:end]))
        if peak > 1.0:
            clip_count += 1
            if args.verbose:
                print(f"  WARNING: SF {sf_idx} clips (peak={peak:.4f})")

        if args.verbose and not args.per_subframe:
            if i < 5 or i == num_active - 1:
                print(f"  Injected SF {sf_idx} (peak={peak:.4f})")
            elif i == 5:
                print(f"  ... ({num_active - 6} more) ...")

    output_data.flush()

    # Grab injected version of first active subframe for plotting
    injected_subframe = np.array(
        output_data[first_active_idx * SAMPLES_PER_SF:
                    (first_active_idx + 1) * SAMPLES_PER_SF])

    print(f"\nInjection complete.")
    print(f"  Subframes modified: {num_active}")
    if clip_count:
        print(f"  WARNING: {clip_count} subframe(s) clipped (|sample| > 1.0). "
              f"Consider lowering --inject-power.")

    # --- Step 6: Write sidecar ---
    input_sidecar = read_sidecar(args.input)
    output_sidecar = dict(input_sidecar) if input_sidecar else {}
    output_sidecar["cw_inject"] = {
        "f_cw_hz": f_cw_hz,
        "subcarrier_active_idx": int(k_active),
        "subcarrier_fft_bin": int(k_bin),
        "inject_power_db": args.inject_power,
        "inject_amplitude": float(inject_amp),
        "per_subframe": args.per_subframe,
        "active_subframes": [int(x) for x in active_indices],
        "num_active": num_active,
        "num_total": num_subframes,
    }
    write_sidecar(args.output, output_sidecar)

    # --- Plots ---
    if args.plot:
        make_plots(
            powers_db, median_db, args.threshold, active_mask,
            sc_power_db, diff_prb_db, args.occupancy_threshold,
            occupied_ranges, k_active, f_cw_hz,
            orig_subframe, injected_subframe, args.rate)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
