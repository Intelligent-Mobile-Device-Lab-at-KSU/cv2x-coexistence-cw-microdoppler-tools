#!/usr/bin/env python3
"""
test_phase_continuity.py — Verify NCO phase coherence across hop boundaries.

Generates a hopped CW signal (no noise, no Doppler), extracts instantaneous
phase, and checks that it is monotonically increasing with no jumps at hop
boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_hopped_cw import HoppedCWParams, generate_hopped_cw


def main():
    Fs = 100e3
    dwell_ms = 5
    duration_s = 0.05  # 50ms — short for clear visualization
    hop_freqs = np.array([10e3, 25e3, 15e3])

    modes = ['step', 'chirp']

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for m, mode in enumerate(modes):
        params = HoppedCWParams(
            Fs=Fs, duration_s=duration_s, dwell_ms=dwell_ms,
            hop_freqs=hop_freqs, hop_sparsity=1.0,
            taper_shape='raised_cosine', taper_pct=0.15,
            min_amplitude=0.05, transition_mode=mode,
            A_cw=1.0, SNR_dB=100)  # Very high SNR

        sig, t, hop_sched, envelope = generate_hopped_cw(params)

        # Extract instantaneous phase
        phase_inst = np.unwrap(np.angle(sig))

        # Instantaneous frequency from phase derivative
        f_inst = np.diff(phase_inst) * Fs / (2 * np.pi)

        # Hop boundary times
        hop_times = [dwell.start_sample / Fs for dwell in hop_sched]

        # Check phase continuity
        max_jump = 0.0
        for d in range(1, len(hop_sched)):
            idx = hop_sched[d].start_sample
            if 0 < idx < len(phase_inst):
                jump = abs(phase_inst[idx] - phase_inst[idx - 1]) - \
                       abs(2 * np.pi * hop_sched[d].freq_hz / Fs)
                max_jump = max(max_jump, abs(jump))

        # Plot: Unwrapped phase
        ax = axes[m, 0]
        ax.plot(t * 1000, phase_inst / (2 * np.pi), 'b-', linewidth=0.5)
        for ht in hop_times[1:]:
            ax.axvline(ht * 1000, color='r', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Phase (cycles)')
        ax.set_title(f'{mode.upper()}: Unwrapped Phase')
        ax.grid(True)

        # Plot: Instantaneous frequency
        ax = axes[m, 1]
        ax.plot(t[:-1] * 1000, f_inst / 1e3, 'b-', linewidth=0.5)
        for ht in hop_times[1:]:
            ax.axvline(ht * 1000, color='r', linestyle='--', linewidth=0.5)
        for dwell in hop_sched:
            t_mid = (dwell.start_sample + dwell.end_sample) / 2 / Fs * 1000
            ax.text(t_mid, dwell.freq_hz / 1e3,
                    f'{dwell.freq_hz/1e3:.0f}k',
                    ha='center', color='r', fontweight='bold', fontsize=8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title(f'{mode.upper()}: Instantaneous Freq')
        ax.grid(True)

        # Plot: Amplitude envelope
        ax = axes[m, 2]
        ax.plot(t * 1000, envelope, 'k-', linewidth=1, label='Designed')
        ax.plot(t * 1000, np.abs(sig), 'b-', linewidth=0.5, alpha=0.3,
                label='Actual |sig|')
        for ht in hop_times[1:]:
            ax.axvline(ht * 1000, color='r', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{mode.upper()}: Envelope')
        ax.legend(loc='lower center', fontsize=8)
        ax.set_ylim([0, 1.2])
        ax.grid(True)

        print(f'{mode.upper()} mode: max phase jump residual = '
              f'{max_jump:.6f} rad (should be ~0)')

    fig.suptitle('Phase Continuity Across Hop Boundaries', fontsize=13)
    fig.tight_layout()
    plt.show()
    print('\nRed dashed lines = hop boundaries.')
    print('Done.')


if __name__ == '__main__':
    main()
