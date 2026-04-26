#!/usr/bin/env python3
"""
test_taper_envelope.py — Visualize taper shapes and spectral leakage reduction.

Generates all taper shapes for a fixed dwell, plots time-domain envelopes,
then compares the power spectrum of a CW hop transition with and without
tapering. Quantifies sidelobe suppression.
"""

import numpy as np
import matplotlib.pyplot as plt
from make_taper_envelope import make_taper_envelope


def compute_peak_sidelobe(S_dB, f_axis, signal_freqs, guard_hz=2000):
    """Measure peak sidelobe level away from main lobes."""
    mask = np.ones(len(S_dB), dtype=bool)
    for sf in signal_freqs:
        mask &= (np.abs(f_axis - sf) > guard_hz)
        mask &= (np.abs(f_axis + sf) > guard_hz)
    mask &= (np.abs(f_axis) > guard_hz)
    if np.any(mask):
        return float(np.max(S_dB[mask]))
    return -80.0


def main():
    # Parameters
    Fs = 100e3
    dwell_ms = 10
    N_dwell = round(Fs * dwell_ms / 1000)
    taper_pct = 0.10
    min_amp = 0.05

    f1, f2 = 10e3, 25e3

    shapes = ['none', 'linear', 'raised_cosine', 'hann', 'blackman']
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(shapes)))

    # ===== FIGURE 1: Taper envelope shapes =====
    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    for k, shape in enumerate(shapes):
        env = make_taper_envelope(N_dwell, shape, taper_pct, min_amp)
        t_ms = np.arange(N_dwell) / Fs * 1000
        ax1.plot(t_ms, env, color=colors[k], linewidth=1.5, label=shape)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Taper Envelopes  (dwell={dwell_ms}ms, taper={taper_pct*100:.0f}%, min={min_amp:.2f})')
    ax1.legend(loc='lower center', ncol=len(shapes))
    ax1.set_ylim([0, 1.1])
    ax1.grid(True)

    # ===== FIGURE 2: Spectral leakage comparison =====
    N_total = 2 * N_dwell
    nfft = int(2 ** np.ceil(np.log2(4 * N_total)))
    f_axis = np.arange(-nfft // 2, nfft // 2) * Fs / nfft

    fig2, axes = plt.subplots(len(shapes), 1, figsize=(10, 2.2 * len(shapes)),
                              sharex=True)

    for k, shape in enumerate(shapes):
        env1 = make_taper_envelope(N_dwell, shape, taper_pct, min_amp)
        env2 = make_taper_envelope(N_dwell, shape, taper_pct, min_amp)
        env_full = np.concatenate([env1, env2])

        # Build CW signal with hop at midpoint using phase accumulator
        sig = np.zeros(N_total, dtype=complex)
        phase = 0.0
        for n in range(N_total):
            f_inst = f1 if n < N_dwell else f2
            sig[n] = env_full[n] * np.exp(1j * phase)
            phase += 2 * np.pi * f_inst / Fs

        S = np.fft.fftshift(np.fft.fft(sig, nfft))
        S_dB = 20 * np.log10(np.abs(S) / (np.max(np.abs(S)) + 1e-30) + 1e-30)

        sl = compute_peak_sidelobe(S_dB, f_axis, [f1, f2])

        ax = axes[k]
        ax.plot(f_axis / 1e3, S_dB, color=colors[k], linewidth=0.8)
        ax.set_xlim([0, Fs / 2e3])
        ax.set_ylim([-80, 5])
        ax.set_ylabel('dB')
        ax.set_title(f'{shape}  (peak sidelobe: {sl:.1f} dB)')
        ax.grid(True)

    axes[-1].set_xlabel('Frequency (kHz)')
    fig2.suptitle(f'Spectral Leakage: 2-dwell hop ({f1/1e3:.0f}→{f2/1e3:.0f} kHz), '
                  f'dwell={dwell_ms}ms, taper={taper_pct*100:.0f}%')
    fig2.tight_layout()

    # ===== FIGURE 3: Leakage vs taper percentage =====
    taper_pcts = np.arange(0, 0.52, 0.02)
    fig3, ax3 = plt.subplots(figsize=(6, 4))

    for k, shape in enumerate(shapes):
        sidelobes = np.zeros(len(taper_pcts))
        for j, tp in enumerate(taper_pcts):
            env1 = make_taper_envelope(N_dwell, shape, tp, min_amp)
            env2 = make_taper_envelope(N_dwell, shape, tp, min_amp)
            env_full = np.concatenate([env1, env2])

            sig = np.zeros(N_total, dtype=complex)
            phase = 0.0
            for n in range(N_total):
                f_inst = f1 if n < N_dwell else f2
                sig[n] = env_full[n] * np.exp(1j * phase)
                phase += 2 * np.pi * f_inst / Fs

            S = np.fft.fftshift(np.fft.fft(sig, nfft))
            S_dB = 20 * np.log10(np.abs(S) / (np.max(np.abs(S)) + 1e-30) + 1e-30)
            sidelobes[j] = compute_peak_sidelobe(S_dB, f_axis, [f1, f2])

        ax3.plot(taper_pcts * 100, sidelobes, color=colors[k],
                 linewidth=1.5, label=shape)

    ax3.set_xlabel('Taper (% of dwell)')
    ax3.set_ylabel('Peak Sidelobe (dB)')
    ax3.set_title('Sidelobe Suppression vs Taper Duration')
    ax3.legend()
    ax3.grid(True)

    plt.show()
    print('Done.')


if __name__ == '__main__':
    main()
