#!/usr/bin/env python3
"""
test_pedestrian_model.py — Verify walking pedestrian micro-Doppler butterfly.

Generates a continuous CW signal modulated by the multi-scatterer pedestrian
model, computes STFT spectrogram, and displays the characteristic
butterfly/swordfish micro-Doppler pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows
from walking_micro_doppler import PedestrianParams, walking_micro_doppler


def main():
    Fs = 100e3
    duration_s = 3.0
    f_cw = 10e3
    A_cw = 1.0

    ped = PedestrianParams(v_bulk=1.2, arm_f_md=80, arm_f_rot=1.8)

    N = round(Fs * duration_s)
    t = np.arange(N) / Fs

    # Generate CW + micro-Doppler
    doppler_sig, f_inst, info = walking_micro_doppler(t, ped)
    sig = A_cw * np.exp(1j * 2 * np.pi * f_cw * t) * doppler_sig

    # ===== FIGURE 1: Per-scatterer instantaneous Doppler =====
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t, info.f_torso, 'k-', linewidth=2, label='Torso')
    ax1.plot(t, info.f_arm_L, 'r-', linewidth=1, label='Left Arm')
    ax1.plot(t, info.f_arm_R, 'b-', linewidth=1, label='Right Arm')
    ax1.plot(t, info.f_leg_L, 'm--', linewidth=1, label='Left Leg')
    ax1.plot(t, info.f_leg_R, 'c--', linewidth=1, label='Right Leg')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Doppler (Hz)')
    ax1.set_title(f'Per-Scatterer Doppler  (v={ped.v_bulk:.1f} m/s, '
                  f'f_bulk={info.f_bulk:.1f} Hz)')
    ax1.legend(loc='center right')
    ax1.grid(True)

    # ===== FIGURE 2: STFT Spectrogram (butterfly pattern) =====
    sig_baseband = sig * np.exp(-1j * 2 * np.pi * f_cw * t)

    window_ms = 100
    overlap_pct = 90
    win_samps = round(window_ms / 1000 * Fs)
    noverlap = round(overlap_pct / 100 * win_samps)
    nfft = int(2 ** np.ceil(np.log2(4 * win_samps)))

    f_spec, t_spec, Sxx = spectrogram(
        sig_baseband, fs=Fs, window=windows.hann(win_samps),
        noverlap=noverlap, nfft=nfft, return_onesided=False, mode='complex')

    Sxx = np.fft.fftshift(Sxx, axes=0)
    f_spec = np.arange(-nfft // 2, nfft // 2) * Fs / nfft

    S_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)

    doppler_range = 300
    f_mask = np.abs(f_spec) <= doppler_range

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    vmax = np.max(S_dB[f_mask, :])
    ax2.pcolormesh(t_spec, f_spec[f_mask], S_dB[f_mask, :],
                   cmap='jet', vmin=vmax - 40, vmax=vmax, shading='auto')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Doppler (Hz)')
    ax2.set_title(f'Walking Pedestrian Micro-Doppler  (v={ped.v_bulk:.1f} m/s, '
                  f'arm={ped.arm_f_md:.0f} Hz @ {ped.arm_f_rot:.1f} Hz)')

    # ===== FIGURE 3: Wide Doppler range =====
    doppler_range2 = 500
    f_mask2 = np.abs(f_spec) <= doppler_range2

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    vmax2 = np.max(S_dB[f_mask2, :])
    ax3.pcolormesh(t_spec, f_spec[f_mask2], S_dB[f_mask2, :],
                   cmap='jet', vmin=vmax2 - 40, vmax=vmax2, shading='auto')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Doppler (Hz)')
    ax3.set_title('Wide Doppler View (±500 Hz)')

    print(f'Bulk Doppler: {info.f_bulk:.1f} Hz')
    print(f'Arm swing: ±{ped.arm_f_md:.0f} Hz at {ped.arm_f_rot:.1f} Hz rate')
    print(f'Leg swing: ±{ped.leg_f_md:.0f} Hz at {ped.leg_f_rot:.1f} Hz rate')

    plt.show()
    print('Done.')


if __name__ == '__main__':
    main()
