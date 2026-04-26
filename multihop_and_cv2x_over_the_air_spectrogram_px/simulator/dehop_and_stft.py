"""
dehop_and_stft.py — De-hop a frequency-hopping CW signal and compute STFT spectrogram.

De-hops the signal by shifting each dwell's CW tone to a common output
frequency, then computes an STFT spectrogram for micro-Doppler extraction.
Uses global time for the de-hopping phasor to preserve phase continuity.
"""

import numpy as np
from scipy.signal import spectrogram, windows
from dataclasses import dataclass
from typing import Optional, List
from generate_hopped_cw import HopDwell


@dataclass
class STFTParams:
    """STFT processing parameters."""
    window_ms: float = 100.0
    window_type: str = 'hann'
    kaiser_beta: float = 6.0
    overlap_pct: float = 80.0
    zero_pad_factor: int = 4
    per_dwell_window: bool = False
    blank_transitions: bool = False
    taper_pct: float = 0.10
    doppler_range_hz: float = 500.0


def _make_window(wtype: str, n: int, kaiser_beta: float = 6.0) -> np.ndarray:
    """Create a window function."""
    if wtype == 'rectangular':
        return np.ones(n)
    elif wtype == 'hann':
        return windows.hann(n)
    elif wtype == 'blackman':
        return windows.blackman(n)
    elif wtype == 'hamming':
        return windows.hamming(n)
    elif wtype == 'kaiser':
        return windows.kaiser(n, kaiser_beta)
    else:
        return windows.hann(n)


def dehop_and_stft(sig: np.ndarray,
                   Fs: float,
                   hop_schedule: List[HopDwell],
                   stft_params: STFTParams
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """De-hop and compute STFT spectrogram.

    Parameters
    ----------
    sig : ndarray, shape (N,), complex
        Complex baseband signal.
    Fs : float
        Sample rate (Hz).
    hop_schedule : list of HopDwell
        Per-dwell timing and frequency info.
    stft_params : STFTParams
        STFT processing parameters.

    Returns
    -------
    S_dB : ndarray, shape (n_freq, n_time)
        Spectrogram in dB.
    f_axis : ndarray
        Frequency axis (Hz), centered on Doppler = 0.
    t_axis : ndarray
        Time axis (seconds).
    S_lin : ndarray
        Linear magnitude spectrogram.
    """
    sig = np.asarray(sig).ravel().copy()
    N = len(sig)
    t_global = np.arange(N) / Fs
    p = stft_params

    # === Step 1: De-hop — shift each dwell to a common output frequency ===
    f_out = hop_schedule[0].freq_hz
    sig_dehop = sig.copy()

    for dwell in hop_schedule:
        s1 = dwell.start_sample
        s2 = dwell.end_sample + 1  # Python slice end
        delta_f = dwell.freq_hz - f_out

        if abs(delta_f) > 0.1:
            sig_dehop[s1:s2] = sig[s1:s2] * np.exp(
                -1j * 2 * np.pi * delta_f * t_global[s1:s2])

    # === Step 2 (optional): Blank transition samples ===
    if p.blank_transitions:
        for dwell in hop_schedule:
            s1 = dwell.start_sample
            s2 = dwell.end_sample
            n_this = s2 - s1 + 1
            n_ramp = int(np.floor(p.taper_pct * n_this))
            if n_ramp >= 1:
                sig_dehop[s1:min(s1 + n_ramp, s2 + 1)] = 0
                sig_dehop[max(s2 - n_ramp + 1, s1):s2 + 1] = 0

    # === Step 3 (optional): Per-dwell windowing ===
    if p.per_dwell_window:
        for dwell in hop_schedule:
            s1 = dwell.start_sample
            s2 = dwell.end_sample + 1
            n_this = s2 - s1
            w = _make_window(p.window_type, n_this, p.kaiser_beta)
            sig_dehop[s1:s2] = sig_dehop[s1:s2] * w

    # === Step 4: CFO correction — shift de-hopped CW peak to 0 Hz ===
    sig_baseband = sig_dehop * np.exp(-1j * 2 * np.pi * f_out * t_global)

    # === Step 5: STFT spectrogram ===
    win_samps = max(round(p.window_ms / 1000 * Fs), 16)
    noverlap = round(p.overlap_pct / 100 * win_samps)
    nfft = int(2 ** np.ceil(np.log2(p.zero_pad_factor * win_samps)))

    w = _make_window(p.window_type, win_samps, p.kaiser_beta)

    f_spec, t_spec, Sxx = spectrogram(
        sig_baseband, fs=Fs, window=w, noverlap=noverlap, nfft=nfft,
        return_onesided=False, mode='complex')

    # fftshift for centered spectrum
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f_spec = np.fft.fftshift(f_spec)
    # Fix frequency axis: spectrogram returns [0, Fs) — shift to [-Fs/2, Fs/2)
    f_spec = np.arange(-nfft // 2, nfft // 2) * Fs / nfft

    # Crop to Doppler range
    doppler_mask = np.abs(f_spec) <= p.doppler_range_hz
    S_lin = np.abs(Sxx[doppler_mask, :])
    f_axis = f_spec[doppler_mask]
    t_axis = t_spec
    S_dB = 20 * np.log10(S_lin + np.finfo(float).eps)

    return S_dB, f_axis, t_axis, S_lin
