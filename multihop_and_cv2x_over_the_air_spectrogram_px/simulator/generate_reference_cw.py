"""
generate_reference_cw.py — Continuous (non-hopping) CW reference signal.

Produces a pure CW tone at a fixed frequency with the same pedestrian
micro-Doppler model as the hopped signal. Serves as ground truth for
spectrogram quality comparison.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from walking_micro_doppler import PedestrianParams, walking_micro_doppler


@dataclass
class RefCWParams:
    """Parameters for reference CW generation."""
    Fs: float = 100e3
    duration_s: float = 2.0
    center_freq: float = 10e3
    A_cw: float = 1.0
    SNR_dB: float = 30.0
    pedestrian: Optional[PedestrianParams] = None


def generate_reference_cw(params: RefCWParams) -> tuple[np.ndarray, np.ndarray]:
    """Generate continuous CW reference signal with optional micro-Doppler.

    Parameters
    ----------
    params : RefCWParams
        Signal generation parameters.

    Returns
    -------
    sig : ndarray, shape (N,), complex
        Complex baseband signal.
    t : ndarray, shape (N,)
        Time vector in seconds.
    """
    N = round(params.Fs * params.duration_s)
    t = np.arange(N) / params.Fs

    # CW tone at center_freq as the carrier.
    # The pedestrian model produces additive Doppler-shifted reflections
    # whose instantaneous frequencies are f_bulk ± f_md*sin(...).
    # At baseband (after mixing down center_freq) these appear as small
    # Doppler offsets around 0 Hz.  We place the carrier so the composite
    # received signal sits at center_freq + Doppler.
    if params.pedestrian is not None:
        # Doppler signal contains the sum of scatterer returns at Doppler
        # frequencies relative to 0 Hz.  Shift them up to center_freq.
        doppler_sig, _, _ = walking_micro_doppler(t, params.pedestrian)
        sig = params.A_cw * doppler_sig * np.exp(1j * 2 * np.pi * params.center_freq * t)
    else:
        sig = params.A_cw * np.exp(1j * 2 * np.pi * params.center_freq * t)

    # Add AWGN
    noise_sigma = params.A_cw / np.sqrt(2 * 10 ** (params.SNR_dB / 10))
    sig = sig + noise_sigma * (np.random.randn(N) + 1j * np.random.randn(N))

    return sig, t
