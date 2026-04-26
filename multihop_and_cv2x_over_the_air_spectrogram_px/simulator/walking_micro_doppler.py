"""
walking_micro_doppler.py — Multi-scatterer walking pedestrian micro-Doppler model.

Models 5 point scatterers (torso, 2 arms, 2 legs) with Boulic-like gait
phasing to produce the characteristic butterfly/swordfish micro-Doppler
pattern.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


C = 299_792_458.0  # speed of light (m/s)


@dataclass
class PedestrianParams:
    """Parameters for the walking pedestrian micro-Doppler model."""
    v_bulk: float = 1.2        # Walking speed toward receiver (m/s)
    arm_f_md: float = 80.0     # Max arm swing Doppler shift (Hz)
    arm_f_rot: float = 1.8     # Arm swing rate (Hz)
    leg_f_md: Optional[float] = None   # Max leg swing Doppler (auto: 1.5*arm)
    leg_f_rot: Optional[float] = None  # Leg swing rate (auto: arm_f_rot)
    f_carrier: float = 5.9e9   # Carrier frequency (Hz)
    torso_rcs: float = 0.6     # Torso RCS fraction
    arm_rcs: float = 0.1       # Per-arm RCS fraction
    leg_rcs: float = 0.1       # Per-leg RCS fraction

    def __post_init__(self):
        if self.leg_f_md is None:
            self.leg_f_md = 1.5 * self.arm_f_md
        if self.leg_f_rot is None:
            self.leg_f_rot = self.arm_f_rot


@dataclass
class ScattererInfo:
    """Per-scatterer debug output."""
    f_bulk: float = 0.0
    f_torso: np.ndarray = field(default_factory=lambda: np.array([]))
    f_arm_L: np.ndarray = field(default_factory=lambda: np.array([]))
    f_arm_R: np.ndarray = field(default_factory=lambda: np.array([]))
    f_leg_L: np.ndarray = field(default_factory=lambda: np.array([]))
    f_leg_R: np.ndarray = field(default_factory=lambda: np.array([]))


def walking_micro_doppler(t: np.ndarray,
                          params: PedestrianParams
                          ) -> tuple[np.ndarray, np.ndarray, ScattererInfo]:
    """Generate composite Doppler modulation from a walking pedestrian.

    Each scatterer (torso, arms, legs) produces an independent Doppler-
    shifted reflection of the transmitted CW tone.  The returned signal
    is the **additive sum** of these reflections — this is what a
    receiver would actually observe.  The STFT of this composite signal
    reveals the characteristic butterfly / swordfish micro-Doppler
    pattern because each scatterer's sinusoidal frequency trace is
    individually resolvable.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Time vector in seconds.
    params : PedestrianParams
        Pedestrian model parameters.

    Returns
    -------
    doppler_sig : ndarray, shape (N,), complex
        Additive composite Doppler signal (NOT unit-magnitude).
        Multiply your CW carrier by this, or — if your CW carrier
        is already at baseband — this IS the received baseband signal
        from the pedestrian target.
    f_inst : ndarray, shape (N,)
        RCS-weighted composite instantaneous Doppler frequency (Hz).
    info : ScattererInfo
        Per-scatterer frequencies for debugging.
    """
    t = np.asarray(t).ravel()
    p = params
    twopi = 2 * np.pi

    # Bulk Doppler from walking speed
    f_bulk = 2 * p.v_bulk * p.f_carrier / C

    # Per-scatterer instantaneous Doppler frequencies
    f_torso = np.full_like(t, f_bulk)

    # Arms: anti-phase sinusoidal swing
    f_arm_L = f_bulk + p.arm_f_md * np.sin(twopi * p.arm_f_rot * t)
    f_arm_R = f_bulk + p.arm_f_md * np.sin(twopi * p.arm_f_rot * t + np.pi)

    # Legs: anti-phase, in quadrature with arms (Boulic gait)
    f_leg_L = f_bulk + p.leg_f_md * np.sin(twopi * p.leg_f_rot * t + np.pi / 2)
    f_leg_R = f_bulk + p.leg_f_md * np.sin(twopi * p.leg_f_rot * t + 3 * np.pi / 2)

    # Per-scatterer phase (integral of frequency)
    phi_torso = twopi * f_bulk * t

    phi_arm_L = twopi * (f_bulk * t
                         - (p.arm_f_md / p.arm_f_rot) * np.cos(twopi * p.arm_f_rot * t))
    phi_arm_R = twopi * (f_bulk * t
                         - (p.arm_f_md / p.arm_f_rot) * np.cos(twopi * p.arm_f_rot * t + np.pi))

    phi_leg_L = twopi * (f_bulk * t
                         - (p.leg_f_md / p.leg_f_rot) * np.cos(twopi * p.leg_f_rot * t + np.pi / 2))
    phi_leg_R = twopi * (f_bulk * t
                         - (p.leg_f_md / p.leg_f_rot) * np.cos(twopi * p.leg_f_rot * t + 3 * np.pi / 2))

    # Composite signal: additive sum of Doppler-shifted reflections
    # Each scatterer contributes an independent CW return weighted by its RCS.
    # Do NOT normalize — the amplitude modulation from constructive /
    # destructive interference is what creates the butterfly pattern in
    # the STFT spectrogram.
    doppler_sig = (p.torso_rcs * np.exp(1j * phi_torso)
                   + p.arm_rcs * np.exp(1j * phi_arm_L)
                   + p.arm_rcs * np.exp(1j * phi_arm_R)
                   + p.leg_rcs * np.exp(1j * phi_leg_L)
                   + p.leg_rcs * np.exp(1j * phi_leg_R))

    # Composite instantaneous Doppler (RCS-weighted)
    f_inst = (p.torso_rcs * f_torso
              + p.arm_rcs * f_arm_L
              + p.arm_rcs * f_arm_R
              + p.leg_rcs * f_leg_L
              + p.leg_rcs * f_leg_R)

    info = ScattererInfo(
        f_bulk=f_bulk,
        f_torso=f_torso,
        f_arm_L=f_arm_L, f_arm_R=f_arm_R,
        f_leg_L=f_leg_L, f_leg_R=f_leg_R,
    )

    return doppler_sig, f_inst, info
