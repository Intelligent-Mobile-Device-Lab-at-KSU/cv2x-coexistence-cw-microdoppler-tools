"""
generate_hopped_cw.py — Frequency-hopping CW with amplitude tapering.

Produces a CW signal that hops between frequencies at 1ms-subframe-aligned
dwell boundaries. Supports amplitude tapering at transitions with two
transition modes: 'step' (hard frequency change while amplitude is low)
and 'chirp' (smooth frequency sweep during guard).

Phase continuity is maintained via a global phase accumulator (NCO model)
that never resets — matching the pattern in generate_cw_baseband.m.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from make_taper_envelope import make_taper_envelope
from walking_micro_doppler import PedestrianParams, walking_micro_doppler


@dataclass
class HopDwell:
    """One dwell in the hop schedule."""
    start_sample: int = 0
    end_sample: int = 0
    freq_hz: float = 0.0
    dwell_idx: int = 0


@dataclass
class HoppedCWParams:
    """Parameters for hopped CW generation."""
    Fs: float = 100e3
    duration_s: float = 2.0
    dwell_ms: float = 5.0
    hop_freqs: np.ndarray = field(default_factory=lambda: np.array([10e3, 25e3, 15e3, 30e3]))
    hop_sparsity: float = 1.0
    taper_shape: str = 'raised_cosine'
    taper_pct: float = 0.10
    min_amplitude: float = 0.05
    transition_mode: str = 'chirp'   # 'step' or 'chirp'
    A_cw: float = 1.0
    SNR_dB: float = 30.0
    pedestrian: Optional[PedestrianParams] = None
    gap_ms: float = 0.0            # Mean gap between dwells in ms (0 = back-to-back)
    gap_jitter_ms: float = 0.0     # Max random deviation from gap_ms (uniform ±jitter, clamped ≥0)


def generate_hopped_cw(params: HoppedCWParams
                       ) -> tuple[np.ndarray, np.ndarray, List[HopDwell], np.ndarray]:
    """Generate frequency-hopping CW signal with amplitude tapering.

    Parameters
    ----------
    params : HoppedCWParams
        Signal generation parameters.

    Returns
    -------
    sig : ndarray, shape (N,), complex
        Complex baseband signal.
    t : ndarray, shape (N,)
        Time vector in seconds.
    hop_schedule : list of HopDwell
        Per-dwell timing and frequency info.
    envelope : ndarray, shape (N,)
        Real amplitude envelope.
    """
    Fs = params.Fs
    N_total = round(Fs * params.duration_s)
    N_dwell = round(Fs * params.dwell_ms / 1000)
    hop_freqs = np.asarray(params.hop_freqs)
    num_freqs = len(hop_freqs)

    t = np.arange(N_total) / Fs

    # === Build hop schedule (subframe-aligned, with per-dwell random gaps) ===
    # Each dwell occupies N_dwell samples of active CW, followed by a gap
    # of silence.  Gap duration = gap_ms ± uniform(gap_jitter_ms), clamped ≥ 0,
    # rounded to integer ms for 1ms-subframe alignment.
    hop_schedule: List[HopDwell] = []
    freq_idx = 0
    sample_pos = 0
    max_dwells = int(np.ceil(N_total / max(N_dwell, 1)))  # upper bound

    for d in range(max_dwells):
        s_start = sample_pos
        if s_start >= N_total:
            break
        s_end = min(s_start + N_dwell, N_total) - 1

        if d > 0 and np.random.rand() <= params.hop_sparsity:
            freq_idx = (freq_idx + 1) % num_freqs

        hop_schedule.append(HopDwell(
            start_sample=s_start,
            end_sample=s_end,
            freq_hz=hop_freqs[freq_idx],
            dwell_idx=d,
        ))

        # Compute gap for this dwell (randomised if jitter > 0)
        if params.gap_jitter_ms > 0:
            jitter = np.random.uniform(-params.gap_jitter_ms,
                                        params.gap_jitter_ms)
            gap_this_ms = max(0, round(params.gap_ms + jitter))
        else:
            gap_this_ms = round(params.gap_ms)
        N_gap_this = round(Fs * gap_this_ms / 1000)
        sample_pos = s_end + 1 + N_gap_this

    # === Build amplitude envelope ===
    # Envelope is nonzero only during active dwells; gap regions stay at 0.
    envelope = np.zeros(N_total)
    for dwell in hop_schedule:
        s1 = dwell.start_sample
        s2 = dwell.end_sample
        n_this = s2 - s1 + 1
        env_d = make_taper_envelope(n_this, params.taper_shape,
                                    params.taper_pct, params.min_amplitude)
        envelope[s1:s2 + 1] = env_d

    # === Generate CW signal with phase accumulator ===
    sig = np.zeros(N_total, dtype=np.complex128)
    global_phase = 0.0

    for d_idx, dwell in enumerate(hop_schedule):
        s1 = dwell.start_sample
        s2 = dwell.end_sample
        f_this = dwell.freq_hz
        n_this = s2 - s1 + 1

        f_prev = hop_schedule[d_idx - 1].freq_hz if d_idx > 0 else f_this
        n_ramp = int(np.floor(params.taper_pct * n_this))

        if (params.transition_mode == 'chirp'
                and f_prev != f_this and n_ramp >= 2):
            # === CHIRP MODE ===
            # Ramp-up region: frequency sweeps from f_prev to f_this
            for n in range(n_ramp):
                alpha = n / (n_ramp - 1)
                f_inst = f_prev + (f_this - f_prev) * alpha
                sig[s1 + n] = params.A_cw * envelope[s1 + n] * np.exp(1j * global_phase)
                global_phase += 2 * np.pi * f_inst / Fs

            # Flat region + ramp-down: frequency is f_this (stable)
            for n in range(n_ramp, n_this):
                sig[s1 + n] = params.A_cw * envelope[s1 + n] * np.exp(1j * global_phase)
                global_phase += 2 * np.pi * f_this / Fs
        else:
            # === STEP MODE (or no frequency change) ===
            for n in range(n_this):
                sig[s1 + n] = params.A_cw * envelope[s1 + n] * np.exp(1j * global_phase)
                global_phase += 2 * np.pi * f_this / Fs

        # === Advance NCO phase through gap (transmitter OFF, signal = 0) ===
        # Phase accumulator keeps running at f_this so phase is continuous
        # when the next dwell begins.  Gap length varies per-dwell.
        if d_idx < len(hop_schedule) - 1:
            n_gap_actual = hop_schedule[d_idx + 1].start_sample - (s2 + 1)
        else:
            n_gap_actual = N_total - (s2 + 1)
        if n_gap_actual > 0:
            global_phase += 2 * np.pi * f_this / Fs * n_gap_actual

    # === Apply pedestrian micro-Doppler ===
    # The pedestrian model returns an additive sum of Doppler-shifted
    # reflections at frequencies near 0 Hz (Doppler offsets).  Multiply
    # by the hopped CW carrier so each scatterer's return appears at
    # carrier_freq + scatterer_doppler — preserving all individual traces
    # for the STFT to resolve as the butterfly pattern.
    if params.pedestrian is not None:
        doppler_sig, _, _ = walking_micro_doppler(t, params.pedestrian)
        sig = sig * doppler_sig

    # === Add AWGN ===
    noise_sigma = params.A_cw / np.sqrt(2 * 10 ** (params.SNR_dB / 10))
    sig = sig + noise_sigma * (np.random.randn(N_total) + 1j * np.random.randn(N_total))

    return sig, t, hop_schedule, envelope
