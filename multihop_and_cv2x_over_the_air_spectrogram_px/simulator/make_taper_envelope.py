"""
make_taper_envelope.py — Per-dwell amplitude envelope generator.

Produces a real-valued amplitude envelope that ramps up from min_amp
at the start, holds at 1.0 in the middle, and ramps back down to
min_amp at the end. The ramp region uses half of a standard window
function to produce a smooth taper.
"""

import numpy as np
from scipy.signal import windows


def make_taper_envelope(n_dwell: int,
                        taper_shape: str = 'raised_cosine',
                        taper_pct: float = 0.10,
                        min_amp: float = 0.0) -> np.ndarray:
    """Generate amplitude envelope for one dwell period.

    Parameters
    ----------
    n_dwell : int
        Total samples in the dwell period.
    taper_shape : str
        Ramp shape: 'none', 'raised_cosine', 'hann', 'blackman', 'linear'.
    taper_pct : float
        Fraction of dwell used for each ramp edge (0 to 0.5).
    min_amp : float
        Minimum amplitude at the guard edges (0 to 0.5).

    Returns
    -------
    env : ndarray, shape (n_dwell,)
        Real envelope, values in [min_amp, 1.0].
    """
    env = np.ones(n_dwell)

    if taper_shape == 'none' or taper_pct <= 0:
        return env

    n_ramp = int(np.floor(taper_pct * n_dwell))
    if n_ramp < 2:
        return env

    # Generate the rising half-window shape (0 → 1)
    shape = taper_shape.lower()
    if shape == 'raised_cosine':
        n = np.arange(n_ramp) / (n_ramp - 1)
        ramp = 0.5 * (1 - np.cos(np.pi * n))
    elif shape == 'hann':
        w = windows.hann(2 * n_ramp, sym=False)
        ramp = w[:n_ramp]
    elif shape == 'blackman':
        w = windows.blackman(2 * n_ramp, sym=False)
        ramp = w[:n_ramp]
    elif shape == 'linear':
        ramp = np.linspace(0, 1, n_ramp)
    else:
        raise ValueError(f'Unknown taper_shape: {taper_shape}')

    # Scale ramp from min_amp → 1.0
    ramp = min_amp + (1 - min_amp) * ramp

    # Apply ramp-up at start, ramp-down at end
    env[:n_ramp] = ramp
    env[-n_ramp:] = ramp[::-1]

    return env
