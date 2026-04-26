"""
compute_quality_metric.py — Compare hopped spectrogram quality against reference.

Computes quantitative similarity metrics between the de-hopped spectrogram
and the ground-truth reference spectrogram.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Spectrogram quality comparison results."""
    correlation: float = 0.0
    peak_sidelobe: float = 0.0
    dynamic_range: float = 0.0
    mse_dB: float = 0.0
    summary_str: str = ''


def compute_quality_metric(S_hopped: np.ndarray,
                           S_reference: np.ndarray,
                           f_axis: np.ndarray,
                           t_axis: np.ndarray) -> QualityMetrics:
    """Compare hopped spectrogram quality against reference.

    Parameters
    ----------
    S_hopped : ndarray, shape (n_freq, n_time)
        Linear magnitude spectrogram from hopped signal.
    S_reference : ndarray, shape (n_freq, n_time)
        Linear magnitude spectrogram from reference signal.
    f_axis : ndarray
        Frequency axis (Hz).
    t_axis : ndarray
        Time axis (s).

    Returns
    -------
    metrics : QualityMetrics
        Comparison metrics.
    """
    # Ensure matching sizes
    nf = min(S_hopped.shape[0], S_reference.shape[0])
    nt = min(S_hopped.shape[1], S_reference.shape[1])
    S_h = S_hopped[:nf, :nt]
    S_r = S_reference[:nf, :nt]

    m = QualityMetrics()

    # 2D Pearson correlation
    h_vec = S_h.ravel()
    r_vec = S_r.ravel()
    if np.std(h_vec) > 0 and np.std(r_vec) > 0:
        m.correlation = float(np.corrcoef(h_vec, r_vec)[0, 1])
    else:
        m.correlation = 0.0

    # Peak sidelobe level
    eps = np.finfo(float).eps
    S_h_dB = 20 * np.log10(S_h + eps)
    peak_val = np.max(S_h_dB)

    S_r_norm = S_r / (np.max(S_r) + eps)
    signal_mask = S_r_norm > 0.1
    noise_mask = ~signal_mask
    if np.any(noise_mask):
        m.peak_sidelobe = float(np.max(S_h_dB[noise_mask]) - peak_val)
    else:
        m.peak_sidelobe = 0.0

    # Dynamic range
    m.dynamic_range = float(peak_val - np.median(S_h_dB))

    # MSE in dB domain
    S_r_dB = 20 * np.log10(S_r + eps)
    m.mse_dB = float(np.mean((S_h_dB - S_r_dB) ** 2))

    # Summary string
    m.summary_str = (
        f'Correlation:     {m.correlation:.3f}\n'
        f'Peak Sidelobe:   {m.peak_sidelobe:.1f} dB\n'
        f'Dynamic Range:   {m.dynamic_range:.1f} dB\n'
        f'MSE (dB):        {m.mse_dB:.1f}'
    )

    return m
