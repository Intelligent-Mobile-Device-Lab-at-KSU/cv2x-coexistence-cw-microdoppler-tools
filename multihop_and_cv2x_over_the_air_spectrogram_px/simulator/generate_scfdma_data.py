"""
generate_scfdma_data.py — SC-FDMA coexistence interference for CW sensing.

Models the spectral leakage from 3GPP Rel 14 C-V2X sidelink SC-FDMA data
transmissions into the CW sensing bin at 7.5 kHz offset.  Used to study
coexistence between data and sensing in the same subframe.

Supports three TX/RX scenarios:
  1. Co-located TX, CW inside data allocation  — 100% overlap when active
  2. Co-located TX, CW outside data allocation — reduced leakage at distance
  3. Other vehicle (external)                  — random allocation overlap

PHY basis: 20 MHz C-V2X, 20 subchannels x 5 PRBs x 12 SC = 1200 data SC,
1199 sensing bins at 7.5 kHz midpoints.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


TOTAL_SUBCH = 20       # 20 MHz C-V2X: 20 subchannels of 5 PRBs each
BINS_PER_SUBCH = 1199 / TOTAL_SUBCH  # ~60 sensing bins per subchannel


@dataclass
class SCFDMAParams:
    """Parameters for SC-FDMA coexistence interference generation."""
    enabled: bool = False
    data_activity: float = 0.5           # fraction of subframes with SC-FDMA [0-1]
    num_subch_max: int = 5               # max subchannels per allocation [1-20]
    data_power_dB: float = 10.0          # SC-FDMA data power relative to CW (dB)
    spectral_isolation_dB: float = 23.0  # DFT-s-OFDM sidelobe at 7.5 kHz (dB)
    colocated_tx: bool = True            # True = same vehicle, False = other vehicle
    cw_inside_alloc: bool = True         # True = CW bin within own data allocation
    adjacent_subch_extra_dB: float = 12.0  # extra isolation per subchannel of distance


@dataclass
class SCFDMAResult:
    """Output from SC-FDMA interference generation."""
    interference: np.ndarray   # complex (N,) — add to CW signal before de-hopping
    data_mask: np.ndarray      # bool (num_subframes,) — SC-FDMA active
    overlap_mask: np.ndarray   # bool (num_subframes,) — allocation includes CW bin


def generate_scfdma_interference(params, N_total, Fs, hop_schedule, A_cw=1.0):
    """Generate SC-FDMA interference for coexistence with CW sensing.

    Three scenarios based on colocated_tx and cw_inside_alloc flags:

    Co-located TX, CW inside allocation (colocated=True, inside=True):
      Every active subframe overlaps the CW bin.  The data allocation
      always contains the sensing subchannel.  Allocation size does NOT
      affect overlap probability — it's always 100%.

    Co-located TX, CW outside allocation (colocated=True, inside=False):
      CW bin is placed on a subchannel NOT in the data allocation.
      Leakage is reduced by spectral distance (adjacent_subch_extra_dB
      per subchannel of separation).  Farther away = less interference.

    Other vehicle (colocated=False):
      Random SC-FDMA allocation from a different vehicle.  Overlap
      depends on whether the CW bin's subchannel falls inside the
      random allocation.  Original model.

    Parameters
    ----------
    params : SCFDMAParams
    N_total : int — total signal samples
    Fs : float — sample rate (Hz)
    hop_schedule : list of HopDwell — CW hopping schedule
    A_cw : float — CW amplitude

    Returns
    -------
    SCFDMAResult with interference signal, data_mask, overlap_mask.
    """
    samples_per_sf = round(Fs * 1e-3)  # 1ms subframe
    num_subframes = int(np.ceil(N_total / samples_per_sf))

    interference = np.zeros(N_total, dtype=np.complex128)
    data_mask = np.zeros(num_subframes, dtype=bool)
    overlap_mask = np.zeros(num_subframes, dtype=bool)

    if not params.enabled:
        return SCFDMAResult(interference, data_mask, overlap_mask)

    # Base leakage power at CW bin when allocation overlaps
    P_data = A_cw ** 2 * 10 ** (params.data_power_dB / 10)
    P_leak_base = P_data * 10 ** (-params.spectral_isolation_dB / 10)

    # Build a quick lookup: for each sample, which dwell is it in?
    dwell_freq_at_sample = np.full(N_total, np.nan)
    for dwell in hop_schedule:
        s1 = dwell.start_sample
        s2 = min(dwell.end_sample, N_total - 1)
        dwell_freq_at_sample[s1:s2 + 1] = dwell.freq_hz

    for sf in range(num_subframes):
        if np.random.rand() > params.data_activity:
            continue

        data_mask[sf] = True
        sf_start = sf * samples_per_sf
        sf_end = min((sf + 1) * samples_per_sf, N_total)
        n_samps = sf_end - sf_start

        # CW frequency at start of this subframe
        cw_freq = dwell_freq_at_sample[sf_start]
        if np.isnan(cw_freq):
            continue

        # CW bin -> subchannel
        bin_idx = round(cw_freq / 7500)
        cw_subch = min(int(bin_idx * TOTAL_SUBCH / 1199), TOTAL_SUBCH - 1)

        # Random SC-FDMA allocation (used for all scenarios)
        L = np.random.randint(1, params.num_subch_max + 1)
        start_subch = np.random.randint(0, max(TOTAL_SUBCH - L + 1, 1))
        alloc = set(range(start_subch, start_subch + L))

        if params.colocated_tx:
            if params.cw_inside_alloc:
                # CW is always inside the data allocation — 100% overlap
                overlap_mask[sf] = True
                sigma = np.sqrt(P_leak_base / 2)
            else:
                # CW is outside the data allocation — leakage at distance
                overlap_mask[sf] = True  # still interferes, just weaker

                # Distance in subchannels from nearest edge of allocation
                alloc_min = start_subch
                alloc_max = start_subch + L - 1
                if cw_subch < alloc_min:
                    dist = alloc_min - cw_subch
                elif cw_subch > alloc_max:
                    dist = cw_subch - alloc_max
                else:
                    # Shouldn't happen (CW is outside), but guard anyway
                    dist = 1

                # Extra isolation from spectral distance
                extra_iso_dB = dist * params.adjacent_subch_extra_dB
                P_leak_dist = P_leak_base * 10 ** (-extra_iso_dB / 10)
                sigma = np.sqrt(P_leak_dist / 2)
        else:
            # Other vehicle — random overlap
            if cw_subch not in alloc:
                continue
            overlap_mask[sf] = True
            sigma = np.sqrt(P_leak_base / 2)

        # Generate interference at CW frequency so it de-hops correctly
        t_sf = np.arange(sf_start, sf_end) / Fs
        noise_bb = sigma * (
            np.random.randn(n_samps) + 1j * np.random.randn(n_samps))
        interference[sf_start:sf_end] = noise_bb * np.exp(
            1j * 2 * np.pi * cw_freq * t_sf)

    return SCFDMAResult(interference, data_mask, overlap_mask)
