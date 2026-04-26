#!/usr/bin/env python3
"""
test_dehop_pipeline.py — End-to-end: generate → de-hop → STFT, compare with reference.

Full pipeline sanity check. Generates hopped CW with pedestrian micro-Doppler,
de-hops, computes STFT, and displays side-by-side with the reference.
"""

import numpy as np
import matplotlib.pyplot as plt
from walking_micro_doppler import PedestrianParams
from generate_reference_cw import RefCWParams, generate_reference_cw
from generate_hopped_cw import HoppedCWParams, HopDwell, generate_hopped_cw
from dehop_and_stft import STFTParams, dehop_and_stft
from compute_quality_metric import compute_quality_metric


def main():
    Fs = 100e3
    duration_s = 3.0
    f_cw = 10e3

    ped = PedestrianParams(v_bulk=1.2, arm_f_md=80, arm_f_rot=1.8)

    stft_p = STFTParams(
        window_ms=100, window_type='hann', overlap_pct=85,
        zero_pad_factor=4, doppler_range_hz=300, taper_pct=0.10)

    # Generate reference
    ref_params = RefCWParams(Fs=Fs, duration_s=duration_s,
                             center_freq=f_cw, SNR_dB=30, pedestrian=ped)
    print('Generating reference signal...')
    sig_ref, _ = generate_reference_cw(ref_params)

    ref_hop = [HopDwell(start_sample=0, end_sample=len(sig_ref) - 1,
                        freq_hz=f_cw, dwell_idx=0)]
    S_ref_dB, f_ref, t_ref, S_ref_lin = dehop_and_stft(sig_ref, Fs, ref_hop, stft_p)

    # Test configurations
    configs = [
        dict(dwell_ms=1,  taper_shape='none',          transition_mode='step',  label='1ms, No taper, Step'),
        dict(dwell_ms=1,  taper_shape='raised_cosine', transition_mode='step',  label='1ms, RC taper, Step'),
        dict(dwell_ms=1,  taper_shape='raised_cosine', transition_mode='chirp', label='1ms, RC taper, Chirp'),
        dict(dwell_ms=5,  taper_shape='none',          transition_mode='step',  label='5ms, No taper, Step'),
        dict(dwell_ms=5,  taper_shape='raised_cosine', transition_mode='chirp', label='5ms, RC taper, Chirp'),
        dict(dwell_ms=10, taper_shape='raised_cosine', transition_mode='chirp', label='10ms, RC taper, Chirp'),
    ]

    ncols = 3
    nrows = int(np.ceil((len(configs) + 1) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.ravel()

    # Plot reference
    vmax_ref = np.max(S_ref_dB)
    axes[0].pcolormesh(t_ref, f_ref, S_ref_dB, cmap='jet',
                       vmin=vmax_ref - 40, vmax=vmax_ref, shading='auto')
    axes[0].set_title('Reference (no hop)')
    axes[0].set_ylabel('Doppler (Hz)')

    for c, cfg in enumerate(configs):
        print(f'Config {c+1}/{len(configs)}: {cfg["label"]}')

        hop_params = HoppedCWParams(
            Fs=Fs, duration_s=duration_s, dwell_ms=cfg['dwell_ms'],
            hop_freqs=np.array([10e3, 25e3, 15e3, 30e3]),
            hop_sparsity=1.0, taper_shape=cfg['taper_shape'],
            taper_pct=0.10, min_amplitude=0.05,
            transition_mode=cfg['transition_mode'],
            SNR_dB=30, pedestrian=ped)

        sig_hop, _, hop_sched, _ = generate_hopped_cw(hop_params)
        S_hop_dB, f_hop, t_hop, S_hop_lin = dehop_and_stft(
            sig_hop, Fs, hop_sched, stft_p)

        metrics = compute_quality_metric(S_hop_lin, S_ref_lin, f_hop, t_hop)

        ax = axes[c + 1]
        ax.pcolormesh(t_hop, f_hop, S_hop_dB, cmap='jet',
                      vmin=vmax_ref - 40, vmax=vmax_ref, shading='auto')
        ax.set_title(f'{cfg["label"]}\ncorr={metrics.correlation:.3f}  '
                     f'SL={metrics.peak_sidelobe:.0f}dB', fontsize=9)
        if (c + 1) % ncols == 0:
            ax.set_ylabel('Doppler (Hz)')

        print(f'  Corr: {metrics.correlation:.3f} | '
              f'SL: {metrics.peak_sidelobe:.1f} dB | '
              f'DR: {metrics.dynamic_range:.1f} dB')

    # Hide unused axes
    for i in range(len(configs) + 1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('De-hop Pipeline: Hopped vs Reference Spectrograms', fontsize=13)
    fig.tight_layout()
    plt.show()
    print('\nDone.')


if __name__ == '__main__':
    main()
