#!/usr/bin/env python3
"""
cwhop_viewer.py -- Interactive Micro-Doppler Spectrogram Viewer
================================================================
Interactive STFT viewer for de-hopped CW hardware captures produced
by cwhop_tx.py / cwhop_rx.py.

First run: de-hops and decimates the raw .cf32 to 10 kHz, caches
           the result to <input>_10k.npz for fast subsequent loads.
Later runs: loads the .npz and jumps straight to the interactive UI.

Sliders (live update on release):
  Window ms | Overlap % | Zero-pad | Doppler range | Dynamic range | DC notch

Toggles:
  Stitch -- per-dwell phase normalization to fix hop-transition phase jumps
  Window type -- hann / blackman / hamming / rectangular

Usage:
    # First run -- processing ~2 min for a 60 s capture:
    python cwhop_viewer.py --input my_capture_rx.cf32 \\
        --sidecar my_capture_schedule.json \\
        --config new_update/python_version/cwhopping_config.json

    # Later runs -- cache loads in < 1 s:
    python cwhop_viewer.py --input my_capture_rx.cf32

    # Force reprocess even if cache exists (also rebuilds dwell ranges):
    python cwhop_viewer.py --input my_capture_rx.cf32 --reprocess
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import spectrogram as _sp_spectrogram
from scipy.signal import windows as _sig_windows


# ============================================================================
# Constants
# ============================================================================
FS_HZ       = 30_720_000
DECIM_TOTAL = 3072
FS_OUT      = FS_HZ / DECIM_TOTAL   # 10000.0 Hz


# ============================================================================
# Path helpers
# ============================================================================
def _cache_path(input_path: str) -> str:
    # Use splitext only — preserves _rx suffix so TX (my_capture.cf32)
    # and RX (my_capture_rx.cf32) get distinct caches and never collide.
    return os.path.splitext(input_path)[0] + '_10k.npz'


def _find_sidecar(input_path: str) -> str:
    base = input_path
    for suf in ('_rx.cf32', '.cf32'):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    for candidate in (base + '_schedule.json',
                      base.replace('_rx', '') + '_schedule.json'):
        if os.path.isfile(candidate):
            return candidate
    return base + '_schedule.json'


def _find_config(input_path: str) -> str:
    d = os.path.dirname(os.path.abspath(input_path))
    p = os.path.join(d, 'cwhopping_config.json')
    if os.path.isfile(p):
        return p
    p2 = os.path.join(d, 'new_update', 'python_version', 'cwhopping_config.json')
    if os.path.isfile(p2):
        return p2
    return p


# ============================================================================
# Import cwhop_rx processing functions
# ============================================================================
def _load_rx_module():
    """Import cwhop_rx.py from the same directory as this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    rx_path = os.path.join(here, 'cwhop_rx.py')
    if not os.path.isfile(rx_path):
        sys.exit(f"ERROR: cwhop_rx.py not found at {rx_path}")
    spec = importlib.util.spec_from_file_location('cwhop_rx', rx_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Cache: build or load
# ============================================================================
def build_cache(input_path: str, sidecar_path: str, cache_path: str):
    """Run de-hop + decimation via cwhop_rx and save 10 kHz signal."""
    rx = _load_rx_module()

    print(f"\n[cache] Processing {os.path.basename(input_path)}")
    print(f"        Sidecar: {sidecar_path}")
    print(f"        Cache will be saved to: {cache_path}")
    print(f"\n  This runs de-hop + 3072x decimation on the raw .cf32.")
    print(f"  For a 60 s capture this takes ~2 minutes.")
    print(f"  Subsequent viewer launches load the cache in < 1 s.\n")

    sidecar = rx.load_sidecar(sidecar_path)

    file_bytes = os.path.getsize(input_path)
    N_file = file_bytes // 8
    Fs = float(sidecar.get('Fs_hz', FS_HZ))
    hop_schedule = sidecar['hop_schedule']
    N_process = min(N_file, sidecar.get('N_total', N_file))

    raw_mmap = np.memmap(input_path, dtype=np.complex64, mode='r')

    DECIM_STAGES = rx.DECIM_STAGES
    total_decim = 1
    for q in DECIM_STAGES:
        total_decim *= q
    Fs_out = Fs / total_decim

    BLOCK = rx.BLOCK_SIZE
    n_blocks = int(np.ceil(N_process / BLOCK))
    N_out = int(np.ceil(N_process / total_decim))
    signal_10k = np.zeros(N_out, dtype=np.complex64)
    out_ptr = 0

    print(f"  {N_process:,} samples  |  {n_blocks} block(s)  |  "
          f"Fs_out = {Fs_out:.0f} Hz")
    t0 = time.time()

    for blk in range(n_blocks):
        blk_start = blk * BLOCK
        blk_end   = min(blk_start + BLOCK, N_process)
        n_blk     = blk_end - blk_start

        elapsed = time.time() - t0
        frac    = (blk + 1) / n_blocks
        eta     = elapsed / frac * (1.0 - frac) if frac > 0 else 0.0
        print(f"  Block {blk+1}/{n_blocks}  "
              f"[{blk_start/1e6:.1f}M-{blk_end/1e6:.1f}M]  "
              f"ETA {eta:.0f}s", end='\r', flush=True)

        raw_block = np.array(raw_mmap[blk_start:blk_end], dtype=np.complex64)
        dh = rx.dehop_block(raw_block, blk_start, hop_schedule, Fs)

        n_dec_expected = n_blk // total_decim
        dec = rx._decimate_block_with_margin(
            block_dehop=dh,
            block_start=blk_start,
            n_decimated_per_block=n_dec_expected,
            verbose=False,
        )
        n_write = min(len(dec), N_out - out_ptr)
        signal_10k[out_ptr:out_ptr + n_write] = dec[:n_write]
        out_ptr += n_write

    signal_10k = signal_10k[:out_ptr]
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f} s  "
          f"({out_ptr:,} samples @ {Fs_out:.0f} Hz = {out_ptr/Fs_out:.3f} s)")

    # Dwell boundary sample indices at 10 kHz (for phase stitching)
    dwell_starts_10k = np.array(
        [int(d['start_sample']) // total_decim for d in hop_schedule],
        dtype=np.int64)
    dwell_ends_10k = np.array(
        [int(d['end_sample']) // total_decim + 1 for d in hop_schedule],
        dtype=np.int64)
    dwell_ends_10k = np.minimum(dwell_ends_10k, out_ptr)

    np.savez(cache_path,
             signal=signal_10k,
             Fs_out=Fs_out,
             duration_s=out_ptr / Fs_out,
             hop_freqs=np.array([d['freq_hz'] for d in hop_schedule]),
             dwell_starts_10k=dwell_starts_10k,
             dwell_ends_10k=dwell_ends_10k)
    print(f"  Cache saved: {cache_path}")
    return signal_10k, Fs_out, dwell_starts_10k, dwell_ends_10k


def load_cache(cache_path: str):
    data = np.load(cache_path, allow_pickle=False)
    signal_10k = data['signal']
    Fs_out     = float(data['Fs_out'])
    duration_s = float(data['duration_s'])
    # dwell ranges may be absent in caches built before this version
    if 'dwell_starts_10k' in data and 'dwell_ends_10k' in data:
        dwell_starts_10k = data['dwell_starts_10k']
        dwell_ends_10k   = data['dwell_ends_10k']
        n_d = len(dwell_starts_10k)
    else:
        dwell_starts_10k = None
        dwell_ends_10k   = None
        n_d = 0
    print(f"[cache] Loaded {cache_path}")
    print(f"        {len(signal_10k):,} samples  "
          f"@ {Fs_out:.0f} Hz  =  {duration_s:.3f} s")
    if n_d:
        print(f"        {n_d} dwell ranges available for phase stitching")
    else:
        print(f"        No dwell ranges -- run with --reprocess to enable stitch")
    return signal_10k, Fs_out, dwell_starts_10k, dwell_ends_10k


# ============================================================================
# STFT
# ============================================================================
def _make_window(wtype: str, n: int) -> np.ndarray:
    wtype = wtype.lower()
    if wtype == 'hann':
        return _sig_windows.hann(n)
    elif wtype == 'blackman':
        return _sig_windows.blackman(n)
    elif wtype == 'hamming':
        return _sig_windows.hamming(n)
    elif wtype == 'rectangular':
        return np.ones(n)
    else:
        return _sig_windows.hann(n)


def compute_stft(signal_10k: np.ndarray, Fs_out: float,
                 window_ms: float, overlap_pct: float,
                 zero_pad_factor: int, doppler_range_hz: float,
                 window_type: str = 'hann'):
    """Compute STFT spectrogram on the 10 kHz de-hopped signal.

    Returns (S_dB, S_lin, f_axis, t_axis).
    """
    win_samps = max(int(round(window_ms / 1000 * Fs_out)), 16)
    noverlap  = int(round(overlap_pct / 100 * win_samps))
    noverlap  = min(noverlap, win_samps - 1)
    nfft      = int(2 ** np.ceil(np.log2(max(zero_pad_factor, 1) * win_samps)))

    w = _make_window(window_type, win_samps)

    f_sp, t_sp, Sxx = _sp_spectrogram(
        signal_10k, fs=Fs_out, window=w,
        noverlap=noverlap, nfft=nfft,
        return_onesided=False, mode='complex')

    Sxx = np.fft.fftshift(Sxx, axes=0)
    f_axis = np.arange(-nfft // 2, nfft // 2) * Fs_out / nfft
    t_axis = t_sp

    mask   = np.abs(f_axis) <= doppler_range_hz
    S_lin  = np.abs(Sxx[mask, :])
    f_axis = f_axis[mask]

    S_dB = 20 * np.log10(S_lin + np.finfo(float).eps)
    return S_dB, S_lin, f_axis, t_axis


# ============================================================================
# Phase stitching
# ============================================================================
def stitch_phase(sig_10k: np.ndarray,
                 dwell_starts: np.ndarray,
                 dwell_ends: np.ndarray) -> np.ndarray:
    """Per-dwell phase normalization to remove hop-transition phase jumps.

    For each dwell, estimates the DC carrier phase via mean phasor and
    rotates that dwell to align its carrier to zero phase.  This removes
    the inter-dwell phase discontinuities that cause bright vertical
    artifact lines in the STFT at every hop transition.

    Works best when the CW tone is close to DC (Doppler << 1/T_dwell).
    For 64 ms dwells this is ideal for Doppler < ~15 Hz; for larger
    Doppler the mean phasor magnitude is smaller but the correction still
    reduces most of the visible choppiness.
    """
    if dwell_starts is None or len(dwell_starts) == 0:
        return sig_10k
    out = sig_10k.astype(np.complex64, copy=True)
    N = len(out)
    for s, e in zip(dwell_starts.astype(int), dwell_ends.astype(int)):
        s = max(0, s)
        e = min(e, N)
        if e <= s + 4:
            continue
        chunk = out[s:e]
        m = np.mean(chunk)
        amp = np.abs(m)
        if amp < 1e-12:
            continue
        # rotate so mean phasor aligns to +real axis (phase 0)
        out[s:e] = (chunk * (np.conj(m) / amp)).astype(np.complex64)
    return out


# ============================================================================
# Interactive UI
# ============================================================================
SLIDER_COLOR = '#303030'
FIG_BG       = '#1a1a1a'
AX_BG        = '#111111'
TEXT_COLOR   = '#dddddd'


def _apply_dc_notch(S_dB: np.ndarray, f_axis: np.ndarray,
                    notch_hz: float) -> np.ndarray:
    """Set bins within +/-notch_hz of DC to floor, hiding them in colormap."""
    if notch_hz <= 0:
        return S_dB
    mask = np.abs(f_axis) <= notch_hz
    if not np.any(mask):
        return S_dB
    out = S_dB.copy()
    out[mask, :] = np.min(S_dB) - 30.0
    return out


def build_ui(signal_10k: np.ndarray, Fs_out: float, init: dict,
             input_label: str,
             dwell_starts_10k=None, dwell_ends_10k=None):
    """Build and show the interactive spectrogram viewer."""

    has_dwells = (dwell_starts_10k is not None and len(dwell_starts_10k) > 0)
    stitch_on = [has_dwells]   # mutable: stitch ON by default when dwell data available

    # Pre-compute stitched copy once (reused on every redraw when ON)
    if has_dwells:
        print("[viewer] Pre-computing phase-stitched signal ...", end=' ', flush=True)
        sig_stitched = stitch_phase(signal_10k, dwell_starts_10k, dwell_ends_10k)
        print("done")
    else:
        sig_stitched = signal_10k

    def _active_signal():
        return sig_stitched if stitch_on[0] else signal_10k

    # ---- Initial STFT -------------------------------------------------------
    print("[viewer] Computing initial spectrogram ...", end=' ', flush=True)
    S_dB, _, f_axis, t_axis = compute_stft(
        _active_signal(), Fs_out,
        init['window_ms'], init['overlap_pct'],
        init['zero_pad_factor'], init['doppler_range_hz'],
        init['window_type'])
    print("done")

    vmax = float(np.max(S_dB))
    vmin = vmax - init['dynamic_range_db']

    # ---- Figure layout ------------------------------------------------------
    fig = plt.figure(figsize=(15, 9), facecolor=FIG_BG)
    fig.canvas.manager.set_window_title('CWHop Micro-Doppler Viewer')

    # Spectrogram axes: leaves room below for 6 sliders + controls
    ax_spec = fig.add_axes([0.07, 0.38, 0.90, 0.56], facecolor=AX_BG)
    im = ax_spec.pcolormesh(t_axis, f_axis, S_dB,
                            shading='auto', cmap='inferno',
                            vmin=vmin, vmax=vmax)
    ax_spec.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax_spec.set_ylabel('Doppler (Hz)', color=TEXT_COLOR)
    ax_spec.tick_params(colors=TEXT_COLOR)
    for sp in ax_spec.spines.values():
        sp.set_edgecolor('#555555')
    cb = fig.colorbar(im, ax=ax_spec, pad=0.01)
    cb.set_label('Power (dB)', color=TEXT_COLOR)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=TEXT_COLOR)

    def _title(w_ms, ol, zp, dr, wt, stitch, notch):
        freq_res = Fs_out / int(2 ** np.ceil(np.log2(
            max(zp, 1) * max(int(round(w_ms / 1000 * Fs_out)), 16))))
        stitch_s = '  STITCH' if stitch else ''
        notch_s  = f'  notch={notch:.0f}Hz' if notch > 0 else ''
        return (f"{os.path.basename(input_label)}  |  "
                f"win={w_ms:.0f}ms  ol={ol:.0f}%  "
                f"zp={zp}x  res={freq_res:.1f}Hz/bin  "
                f"+/-{dr:.0f}Hz  [{wt}]{stitch_s}{notch_s}")

    ax_spec.set_title(_title(init['window_ms'], init['overlap_pct'],
                             init['zero_pad_factor'], init['doppler_range_hz'],
                             init['window_type'], stitch_on[0], 0.0),
                      color=TEXT_COLOR, fontsize=9)

    # ---- Sliders  [x, y, w, h]  -- 6 rows, spaced 0.044 apart --------------
    slider_defs = [
        # (attr,             label,              vmin,  vmax,  vinit,               vstep)
        ('window_ms',        'Window (ms)',        5,   2000,  init['window_ms'],       5),
        ('overlap_pct',      'Overlap (%)',         0,    95,  init['overlap_pct'],     5),
        ('zero_pad_factor',  'Zero-pad',            1,     8,  init['zero_pad_factor'], 1),
        ('doppler_range',    'Doppler +/- (Hz)',   10,  5000,  init['doppler_range_hz'],10),
        ('dynamic_range',    'Dyn. range (dB)',    10,   120,  init['dynamic_range_db'],5),
        ('dc_notch',         'DC notch (Hz)',        0,   200,  0.0,                    5),
    ]

    sliders = {}
    for idx, (attr, label, vmin_s, vmax_s, vinit, vstep) in enumerate(slider_defs):
        ax_s = fig.add_axes([0.12, 0.315 - idx * 0.044, 0.72, 0.026],
                            facecolor=SLIDER_COLOR)
        s = Slider(ax_s, label, vmin_s, vmax_s, valinit=vinit, valstep=vstep,
                   color='#4a90d9', track_color='#2a2a2a')
        s.label.set_color(TEXT_COLOR)
        s.valtext.set_color(TEXT_COLOR)
        sliders[attr] = s

    # Window-type radio  (right column, upper)
    ax_wt = fig.add_axes([0.88, 0.140, 0.10, 0.160], facecolor=SLIDER_COLOR)
    ax_wt.set_title('Window', color=TEXT_COLOR, fontsize=8)
    wtypes = ['hann', 'blackman', 'hamming', 'rectangular']
    wt_init_idx = (wtypes.index(init['window_type'])
                   if init['window_type'] in wtypes else 0)
    from matplotlib.widgets import RadioButtons as _RB
    radio_wt = _RB(ax_wt, wtypes, active=wt_init_idx, activecolor='#4a90d9')
    for lbl in radio_wt.labels:
        lbl.set_color(TEXT_COLOR)
        lbl.set_fontsize(8)
    current_wtype = [init['window_type']]

    # Stitch checkbox  (right column, lower)
    stitch_label = 'Stitch' if has_dwells else 'Stitch(N/A)'
    ax_stitch = fig.add_axes([0.88, 0.078, 0.10, 0.050], facecolor=SLIDER_COLOR)
    check_stitch = CheckButtons(ax_stitch, [stitch_label],
                                actives=[stitch_on[0]])
    for lbl in check_stitch.labels:
        lbl.set_color(TEXT_COLOR)
        lbl.set_fontsize(8)
    # Status bar
    ax_st = fig.add_axes([0.07, 0.005, 0.78, 0.022])
    ax_st.set_axis_off()
    _init_msg = 'phase stitch ON' if stitch_on[0] else 'phase stitch OFF (no dwell data)'
    status = ax_st.text(0, 0.5,
                        f'Ready -- {_init_msg}.',
                        color=TEXT_COLOR, fontsize=8, va='center')

    # Save button
    ax_save = fig.add_axes([0.88, 0.012, 0.10, 0.028])
    btn_save = Button(ax_save, 'Save PNG', color='#333333', hovercolor='#4a90d9')
    btn_save.label.set_color(TEXT_COLOR)
    btn_save.label.set_fontsize(8)

    # ---- Update callback ----------------------------------------------------
    _busy = [False]

    def _update(val=None):
        if _busy[0]:
            return
        _busy[0] = True
        status.set_text('Computing ...')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        w_ms   = sliders['window_ms'].val
        ol     = sliders['overlap_pct'].val
        zp     = int(sliders['zero_pad_factor'].val)
        dr     = sliders['doppler_range'].val
        dyn    = sliders['dynamic_range'].val
        notch  = sliders['dc_notch'].val
        wt     = current_wtype[0]
        stitch = stitch_on[0]

        try:
            t0 = time.time()
            S_dB_new, _, f_new, t_new = compute_stft(
                _active_signal(), Fs_out, w_ms, ol, zp, dr, wt)
            S_dB_new = _apply_dc_notch(S_dB_new, f_new, notch)
            elapsed = time.time() - t0

            vmax_new = float(np.max(S_dB_new))
            vmin_new = vmax_new - dyn

            ax_spec.clear()
            ax_spec.pcolormesh(t_new, f_new, S_dB_new,
                               shading='auto', cmap='inferno',
                               vmin=vmin_new, vmax=vmax_new)
            ax_spec.set_xlabel('Time (s)', color=TEXT_COLOR)
            ax_spec.set_ylabel('Doppler (Hz)', color=TEXT_COLOR)
            ax_spec.tick_params(colors=TEXT_COLOR)
            for sp in ax_spec.spines.values():
                sp.set_edgecolor('#555555')
            ax_spec.set_title(_title(w_ms, ol, zp, dr, wt, stitch, notch),
                              color=TEXT_COLOR, fontsize=9)

            stitch_s = 'ON' if stitch else 'OFF'
            status.set_text(
                f"win={w_ms:.0f}ms  ol={ol:.0f}%  zp={zp}x  "
                f"+/-{dr:.0f}Hz  dyn={dyn:.0f}dB  "
                f"notch={notch:.0f}Hz  stitch={stitch_s}  "
                f"-- {S_dB_new.shape[0]} bins x {S_dB_new.shape[1]} frames  "
                f"({elapsed*1000:.0f}ms)")

        except Exception as exc:
            status.set_text(f'Error: {exc}')

        _busy[0] = False
        fig.canvas.draw_idle()

    def _on_wtype(label):
        current_wtype[0] = label
        _update()

    def _on_stitch(label):
        if has_dwells:
            stitch_on[0] = check_stitch.get_status()[0]
        _update()

    for s in sliders.values():
        s.on_changed(_update)
    radio_wt.on_clicked(_on_wtype)
    check_stitch.on_clicked(_on_stitch)

    # ---- Save button --------------------------------------------------------
    _save_idx = [0]

    def _on_save(event):
        base  = os.path.splitext(input_label)[0]
        fname = f"{base}_viewer_{_save_idx[0]:02d}.png"
        _save_idx[0] += 1
        fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
        status.set_text(f'Saved: {fname}')
        fig.canvas.draw_idle()

    btn_save.on_clicked(_on_save)

    plt.show()


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', '-i', required=True,
                   help='RX capture file (.cf32)')
    p.add_argument('--sidecar', '-s', default=None,
                   help='TX sidecar _schedule.json '
                        '(default: auto-detected from input path)')
    p.add_argument('--config', '-c', default=None,
                   help='cwhopping_config.json '
                        '(default: auto-detected)')
    p.add_argument('--reprocess', action='store_true',
                   help='Ignore existing cache and reprocess from raw .cf32')
    return p.parse_args()


def main():
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"ERROR: Input file not found: {input_path}")

    cache = _cache_path(input_path)

    # ---- Step 1: get or build the 10 kHz signal ----------------------------
    if os.path.isfile(cache) and not args.reprocess:
        signal_10k, Fs_out, dwell_starts_10k, dwell_ends_10k = load_cache(cache)
    else:
        sidecar_path = os.path.abspath(
            args.sidecar if args.sidecar else _find_sidecar(input_path))
        if not os.path.isfile(sidecar_path):
            sys.exit(f"ERROR: Sidecar not found: {sidecar_path}\n"
                     f"       Pass --sidecar explicitly.")
        signal_10k, Fs_out, dwell_starts_10k, dwell_ends_10k = build_cache(
            input_path, sidecar_path, cache)

    # ---- Step 2: determine initial STFT params from config -----------------
    config_path = args.config
    if config_path is None:
        config_path = _find_config(input_path)

    init = {
        'window_ms':        100.0,
        'overlap_pct':       80.0,
        'zero_pad_factor':      4,
        'doppler_range_hz':  400.0,
        'dynamic_range_db':   60.0,
        'window_type':       'hann',
    }

    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        rx_cfg = cfg.get('receiver', {})
        init['window_ms']        = float(rx_cfg.get('stft_window_ms',
                                                     init['window_ms']))
        init['overlap_pct']      = float(rx_cfg.get('overlap_pct',
                                                     init['overlap_pct']))
        init['zero_pad_factor']  = int(rx_cfg.get('zero_pad_factor',
                                                   init['zero_pad_factor']))
        init['doppler_range_hz'] = float(rx_cfg.get('doppler_range_hz',
                                                     init['doppler_range_hz']))
        wt = rx_cfg.get('window_type', init['window_type']).lower()
        init['window_type'] = wt if wt in ('hann', 'blackman',
                                            'hamming', 'rectangular') else 'hann'
        print(f"[config] Loaded initial STFT params from {config_path}")
    else:
        print("[config] No config found -- using defaults")

    print(f"\n[viewer] Initial params:")
    print(f"  Window:      {init['window_ms']:.0f} ms  "
          f"({int(round(init['window_ms']/1000*Fs_out))} samples)")
    print(f"  Overlap:     {init['overlap_pct']:.0f}%")
    print(f"  Zero-pad:    {init['zero_pad_factor']}x")
    print(f"  Doppler:     +/-{init['doppler_range_hz']:.0f} Hz")
    print(f"  Window type: {init['window_type']}")
    if dwell_starts_10k is not None:
        print(f"  Stitch:      ON  ({len(dwell_starts_10k)} dwells)")
    else:
        print(f"  Stitch:      N/A -- run with --reprocess to enable")

    # ---- Step 3: show interactive viewer -----------------------------------
    build_ui(signal_10k, Fs_out, init, input_path,
             dwell_starts_10k, dwell_ends_10k)


if __name__ == '__main__':
    main()
