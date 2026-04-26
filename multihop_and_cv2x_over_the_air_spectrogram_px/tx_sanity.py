#!/usr/bin/env python3
"""
tx_sanity.py -- Interactive raw spectrogram of a hopped-CW .cf32 file.

No de-hopping. Shows the CW tone jumping between hop frequencies.
Sidecar overlay (cyan) shows the expected schedule -- should sit on the tone.

Usage:
    python tx_sanity.py --input my_capture.cf32 --sidecar my_capture_schedule.json
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# ── spectrogram engine ────────────────────────────────────────────────────────
def build_spectrogram(iq, rate, chunk_ms, overlap_pct, zoom_hz, win_name):
    chunk    = max(64, int(rate * chunk_ms / 1000.0))
    step     = max(1,  int(chunk * (1.0 - overlap_pct / 100.0)))
    n_frames = max(1, (len(iq) - chunk) // step + 1)
    freq_res = rate / chunk
    n_keep   = max(1, min(int(zoom_hz / freq_res), chunk // 2))
    actual_zoom = n_keep * freq_res

    if   win_name == "hann":     win = np.hanning(chunk).astype(np.float32)
    elif win_name == "blackman": win = np.blackman(chunk).astype(np.float32)
    elif win_name == "hamming":  win = np.hamming(chunk).astype(np.float32)
    else:                        win = np.ones(chunk, dtype=np.float32)

    mid  = chunk // 2
    spec = np.empty((2 * n_keep, n_frames), dtype=np.float32)
    for i in range(n_frames):
        s   = i * step
        seg = iq[s:s + chunk] * win
        S   = np.abs(np.fft.fftshift(np.fft.fft(seg)))
        spec[:, i] = S[mid - n_keep : mid + n_keep]

    spec_db  = 20.0 * np.log10(spec + 1e-30)
    spec_db -= np.max(spec_db)

    f_axis = np.linspace(-actual_zoom, actual_zoom, 2 * n_keep)
    t_axis = (np.arange(n_frames) * step + chunk // 2) / rate
    return f_axis, t_axis, spec_db, freq_res, actual_zoom


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",     required=True)
    parser.add_argument("--sidecar",   default=None)
    parser.add_argument("--zoom-hz",   type=float, default=20000)
    parser.add_argument("--chunk-ms",  type=float, default=5.0)
    parser.add_argument("--dyn-range", type=float, default=60.0)
    parser.add_argument("--decimate",  type=int,   default=256,
                        help="Decimate factor (default 256 → 120 kHz). "
                             "Use 1 to skip.")
    args = parser.parse_args()

    # ── load sidecar ──────────────────────────────────────────────────────────
    rate    = 30720000.0
    sidecar = None
    if args.sidecar:
        with open(args.sidecar) as f:
            sidecar = json.load(f)
        rate = float(sidecar.get("Fs_hz", sidecar.get("tx_rate", rate)))

    # ── load IQ ───────────────────────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    iq_raw = np.fromfile(args.input, dtype=np.complex64)
    print(f"  {len(iq_raw):,} samp  {rate/1e6:.3f} MHz  {len(iq_raw)/rate:.3f} s")

    # ── decimate once ─────────────────────────────────────────────────────────
    if args.decimate > 1:
        from scipy.signal import decimate as sp_dec
        D = args.decimate
        print(f"  Decimating x{D} → {rate/D/1e3:.1f} kHz ...")
        iq = iq_raw
        for factor in [4, 4, 4, 4, 2, 2]:
            if D <= 1: break
            if D % factor == 0:
                iq = sp_dec(iq, factor, ftype='fir', zero_phase=True)
                D //= factor
        if D > 1:
            iq = sp_dec(iq, D, ftype='fir', zero_phase=True)
        rate = rate / args.decimate
        print(f"  Done: {len(iq):,} samp  {rate/1e3:.1f} kHz")
    else:
        iq = iq_raw
    del iq_raw

    # ── initial spectrogram ───────────────────────────────────────────────────
    windows  = ["hann", "blackman", "hamming", "rect"]
    cur_win  = [0]

    f_ax, t_ax, spec_db, _, _ = build_spectrogram(
        iq, rate, args.chunk_ms, 0.0, args.zoom_hz, windows[cur_win[0]])

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8), facecolor='#1a1a1a')
    ax  = fig.add_axes([0.06, 0.38, 0.82, 0.56], facecolor='black')
    cax = fig.add_axes([0.90, 0.38, 0.015, 0.56])

    dyn     = args.dyn_range
    pcm_ref = [ax.pcolormesh(t_ax, f_ax, np.maximum(spec_db, -dyn),
                              shading='nearest', cmap='inferno', vmin=-dyn, vmax=0)]
    cbar    = fig.colorbar(pcm_ref[0], cax=cax)
    cbar.set_label('Power (dB re peak)', color='white')
    cbar.ax.tick_params(colors='white')

    def _style_ax():
        ax.set_xlabel("Time (s)",                     color='white')
        ax.set_ylabel("Freq offset from centre (Hz)", color='white')
        ax.tick_params(colors='white')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')

    _style_ax()

    # ── hop-schedule overlay (stored so we can remove/replace) ────────────────
    hop_artists = []

    def draw_hops(zoom):
        for a in hop_artists:
            a.remove()
        hop_artists.clear()
        if not sidecar:
            return
        hops = sidecar.get("hop_schedule", [])
        if not hops or not isinstance(hops[0], dict):
            return
        labeled = False
        for h in hops:
            fh = h["freq_hz"]
            if abs(fh) > zoom:
                continue
            ln, = ax.plot([h["start_ms"]/1000, h["end_ms"]/1000], [fh, fh],
                          color='cyan', linewidth=1.0, alpha=0.6,
                          label="schedule" if not labeled else "")
            hop_artists.append(ln)
            labeled = True
        if labeled:
            leg = ax.legend(loc='upper right', fontsize=8,
                            facecolor='#333333', labelcolor='white')
            hop_artists.append(leg)

    draw_hops(args.zoom_hz)

    # ── sliders ───────────────────────────────────────────────────────────────
    def mks(rect, label, lo, hi, init, step):
        ax = fig.add_axes(rect, facecolor='#2a2a2a')
        s  = Slider(ax, label, lo, hi, valinit=init, valstep=step, color='steelblue')
        s.label.set_color('white')
        s.valtext.set_color('white')
        return s

    s_chunk = mks([0.06, 0.27, 0.55, 0.022], "Chunk (ms)",   0.5,  50.0, args.chunk_ms, 0.5)
    s_ol    = mks([0.06, 0.24, 0.55, 0.022], "Overlap (%)",  0,    90,   0,             5  )
    s_zoom  = mks([0.06, 0.21, 0.55, 0.022], "Zoom ±(Hz)",   500,  60000,args.zoom_hz,  500)
    s_dyn   = mks([0.06, 0.18, 0.55, 0.022], "Dyn range(dB)",10,   100,  args.dyn_range,5  )

    ax_btn = fig.add_axes([0.72, 0.17, 0.12, 0.10], facecolor='#2a2a2a')
    btn    = Button(ax_btn, f"Window:\n{windows[0]}", color='#2a2a2a', hovercolor='#444')
    btn.label.set_color('white')
    btn.label.set_fontsize(9)

    def cycle_win(event):
        cur_win[0] = (cur_win[0] + 1) % len(windows)
        btn.label.set_text(f"Window:\n{windows[cur_win[0]]}")
        _do_update()

    btn.on_clicked(cycle_win)

    # ── update ────────────────────────────────────────────────────────────────
    def _do_update():
        f_ax, t_ax, spec_db, freq_res, actual_zoom = build_spectrogram(
            iq, rate,
            s_chunk.val, s_ol.val, s_zoom.val, windows[cur_win[0]])
        dyn = s_dyn.val

        # replace pcolormesh without clearing axes
        pcm_ref[0].remove()
        pcm_ref[0] = ax.pcolormesh(t_ax, f_ax, np.maximum(spec_db, -dyn),
                                   shading='nearest', cmap='inferno',
                                   vmin=-dyn, vmax=0)
        cbar.update_normal(pcm_ref[0])
        ax.set_xlim(t_ax[0], t_ax[-1])
        ax.set_ylim(-actual_zoom, actual_zoom)
        _style_ax()
        ax.set_title(
            f"{args.input}  chunk={s_chunk.val:.1f}ms  ol={s_ol.val:.0f}%  "
            f"zoom=±{actual_zoom:.0f}Hz  res={freq_res:.1f}Hz/bin  "
            f"dyn={dyn:.0f}dB  win={windows[cur_win[0]]}",
            color='white', fontsize=8)
        draw_hops(actual_zoom)
        fig.canvas.draw_idle()

    s_chunk.on_changed(lambda v: _do_update())
    s_ol   .on_changed(lambda v: _do_update())
    s_zoom .on_changed(lambda v: _do_update())
    s_dyn  .on_changed(lambda v: _do_update())

    ax.set_title(
        f"{args.input}  chunk={args.chunk_ms:.1f}ms  zoom=±{args.zoom_hz:.0f}Hz  "
        f"dyn={args.dyn_range:.0f}dB  win={windows[0]}",
        color='white', fontsize=8)

    ax_st = fig.add_axes([0.06, 0.02, 0.88, 0.02])
    ax_st.set_axis_off()
    ax_st.text(0, 0.5,
               f"{len(iq):,} samp @ {rate/1e3:.1f} kHz  "
               f"({len(iq)/rate:.3f} s)  —  adjust sliders to update",
               color='#aaaaaa', fontsize=8, va='center')

    plt.show()


if __name__ == "__main__":
    main()
