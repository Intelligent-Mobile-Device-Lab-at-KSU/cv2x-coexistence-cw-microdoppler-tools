#!/usr/bin/env python3
"""
cwhopping_explorer.py — Interactive CW frequency hopping micro-Doppler explorer.

Tabbed UI with three panels:
  1. Hopping CW — dwell, taper, gap, hop pattern
  2. SC-FDMA Data — coexistence with sidelink data transmissions
  3. Rx / Sim — pedestrian model, STFT processing, duration, SNR

PHY basis: 20 MHz C-V2X (1199 sensing bins at 7.5 kHz spacing).

Usage:
    python cwhopping_explorer.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from matplotlib.gridspec import GridSpec
import time

from walking_micro_doppler import PedestrianParams
from generate_reference_cw import RefCWParams, generate_reference_cw
from generate_hopped_cw import HoppedCWParams, HopDwell, generate_hopped_cw
from dehop_and_stft import STFTParams, dehop_and_stft
from compute_quality_metric import compute_quality_metric
from generate_scfdma_data import SCFDMAParams, generate_scfdma_interference


class CWHoppingExplorer:
    """Interactive CW hopping / micro-Doppler explorer with tabbed controls."""

    def __init__(self):
        self.Fs = 100e3
        self.sig_hopped = None
        self.sig_reference = None
        self.hop_schedule = None
        self.envelope = None
        self.scfdma_result = None
        self.dirty_signal = True
        self.dirty_stft = True
        self._updating = False
        self._current_tab = 0
        self._tab_artists = [[], [], []]  # per-tab visibility groups

        self._build_gui()
        self._switch_tab(0)
        self._update_all()

    # ================================================================
    #  GUI CONSTRUCTION
    # ================================================================

    def _build_gui(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor='#1a1a1a')
        self.fig.canvas.manager.set_window_title(
            'CW Hopping / Micro-Doppler Explorer')

        # --- Plots (right side, always visible) ---
        gs = GridSpec(2, 2, figure=self.fig,
                      left=0.32, right=0.98, top=0.95, bottom=0.05,
                      hspace=0.30, wspace=0.20)
        self.ax_hop = self.fig.add_subplot(gs[0, 0])
        self.ax_ref = self.fig.add_subplot(gs[0, 1])
        self.ax_td  = self.fig.add_subplot(gs[1, 0])
        self.ax_met = self.fig.add_subplot(gs[1, 1])
        for ax in [self.ax_hop, self.ax_ref, self.ax_td, self.ax_met]:
            ax.set_facecolor('#111115')
            ax.tick_params(colors='#cccccc', labelsize=8)
            for sp in ax.spines.values():
                sp.set_color('#444444')

        # --- Tab buttons (wider, better spaced) ---
        tab_names = ['Hopping CW', 'SC-FDMA', 'Rx / Sim']
        self._tab_btn_axes = []
        self._tab_btns = []
        for i, name in enumerate(tab_names):
            ax = self.fig.add_axes([0.02 + i * 0.09, 0.955, 0.085, 0.035])
            btn = Button(ax, name, color='#223344', hovercolor='#335566')
            btn.label.set_color('white')
            btn.label.set_fontsize(9)
            btn.on_clicked(lambda _, idx=i: self._switch_tab(idx))
            self._tab_btn_axes.append(ax)
            self._tab_btns.append(btn)

        # --- Control panel border ---
        from matplotlib.patches import FancyBboxPatch
        border = FancyBboxPatch((0.015, 0.06), 0.275, 0.89,
            boxstyle='round,pad=0.005', linewidth=1,
            edgecolor='#334455', facecolor='none',
            transform=self.fig.transFigure, clip_on=False)
        self.fig.patches.append(border)

        # --- Shared styling ---
        sc = '#334455'   # slider color
        lc = '#cccccc'   # label color
        bg = '#1a1a2a'   # radio bg
        SL_X = 0.06      # slider left edge (room for labels)
        SL_W = 0.20      # slider width
        SL_H = 0.018     # slider height
        LBL_X = 0.03     # label x position

        def _sl(y, label, lo, hi, v0, fmt='%.1f', tab=0):
            ax = self.fig.add_axes([SL_X, y, SL_W, SL_H], facecolor=sc)
            sl = Slider(ax, label, lo, hi, valinit=v0, valfmt=fmt,
                        color='#5588bb')
            ax.xaxis.label.set_color(lc)
            ax.tick_params(colors=lc, labelsize=7)
            for t in [sl.label, sl.valtext]:
                t.set_color(lc); t.set_fontsize(8)
            self._tab_artists[tab].append(ax)
            return sl

        def _lbl(y, txt, color='#66aaff', tab=0, **kw):
            t = self.fig.text(LBL_X, y, txt, fontsize=kw.get('fs', 10),
                              fontweight=kw.get('fw', 'bold'), color=color)
            self._tab_artists[tab].append(t)
            return t

        def _radio(rect, items, active, tab, fs=8):
            ax = self.fig.add_axes(rect, facecolor=bg)
            ax.set_frame_on(True)
            for sp in ax.spines.values():
                sp.set_color('#334455')
                sp.set_linewidth(0.5)
            rb = RadioButtons(ax, items, active=active, activecolor='#5588bb')
            for l in rb.labels:
                l.set_color(lc); l.set_fontsize(fs)
            self._tab_artists[tab].append(ax)
            return rb

        # ============================================================
        #  TAB 0 — HOPPING CW
        # ============================================================
        _lbl(0.925, 'TRANSMITTER', '#66aaff', 0)
        self.sl_dwell     = _sl(0.895, 'Dwell (ms)',  1, 100, 5,   '%.0f', 0)
        self.sl_sparsity  = _sl(0.865, 'Hop Sparse',  0,   1, 1.0, '%.2f', 0)
        self.sl_taper_pct = _sl(0.835, 'Taper %',     0,  50, 10,  '%.0f', 0)
        self.sl_min_amp   = _sl(0.805, 'Min Amp',     0, 0.5, 0.05,'%.2f', 0)
        self.sl_gap       = _sl(0.775, 'Gap (ms)',     0, 100, 0,   '%.0f', 0)
        self.sl_gap_jit   = _sl(0.745, 'Gap Jitter',  0, 100, 0,   '%.0f', 0)

        # Radio groups — left column: taper shape, right column: transition + hops
        t = self.fig.text(LBL_X, 0.722, 'Taper Shape:', fontsize=8, color=lc)
        self._tab_artists[0].append(t)
        self.radio_taper = _radio([0.03, 0.60, 0.12, 0.12],
            ['none', 'linear', 'raised_cosine', 'hann', 'blackman'], 2, 0, fs=8)

        t = self.fig.text(0.165, 0.722, 'Transition:', fontsize=8, color=lc)
        self._tab_artists[0].append(t)
        self.radio_trans = _radio([0.165, 0.67, 0.11, 0.05],
            ['step', 'chirp'], 1, 0, fs=8)

        t = self.fig.text(0.165, 0.655, 'Hop Freqs:', fontsize=8, color=lc)
        self._tab_artists[0].append(t)
        self.radio_nhops = _radio([0.165, 0.60, 0.11, 0.055],
            ['2', '4', '8'], 1, 0, fs=8)

        # ============================================================
        #  TAB 1 — SC-FDMA DATA
        # ============================================================
        _lbl(0.925, 'SC-FDMA COEXISTENCE', '#ff8866', 1)
        _lbl(0.90, 'Co-located TX: data + sensing same vehicle', '#888888', 1,
             fs=8, fw='normal')

        # Enable checkbox
        ax_en = self.fig.add_axes([0.03, 0.87, 0.24, 0.025], facecolor=bg)
        ax_en.set_frame_on(True)
        for sp in ax_en.spines.values():
            sp.set_color('#334455'); sp.set_linewidth(0.5)
        self.cb_scfdma = CheckButtons(ax_en, ['Enable SC-FDMA data'], [False])
        for l in self.cb_scfdma.labels:
            l.set_color(lc); l.set_fontsize(9)
        self._tab_artists[1].append(ax_en)

        # Scenario radio buttons
        _lbl(0.845, 'TX SCENARIO:', '#ff8866', 1, fs=9)
        self.radio_scenario = _radio([0.03, 0.77, 0.24, 0.07],
            ['Co-located, CW inside alloc',
             'Co-located, CW outside alloc',
             'Other vehicle (external)'], 0, 1, fs=8)

        self.sl_data_act   = _sl(0.74, 'Data Activity', 0, 1, 0.5, '%.2f', 1)
        self.sl_subch_max  = _sl(0.71, 'Max SubCH',     1, 20, 5,  '%.0f', 1)
        self.sl_data_pwr   = _sl(0.68, 'Data Pwr (dB)', -10, 30, 10,'%.0f', 1)
        self.sl_isolation  = _sl(0.65, 'Isolation (dB)', 10, 40, 23,'%.0f', 1)

        _lbl(0.62, 'SCENARIO INFO:', '#ff8866', 1, fs=9)
        self._scenario_text = self.fig.text(LBL_X, 0.56,
            'CW inside: 100% overlap when data active.\n'
            'Allocation size does not affect overlap.\n'
            'Interference = sidelobe at 7.5 kHz offset.',
            fontsize=8, color='#777777', linespacing=1.5)
        self._tab_artists[1].append(self._scenario_text)

        _lbl(0.50, 'SIR AT CW BIN:', '#ff8866', 1, fs=9)
        self._sir_text = self.fig.text(LBL_X, 0.47,
            'SIR = Isolation - Data Pwr = 13.0 dB',
            fontsize=10, color='#ffaa66', fontweight='bold')
        self._tab_artists[1].append(self._sir_text)

        # ============================================================
        #  TAB 2 — RX / SIMULATION
        # ============================================================
        _lbl(0.925, 'TARGET', '#66ff99', 2)
        self.sl_vbulk    = _sl(0.895, 'Speed (m/s)',  0,   3, 1.2, '%.1f', 2)
        self.sl_arm_fmd  = _sl(0.865, 'Arm fmd (Hz)', 0, 300, 80,  '%.0f', 2)
        self.sl_arm_frot = _sl(0.835, 'Arm Rate',   0.5,   4, 1.8, '%.1f', 2)

        _lbl(0.805, 'PROCESSING', '#ffcc66', 2)
        self.sl_stft_win = _sl(0.775, 'STFT Win (ms)', 10, 500, 100, '%.0f', 2)
        self.sl_overlap  = _sl(0.745, 'Overlap %',     50,  95,  85, '%.0f', 2)
        self.sl_doppler  = _sl(0.715, 'Doppler (Hz)', 100,2000, 400, '%.0f', 2)
        self.sl_snr      = _sl(0.685, 'SNR (dB)',       0,  60,  30, '%.0f', 2)

        # Radio groups — left: window type, right: zero-pad
        t = self.fig.text(LBL_X, 0.660, 'Window:', fontsize=8, color=lc)
        self._tab_artists[2].append(t)
        self.radio_wtype = _radio([0.03, 0.535, 0.12, 0.12],
            ['hann', 'blackman', 'hamming', 'kaiser', 'rectangular'], 0, 2, fs=8)

        t = self.fig.text(0.165, 0.660, 'Zero-Pad:', fontsize=8, color=lc)
        self._tab_artists[2].append(t)
        self.radio_zeropad = _radio([0.165, 0.585, 0.11, 0.07],
            ['1x', '2x', '4x', '8x'], 2, 2, fs=8)

        # Checkboxes
        ax_ck = self.fig.add_axes([0.03, 0.49, 0.24, 0.04], facecolor=bg)
        ax_ck.set_frame_on(True)
        for sp in ax_ck.spines.values():
            sp.set_color('#334455'); sp.set_linewidth(0.5)
        self.checks = CheckButtons(ax_ck,
            ['Per-dwell window', 'Blank transitions'], [False, False])
        for l in self.checks.labels:
            l.set_color(lc); l.set_fontsize(8)
        self._tab_artists[2].append(ax_ck)

        t = self.fig.text(LBL_X, 0.470, 'Duration:', fontsize=8, color=lc)
        self._tab_artists[2].append(t)
        self.radio_dur = _radio([0.03, 0.40, 0.24, 0.065],
            ['1s', '2s', '3s', '5s'], 1, 2, fs=8)

        # ============================================================
        #  ALWAYS VISIBLE — bottom bar
        # ============================================================
        ax_btn = self.fig.add_axes([0.03, 0.07, 0.12, 0.04])
        self.btn_regen = Button(ax_btn, 'REGENERATE',
                                color='#224466', hovercolor='#336688')
        self.btn_regen.label.set_color('white')
        self.btn_regen.label.set_fontweight('bold')

        ax_exp = self.fig.add_axes([0.16, 0.07, 0.12, 0.04])
        self.btn_export = Button(ax_exp, 'EXPORT CFG',
                                 color='#224455', hovercolor='#336655')
        self.btn_export.label.set_color('white')
        self.btn_export.label.set_fontweight('bold')

        self.status_text = self.fig.text(
            LBL_X, 0.03, 'Ready', fontsize=9, color='#88ff88')
        self.fig.text(LBL_X, 0.012,
            '20 MHz C-V2X  |  Fs=100kHz  |  1199 sensing bins',
            fontsize=8, color='#666666')

        # ============================================================
        #  WIRE CALLBACKS
        # ============================================================
        sig_sliders = [self.sl_dwell, self.sl_sparsity, self.sl_taper_pct,
                       self.sl_min_amp, self.sl_gap, self.sl_gap_jit,
                       self.sl_vbulk, self.sl_arm_fmd, self.sl_arm_frot,
                       self.sl_snr, self.sl_data_act, self.sl_subch_max,
                       self.sl_data_pwr, self.sl_isolation]
        for sl in sig_sliders:
            sl.on_changed(self._on_signal_changed)

        for rb in [self.radio_taper, self.radio_trans, self.radio_nhops,
                   self.radio_dur, self.radio_scenario]:
            rb.on_clicked(self._on_signal_changed)
        self.cb_scfdma.on_clicked(self._on_signal_changed)

        stft_sliders = [self.sl_stft_win, self.sl_overlap, self.sl_doppler]
        for sl in stft_sliders:
            sl.on_changed(self._on_stft_changed)
        for rb in [self.radio_wtype, self.radio_zeropad]:
            rb.on_clicked(self._on_stft_changed)
        self.checks.on_clicked(self._on_stft_changed)

        self.btn_regen.on_clicked(self._on_signal_changed)
        self.btn_export.on_clicked(self._on_export_config)

    # ================================================================
    #  TAB SWITCHING
    # ================================================================

    def _switch_tab(self, idx):
        self._current_tab = idx
        for i, group in enumerate(self._tab_artists):
            vis = (i == idx)
            for artist in group:
                artist.set_visible(vis)
                # Force-hide axes children too (prevents radio/check dots
                # from bleeding through on some matplotlib backends)
                if isinstance(artist, plt.Axes):
                    for child in artist.get_children():
                        child.set_visible(vis)
        # Highlight active tab button
        for i, btn in enumerate(self._tab_btns):
            btn.color = '#446688' if i == idx else '#223344'
            btn.ax.set_facecolor(btn.color)
        self.fig.canvas.draw_idle()

    # ================================================================
    #  CALLBACKS
    # ================================================================

    def _on_signal_changed(self, _=None):
        self._update_sir_text()
        self._update_scenario_text()
        self.dirty_signal = True
        self.dirty_stft = True
        self._update_all()

    def _on_stft_changed(self, _=None):
        self.dirty_stft = True
        self._update_all()

    def _update_sir_text(self):
        sir = self.sl_isolation.val - self.sl_data_pwr.val
        self._sir_text.set_text(
            f'SIR = Isolation - Data Pwr = {sir:.1f} dB')

    def _update_scenario_text(self):
        sel = self.radio_scenario.value_selected
        if sel.startswith('Co-located, CW inside'):
            self._scenario_text.set_text(
                'CW inside: 100% overlap when data active.\n'
                'Allocation size does not affect overlap.\n'
                'Interference = sidelobe at 7.5 kHz offset.')
        elif sel.startswith('Co-located, CW outside'):
            self._scenario_text.set_text(
                'CW outside own allocation. Leakage reduced\n'
                'by spectral distance (~12 dB/subchannel).\n'
                'Larger alloc = CW pushed farther = less interference.')
        else:
            self._scenario_text.set_text(
                'Other vehicle transmits data randomly.\n'
                'Overlap depends on random allocation vs CW bin.\n'
                'Larger alloc = higher overlap probability.')

    # ================================================================
    #  PARAMETER GATHERING
    # ================================================================

    def _gather_params(self):
        duration_s = float(self.radio_dur.value_selected.replace('s', ''))
        n_hops = int(self.radio_nhops.value_selected)
        hop_freqs = np.linspace(7.5e3, 7.5e3 * n_hops, n_hops)

        ped = PedestrianParams(
            v_bulk=self.sl_vbulk.val,
            arm_f_md=self.sl_arm_fmd.val,
            arm_f_rot=self.sl_arm_frot.val)

        hop_params = HoppedCWParams(
            Fs=self.Fs, duration_s=duration_s,
            dwell_ms=round(self.sl_dwell.val),
            hop_freqs=hop_freqs,
            hop_sparsity=self.sl_sparsity.val,
            taper_shape=self.radio_taper.value_selected,
            taper_pct=self.sl_taper_pct.val / 100,
            min_amplitude=self.sl_min_amp.val,
            transition_mode=self.radio_trans.value_selected,
            A_cw=1.0, SNR_dB=self.sl_snr.val,
            pedestrian=ped,
            gap_ms=round(self.sl_gap.val),
            gap_jitter_ms=round(self.sl_gap_jit.val))

        ref_params = RefCWParams(
            Fs=self.Fs, duration_s=duration_s,
            center_freq=hop_freqs[0],
            A_cw=1.0, SNR_dB=self.sl_snr.val,
            pedestrian=ped)

        zp_str = self.radio_zeropad.value_selected
        check_states = self.checks.get_status()
        stft_params = STFTParams(
            window_ms=self.sl_stft_win.val,
            window_type=self.radio_wtype.value_selected,
            overlap_pct=self.sl_overlap.val,
            zero_pad_factor=int(zp_str.replace('x', '')),
            per_dwell_window=check_states[0],
            blank_transitions=check_states[1],
            taper_pct=self.sl_taper_pct.val / 100,
            doppler_range_hz=self.sl_doppler.val)

        scfdma_en = self.cb_scfdma.get_status()[0]
        scenario = self.radio_scenario.value_selected
        colocated = scenario.startswith('Co-located')
        cw_inside = scenario.startswith('Co-located, CW inside')
        scfdma_params = SCFDMAParams(
            enabled=scfdma_en,
            data_activity=self.sl_data_act.val,
            num_subch_max=max(1, round(self.sl_subch_max.val)),
            data_power_dB=self.sl_data_pwr.val,
            spectral_isolation_dB=self.sl_isolation.val,
            colocated_tx=colocated,
            cw_inside_alloc=cw_inside)

        return hop_params, ref_params, stft_params, scfdma_params, duration_s

    # ================================================================
    #  UPDATE / RENDER
    # ================================================================

    def _update_all(self):
        if self._updating:
            return
        self._updating = True
        self.status_text.set_text('Computing...')
        self.status_text.set_color('#ffff66')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        try:
            t0 = time.time()
            hp, rp, sp, sfp, dur = self._gather_params()

            if self.dirty_signal:
                self.sig_hopped, _, self.hop_schedule, self.envelope = \
                    generate_hopped_cw(hp)
                self.sig_reference, _ = generate_reference_cw(rp)

                # SC-FDMA interference
                self.scfdma_result = generate_scfdma_interference(
                    sfp, len(self.sig_hopped), self.Fs,
                    self.hop_schedule, hp.A_cw)
                self.sig_hopped = self.sig_hopped + self.scfdma_result.interference

                self.dirty_signal = False
                self.dirty_stft = True

            if self.dirty_stft:
                S_hop_dB, f_hop, t_hop, S_hop_lin = dehop_and_stft(
                    self.sig_hopped, self.Fs, self.hop_schedule, sp)

                ref_hop = [HopDwell(start_sample=0,
                           end_sample=len(self.sig_reference) - 1,
                           freq_hz=hp.hop_freqs[0], dwell_idx=0)]
                S_ref_dB, f_ref, t_ref, S_ref_lin = dehop_and_stft(
                    self.sig_reference, self.Fs, ref_hop, sp)

                metrics = compute_quality_metric(
                    S_hop_lin, S_ref_lin, f_hop, t_hop)
                self.dirty_stft = False

            # === RENDER ===
            self.ax_hop.clear()
            if S_hop_dB.size > 0:
                vm = np.max(S_hop_dB)
                self.ax_hop.pcolormesh(t_hop, f_hop, S_hop_dB, cmap='jet',
                                       vmin=vm-40, vmax=vm, shading='auto')
            self.ax_hop.set_title('De-hopped Spectrogram', color='white', fontsize=10)
            self.ax_hop.set_xlabel('Time (s)', color='#aaa', fontsize=8)
            self.ax_hop.set_ylabel('Doppler (Hz)', color='#aaa', fontsize=8)

            self.ax_ref.clear()
            if S_ref_dB.size > 0:
                vm = np.max(S_ref_dB)
                self.ax_ref.pcolormesh(t_ref, f_ref, S_ref_dB, cmap='jet',
                                       vmin=vm-40, vmax=vm, shading='auto')
            self.ax_ref.set_title('Reference (no hop)', color='white', fontsize=10)
            self.ax_ref.set_xlabel('Time (s)', color='#aaa', fontsize=8)
            self.ax_ref.set_ylabel('Doppler (Hz)', color='#aaa', fontsize=8)

            # Envelope + hops + SC-FDMA overlay
            self.ax_td.clear()
            t_td = np.arange(len(self.envelope)) / self.Fs * 1000
            self.ax_td.plot(t_td, self.envelope, 'w-', linewidth=0.5)

            mf = max(d.freq_hz for d in self.hop_schedule)
            for dw in self.hop_schedule:
                ts = dw.start_sample / self.Fs * 1000
                te = dw.end_sample / self.Fs * 1000
                fn = dw.freq_hz / mf
                self.ax_td.plot([ts, te], [fn, fn], '-',
                                color=(0.3, 0.8, 1.0, 0.6), linewidth=2)

            # SC-FDMA subframe markers
            if sfp.enabled and self.scfdma_result is not None:
                sps = round(self.Fs * 1e-3)
                for sf in range(len(self.scfdma_result.data_mask)):
                    if not self.scfdma_result.data_mask[sf]:
                        continue
                    t_s = sf * sps / self.Fs * 1000
                    t_e = (sf + 1) * sps / self.Fs * 1000
                    if self.scfdma_result.overlap_mask[sf]:
                        c = (1.0, 0.3, 0.3, 0.25)  # red = overlap
                    else:
                        c = (1.0, 0.8, 0.2, 0.15)  # yellow = data, no overlap
                    self.ax_td.axvspan(t_s, t_e, color=c)

            self.ax_td.set_xlabel('Time (ms)', color='#aaa', fontsize=8)
            self.ax_td.set_ylabel('Amp / Freq', color='#aaa', fontsize=8)
            dv = max(round(hp.dwell_ms), 1)
            nh = len(hp.hop_freqs)
            gap_avg = round(self.sl_gap.val)
            cyc = dv * nh + gap_avg * nh
            span = min(max(3 * cyc, 30), dur * 1000)
            self.ax_td.set_xlim([0, span])
            ttl = f'Envelope + Hops (dwell={dv}ms)'
            if sfp.enabled:
                n_data = int(np.sum(self.scfdma_result.data_mask))
                n_ovlp = int(np.sum(self.scfdma_result.overlap_mask))
                ttl += f'  |  SC-FDMA: {n_data} sf, {n_ovlp} overlap'
            self.ax_td.set_title(ttl, color='white', fontsize=9)

            # Quality metrics
            self.ax_met.clear()
            labels = ['Correlation', 'Sidelobe\n(norm)', 'Dyn Range\n(norm)']
            vals = [metrics.correlation,
                    max(metrics.peak_sidelobe, -60) / -60,
                    min(metrics.dynamic_range, 40) / 40]
            self.ax_met.barh(labels, vals, color='#5588bb', height=0.5)
            self.ax_met.set_xlim([0, 1.1])
            self.ax_met.set_title('Quality Metrics', color='white', fontsize=10)
            txt = (f'Corr: {metrics.correlation:.3f}\n'
                   f'Sidelobe: {metrics.peak_sidelobe:.1f} dB\n'
                   f'Dyn Range: {metrics.dynamic_range:.1f} dB\n'
                   f'MSE: {metrics.mse_dB:.1f} dB')
            self.ax_met.text(0.55, 1.5, txt, color='white', fontsize=8,
                             va='center', transform=self.ax_met.transData)

            elapsed = (time.time() - t0) * 1000
            self.status_text.set_text(f'Done ({elapsed:.0f} ms)')
            self.status_text.set_color('#88ff88')

        except Exception as e:
            self.status_text.set_text(f'Error: {e}')
            self.status_text.set_color('#ff6666')
            import traceback
            traceback.print_exc()

        self.fig.canvas.draw_idle()
        self._updating = False

    # ================================================================
    #  EXPORT CONFIG
    # ================================================================

    def _on_export_config(self, _=None):
        n_hops = int(self.radio_nhops.value_selected)
        hop_freqs = np.linspace(7.5e3, 7.5e3 * n_hops, n_hops).tolist()
        zp_str = self.radio_zeropad.value_selected
        ck = self.checks.get_status()
        Fs_hw = 30720000
        scfdma_en = self.cb_scfdma.get_status()[0]

        config = {
            'version': 1,
            'transmitter': {
                'dwell_ms':        round(self.sl_dwell.val),
                'hop_freqs_hz':    hop_freqs,
                'hop_sparsity':    round(self.sl_sparsity.val, 3),
                'taper_shape':     self.radio_taper.value_selected,
                'taper_pct':       round(self.sl_taper_pct.val / 100, 3),
                'min_amplitude':   round(self.sl_min_amp.val, 3),
                'transition_mode': self.radio_trans.value_selected,
                'gap_ms':          round(self.sl_gap.val),
                'gap_jitter_ms':   round(self.sl_gap_jit.val),
                'A_cw':            1.0,
            },
            'scfdma': {
                'enabled':              scfdma_en,
                'scenario':             self.radio_scenario.value_selected,
                'colocated_tx':         self.radio_scenario.value_selected.startswith('Co-located'),
                'cw_inside_alloc':      self.radio_scenario.value_selected.startswith('Co-located, CW inside'),
                'data_activity':        round(self.sl_data_act.val, 3),
                'num_subch_max':        max(1, round(self.sl_subch_max.val)),
                'data_power_dB':        round(self.sl_data_pwr.val, 1),
                'spectral_isolation_dB': round(self.sl_isolation.val, 1),
            },
            'receiver': {
                'stft_window_ms':    round(self.sl_stft_win.val),
                'window_type':       self.radio_wtype.value_selected,
                'overlap_pct':       round(self.sl_overlap.val),
                'zero_pad_factor':   int(zp_str.replace('x', '')),
                'per_dwell_window':  bool(ck[0]),
                'blank_transitions': bool(ck[1]),
                'doppler_range_hz':  round(self.sl_doppler.val),
            },
            'hardware': {
                'Fs_hz':             Fs_hw,
                'center_freq_hz':    5900000000,
                'bandwidth_hz':      20000000,
                'num_sensing_bins':  1199,
                'bin_spacing_hz':    7500,
                'decimation_factor': Fs_hw // 10000,
            },
        }
        fname = 'cwhopping_config.json'
        with open(fname, 'w') as f:
            json.dump(config, f, indent=2)
        self.status_text.set_text(f'Exported: {fname}')
        self.status_text.set_color('#66ffaa')
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    explorer = CWHoppingExplorer()
    explorer.show()


if __name__ == '__main__':
    main()
