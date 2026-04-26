#!/usr/bin/env python3
"""
cw_doppler.py  --  CW Micro-Doppler Monitor
=============================================
Single CW tone TX + real-time micro-Doppler display.

Display layout (3 panels stacked):
  1. TX baseband spectrum  -- confirms CW tone is transmitting
  2. RX averaged spectrum  -- FFT with temporal averaging so Doppler
                              energy is visible as a stable bump either
                              side of the CW peak
  3. RX waterfall          -- scrolling spectrogram: time flows downward,
                              Doppler shifts appear as coloured arcs
                              moving left/right of the CW centre line

Key parameters for tuning responsiveness vs smoothness:
  --avg-frames N    Number of FFT frames to average (default 20).
                    Higher = smoother but slower to respond.
                    Try 10 for fast gestures, 50 for slow vehicle motion.
  --update-time S   Display refresh rate in seconds (default 0.05 = 20 fps).
-----
# TX + RX on same X310:
python cw_doppler.py -r 1e6 --fft-size 1024 \
    --tx --tx-freq 5.915e9 --tx-gain 15 --tx-args "serial=33767A5" \
    --rx --rx-freq 5.915e9 --rx-gain 30 --rx-args "serial=33767A5" \
    -s 200e3 --avg-frames 20

# Software only (no hardware):
python cw_doppler.py -s 200e3
"""

import argparse
import sys
import time

from gnuradio import gr, analog, blocks, qtgui
from PyQt5 import Qt
import sip

try:
    from gnuradio.fft import window as grwindow
    WIN_BLACKMAN_HARRIS = grwindow.WIN_BLACKMAN_hARRIS
except (ImportError, AttributeError):
    try:
        from gnuradio.filter import firdes
        WIN_BLACKMAN_HARRIS = firdes.WIN_BLACKMAN_hARRIS
    except (ImportError, AttributeError):
        WIN_BLACKMAN_HARRIS = 5


# =============================================================================
# Flowgraph
# =============================================================================
class CWDoppler(gr.top_block, Qt.QWidget):

    def __init__(self, cw_offset, samp_rate, fft_size, update_time,
                 avg_frames,
                 tx_enabled, tx_freq, tx_gain, tx_args,
                 rx_enabled, rx_freq, rx_gain, rx_args, rx_file):

        gr.top_block.__init__(self, "CW Micro-Doppler Monitor")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("CW Micro-Doppler Monitor")

        layout = Qt.QVBoxLayout(self)

        fc = rx_freq if rx_enabled else (tx_freq if tx_enabled else 0.0)

        print(f"\n[cw_doppler] CW offset  : {cw_offset/1e3:.1f} kHz")
        print(f"[cw_doppler] Sample rate : {samp_rate/1e6:.3f} MHz")
        print(f"[cw_doppler] FFT size    : {fft_size}")
        print(f"[cw_doppler] Avg frames  : {avg_frames}")
        print(f"[cw_doppler] Carrier     : {fc/1e9:.6f} GHz\n")

        # ── CW tone source ────────────────────────────────────────────────────
        self.cw_src = analog.sig_source_c(
            samp_rate, analog.GR_COS_WAVE, cw_offset, 0.7, 0)

        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        # ── STEP 1: RX first -- claims the NI-RPC session on X310 ─────────────
        self.rx_src = None

        if rx_enabled:
            try:
                from gnuradio import uhd
                print("[rx] Opening RX first to claim UHD session...")
                self.rx_src = uhd.usrp_source(
                    rx_args,
                    uhd.stream_args(cpu_format="fc32", channels=[0]),
                )
                self.rx_src.set_samp_rate(samp_rate)
                self.rx_src.set_center_freq(rx_freq)
                self.rx_src.set_gain(rx_gain)
                self.rx_src.set_antenna("RX2")
                print(f"[rx] freq={rx_freq/1e9:.4f} GHz  gain={rx_gain} dB  "
                      f"args='{rx_args}'")
                print("[rx] Waiting 2 s for X300 init to settle...")
                time.sleep(2.0)
            except ImportError:
                print("[warn] UHD not available -- RX disabled.")
                rx_enabled = False
            except Exception as e:
                print(f"[error] RX init failed: {e}")
                rx_enabled = False
                self.rx_src = None

        # ── STEP 2: TX second -- reuses the existing RPC session ──────────────
        self.tx_sink = None

        if tx_enabled:
            try:
                from gnuradio import uhd
                print("[tx] Opening TX (reusing UHD session)...")
                self.tx_sink = uhd.usrp_sink(
                    tx_args,
                    uhd.stream_args(cpu_format="fc32", channels=[0]),
                )
                self.tx_sink.set_samp_rate(samp_rate)
                self.tx_sink.set_center_freq(tx_freq)
                self.tx_sink.set_gain(tx_gain)
                self.tx_sink.set_antenna("TX/RX")
                print(f"[tx] freq={tx_freq/1e9:.4f} GHz  gain={tx_gain} dB  "
                      f"args='{tx_args}'")
            except ImportError:
                print("[warn] UHD not available -- TX disabled.")
                tx_enabled = False
            except Exception as e:
                print(f"[error] TX init failed: {e}")
                tx_enabled = False
                self.tx_sink = None

        # =====================================================================
        # Panel 1: TX baseband spectrum (confirms tone is live)
        # =====================================================================
        tx_label = (f"TX Baseband  |  CW @ {cw_offset/1e3:.0f} kHz offset"
                    + (f"  |  RF {tx_freq/1e9:.4f} GHz" if tx_enabled else ""))
        self.tx_fft = qtgui.freq_sink_c(
            fft_size, WIN_BLACKMAN_HARRIS,
            0, samp_rate, tx_label)
        self.tx_fft.set_update_time(update_time)
        self.tx_fft.set_y_axis(-120, 0)
        self.tx_fft.enable_autoscale(False)
        self.tx_fft.enable_grid(True)
        layout.addWidget(sip.wrapinstance(self.tx_fft.qwidget(), Qt.QWidget))

        # =====================================================================
        # Panels 2 & 3: RX averaged spectrum + waterfall (only if RX active)
        # =====================================================================
        self.rx_fft  = None
        self.rx_wfall = None

        if rx_enabled and self.rx_src is not None:

            # -- Panel 2: averaged frequency spectrum -------------------------
            # set_fft_average(N) averages N consecutive FFT frames.
            # This is the core "video integration" step -- noise floor drops,
            # Doppler bumps become stable and visible.
            rx_spec_label = (
                f"RX Spectrum (avg {avg_frames} frames)  |  "
                f"CW @ centre  |  RF {rx_freq/1e9:.4f} GHz"
            )
            self.rx_fft = qtgui.freq_sink_c(
                fft_size, WIN_BLACKMAN_HARRIS,
                cw_offset,   # centre display on the CW tone
                samp_rate, rx_spec_label)
            self.rx_fft.set_update_time(update_time)
            self.rx_fft.set_y_axis(-120, 0)
            self.rx_fft.enable_autoscale(True)
            self.rx_fft.enable_grid(True)
            # ---- THIS IS THE KEY LINE: temporal averaging -------------------
            self.rx_fft.set_fft_average(avg_frames)
            # -----------------------------------------------------------------
            layout.addWidget(
                sip.wrapinstance(self.rx_fft.qwidget(), Qt.QWidget))

            # -- Panel 3: waterfall (scrolling spectrogram) -------------------
            # Shows Doppler energy as coloured trails over time.
            # Positive Doppler (target approaching) drifts RIGHT of centre.
            # Negative Doppler (target receding)   drifts LEFT  of centre.
            rx_wfall_label = (
                f"RX Waterfall (micro-Doppler)  |  "
                f"CW @ centre  |  RF {rx_freq/1e9:.4f} GHz"
            )
            self.rx_wfall = qtgui.waterfall_sink_c(
                fft_size, WIN_BLACKMAN_HARRIS,
                cw_offset,   # centre display on the CW tone
                samp_rate, rx_wfall_label)
            self.rx_wfall.set_update_time(update_time)
            self.rx_wfall.set_intensity_range(-120, 0)
            layout.addWidget(
                sip.wrapinstance(self.rx_wfall.qwidget(), Qt.QWidget))

        # =====================================================================
        # Connect TX path
        # =====================================================================
        if tx_enabled and self.tx_sink is not None:
            self.connect(self.cw_src, self.tx_sink)
            self.connect(self.cw_src, self.tx_fft)
        else:
            self.connect(self.cw_src, self.throttle)
            self.connect(self.throttle, self.tx_fft)

        # =====================================================================
        # Connect RX path
        # =====================================================================
        if rx_enabled and self.rx_src is not None:
            if self.rx_fft is not None:
                self.connect(self.rx_src, self.rx_fft)
            if self.rx_wfall is not None:
                self.connect(self.rx_src, self.rx_wfall)
            if rx_file is not None:
                self.file_sink = blocks.file_sink(
                    gr.sizeof_gr_complex, rx_file, False)
                self.file_sink.set_unbuffered(False)
                self.connect(self.rx_src, self.file_sink)
                print(f"[file] Saving RX IQ to '{rx_file}'")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="CW micro-Doppler monitor -- gr.top_block + Qt GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-s", "--cw-offset", type=float, default=200e3,
                   metavar="HZ",
                   help="CW tone offset from carrier (Hz)")
    p.add_argument("-r", "--rate", type=float, default=1e6,
                   metavar="HZ", help="Sample rate (Hz)")
    p.add_argument("--fft-size", type=int, default=1024,
                   metavar="N", help="FFT size")
    p.add_argument("--update-time", type=float, default=0.05,
                   metavar="S", help="Display refresh interval (s)")
    p.add_argument("--avg-frames", type=int, default=20,
                   metavar="N",
                   help="FFT frames to average (higher=smoother, "
                        "lower=faster response). Try 10 for gestures, "
                        "50 for slow vehicle motion.")

    p.add_argument("--tx", action="store_true", help="Enable hardware TX")
    p.add_argument("--tx-freq", type=float, default=5.915e9, metavar="HZ")
    p.add_argument("--tx-gain", type=float, default=15,      metavar="dB")
    p.add_argument("--tx-args", type=str,   default="",      metavar="ARGS")

    p.add_argument("--rx", action="store_true", help="Enable hardware RX")
    p.add_argument("--rx-freq", type=float, default=5.915e9, metavar="HZ")
    p.add_argument("--rx-gain", type=float, default=30,      metavar="dB")
    p.add_argument("--rx-args", type=str,   default="",      metavar="ARGS")
    p.add_argument("--rx-file", type=str,   default=None,    metavar="FILE",
                   help="Save RX IQ to binary file (complex64)")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    a    = parse_args()
    qapp = Qt.QApplication(sys.argv)

    tb = CWDoppler(
        cw_offset   = a.cw_offset,
        samp_rate   = a.rate,
        fft_size    = a.fft_size,
        update_time = a.update_time,
        avg_frames  = a.avg_frames,
        tx_enabled  = a.tx,
        tx_freq     = a.tx_freq,
        tx_gain     = a.tx_gain,
        tx_args     = a.tx_args,
        rx_enabled  = a.rx,
        rx_freq     = a.rx_freq,
        rx_gain     = a.rx_gain,
        rx_args     = a.rx_args,
        rx_file     = a.rx_file,
    )

    tb.show()
    tb.start()

    qapp.exec_()

    tb.stop()
    tb.wait()


if __name__ == "__main__":
    main()