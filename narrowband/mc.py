#!/usr/bin/env python3
"""
Multi-Subcarrier Transceiver

Programmatically builds a GNU Radio flowgraph with N native C++
sig_source_c blocks summed via an Add block. Includes QT GUI FFT
sinks for real-time spectrum display of both TX and RX paths.

Usage examples:
    # Software only (no hardware)
    python3 mc.py -n 9

    # TX only
    python3 mc.py -n 10 -s 100e3 -r 20e6 --fft-size 1024 \
        --tx --tx-freq 5.915e9 --tx-gain 15 --tx-args "serial=33767A5"

    # TX + RX (same or different USRP)
    python3 mc.py -n 10 -s 100e3 -r 20e6 --fft-size 1024 \
        --tx --tx-freq 5.915e9 --tx-gain 15 --tx-args "serial=33767A5" \
        --rx --rx-freq 5.915e9 --rx-gain 30 --rx-args "serial=33767A5"

    # RX only (just view spectrum)
    python3 mc.py -n 1 --rx --rx-freq 5.915e9 --rx-gain 30 \
        --rx-args "serial=33767A5" --fft-size 1024

Subcarrier placement (symmetric outward from DC):
    n=1  ->  [0]
    n=2  ->  [-spacing, +spacing]
    n=3  ->  [-spacing, 0, +spacing]
    n=4  ->  [-2*spacing, -spacing, +spacing, +2*spacing]
    n=5  ->  [-2*spacing, -spacing, 0, +spacing, +2*spacing]
    ...
"""

import argparse
import sys
import numpy as np

from gnuradio import gr, analog, blocks, qtgui
from PyQt5 import Qt
import sip

# Window type constant varies across GNU Radio versions
try:
    from gnuradio.fft import window
    WIN_BLACKMAN_HARRIS = window.WIN_BLACKMAN_hARRIS
except (ImportError, AttributeError):
    try:
        from gnuradio.filter import firdes
        WIN_BLACKMAN_HARRIS = firdes.WIN_BLACKMAN_hARRIS
    except (ImportError, AttributeError):
        WIN_BLACKMAN_HARRIS = 5  # raw enum value as fallback


def compute_freqs(num_subcarriers, spacing, samp_rate):
    """Compute subcarrier frequencies, symmetric around DC, Nyquist-guarded."""
    n = num_subcarriers
    nyquist = samp_rate / 2.0

    freqs = []
    if n % 2 == 1:
        freqs.append(0.0)
        pairs_needed = (n - 1) // 2
    else:
        pairs_needed = n // 2

    for k in range(1, pairs_needed + 1):
        f = k * spacing
        if f < nyquist:
            freqs.append(f)
            freqs.append(-f)
        else:
            print(f"Warning: subcarrier at +/-{f/1e6:.3f} MHz exceeds Nyquist "
                  f"({nyquist/1e6:.3f} MHz), skipping remaining.")
            break

    return sorted(freqs)


class MulticarrierTRx(gr.top_block, Qt.QWidget):

    def __init__(self, num_subcarriers, spacing, samp_rate, amplitude,
                 fft_size,
                 tx_enabled, tx_freq, tx_gain, tx_args,
                 rx_enabled, rx_freq, rx_gain, rx_args, rx_file):
        gr.top_block.__init__(self, "Multi-Subcarrier Transceiver")
        Qt.QWidget.__init__(self)

        self.setWindowTitle("Multi-Subcarrier Transceiver")
        self.top_layout = Qt.QVBoxLayout(self)

        # ----------------------------------------------------------
        # Compute subcarrier frequencies
        # ----------------------------------------------------------
        freqs = compute_freqs(num_subcarriers, spacing, samp_rate)
        n_active = len(freqs)
        print(f"\nActive subcarriers: {n_active}")
        print(f"Spacing: {spacing/1e3:.1f} kHz")
        print(f"Sample rate: {samp_rate/1e6:.1f} MHz")
        if n_active > 0:
            print(f"Span: {freqs[0]/1e6:.3f} MHz to {freqs[-1]/1e6:.3f} MHz")
        print(f"Frequencies (MHz): {[f'{f/1e6:.3f}' for f in freqs]}\n")

        # Normalised amplitude per tone so sum doesn't clip
        if n_active > 0:
            tone_amp = amplitude / n_active
        else:
            tone_amp = 0

        # ----------------------------------------------------------
        # Build signal source blocks
        # ----------------------------------------------------------
        self.sources = []
        for i, freq in enumerate(freqs):
            src = analog.sig_source_c(
                samp_rate,
                analog.GR_COS_WAVE,
                freq,
                tone_amp,
                0
            )
            self.sources.append(src)

        # ----------------------------------------------------------
        # Adder block
        # ----------------------------------------------------------
        if n_active > 1:
            self.adder = blocks.add_cc(1)
        elif n_active == 1:
            self.adder = None
        else:
            self.null_src = analog.sig_source_c(
                samp_rate, analog.GR_CONST_WAVE, 0, 0, 0)
            self.adder = None

        # ----------------------------------------------------------
        # Throttle (limits CPU when no hardware sink/source)
        # ----------------------------------------------------------
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)

        # ----------------------------------------------------------
        # TX FFT Sink — shows the baseband TX spectrum
        # ----------------------------------------------------------
        tx_label = "TX Baseband Spectrum"
        if tx_enabled:
            tx_label += f"  (CF: {tx_freq/1e9:.4f} GHz, Gain: {tx_gain} dB)"

        self.tx_fft_sink = qtgui.freq_sink_c(
            fft_size,
            WIN_BLACKMAN_HARRIS,
            0,
            samp_rate,
            tx_label
        )
        self.tx_fft_sink.set_update_time(0.05)
        self.tx_fft_sink.set_y_axis(-120, 0)
        self.tx_fft_sink.set_y_label("Relative Gain", "dB")
        self.tx_fft_sink.enable_autoscale(False)
        self.tx_fft_sink.enable_grid(True)

        tx_fft_widget = sip.wrapinstance(self.tx_fft_sink.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(tx_fft_widget)

        # ----------------------------------------------------------
        # Optional hardware TX sink
        # ----------------------------------------------------------
        self.tx_sink = None
        if tx_enabled:
            try:
                from gnuradio import uhd
                self.tx_sink = uhd.usrp_sink(
                    tx_args,
                    uhd.stream_args(cpu_format="fc32", channels=[0])
                )
                self.tx_sink.set_samp_rate(samp_rate)
                self.tx_sink.set_center_freq(tx_freq)
                self.tx_sink.set_gain(tx_gain)
                print(f"TX enabled: freq={tx_freq/1e9:.4f} GHz, "
                      f"gain={tx_gain} dB, args=\"{tx_args}\"")
            except ImportError:
                print("Warning: UHD not available, TX disabled.")
                tx_enabled = False
                self.tx_sink = None

        # ----------------------------------------------------------
        # Optional hardware RX source + RX FFT Sink
        # ----------------------------------------------------------
        self.rx_source = None
        self.rx_fft_sink = None
        if rx_enabled:
            try:
                from gnuradio import uhd

                self.rx_source = uhd.usrp_source(
                    rx_args,
                    uhd.stream_args(cpu_format="fc32", channels=[0])
                )
                self.rx_source.set_samp_rate(samp_rate)
                self.rx_source.set_center_freq(rx_freq)
                self.rx_source.set_gain(rx_gain)
                print(f"RX enabled: freq={rx_freq/1e9:.4f} GHz, "
                      f"gain={rx_gain} dB, args=\"{rx_args}\"")

                # RX FFT Sink
                rx_label = (f"RX Spectrum  (CF: {rx_freq/1e9:.4f} GHz, "
                            f"Gain: {rx_gain} dB)")
                self.rx_fft_sink = qtgui.freq_sink_c(
                    fft_size,
                    WIN_BLACKMAN_HARRIS,
                    rx_freq,
                    samp_rate,
                    rx_label
                )
                self.rx_fft_sink.set_update_time(0.05)
                self.rx_fft_sink.set_y_axis(-120, 0)
                self.rx_fft_sink.set_y_label("Relative Gain", "dB")
                self.rx_fft_sink.enable_autoscale(False)
                self.rx_fft_sink.enable_grid(True)

                rx_fft_widget = sip.wrapinstance(
                    self.rx_fft_sink.qwidget(), Qt.QWidget)
                self.top_layout.addWidget(rx_fft_widget)

            except ImportError:
                print("Warning: UHD not available, RX disabled.")
                rx_enabled = False
                self.rx_source = None

        # ----------------------------------------------------------
        # Connect the TX flowgraph
        # ----------------------------------------------------------
        if n_active > 1:
            for i, src in enumerate(self.sources):
                self.connect((src, 0), (self.adder, i))
            signal_out = self.adder
        elif n_active == 1:
            signal_out = self.sources[0]
        else:
            signal_out = self.null_src

        if tx_enabled and self.tx_sink is not None:
            # Hardware TX provides backpressure — no throttle on TX path
            self.connect(signal_out, self.tx_sink)
            self.connect(signal_out, self.tx_fft_sink)
        else:
            # Software only — throttle the TX display path
            self.connect(signal_out, self.throttle)
            self.connect(self.throttle, self.tx_fft_sink)

        # ----------------------------------------------------------
        # Connect the RX flowgraph
        # ----------------------------------------------------------
        if rx_enabled and self.rx_source is not None and self.rx_fft_sink is not None:
            self.connect(self.rx_source, self.rx_fft_sink)

            # Optional file capture
            if rx_file is not None:
                self.rx_file_sink = blocks.file_sink(
                    gr.sizeof_gr_complex, rx_file, False)
                self.rx_file_sink.set_unbuffered(False)
                self.connect(self.rx_source, self.rx_file_sink)
                print(f"RX capture: saving to \"{rx_file}\"")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Subcarrier Transceiver using native GNU Radio blocks")

    parser.add_argument("--num-subcarriers", "-n", type=int, default=9,
                        help="Number of subcarriers (1-1800, default: 9)")
    parser.add_argument("--spacing", "-s", type=float, default=100e3,
                        help="Subcarrier spacing in Hz (default: 100e3)")
    parser.add_argument("--samp-rate", "-r", type=float, default=20e6,
                        help="Sample rate in Hz (default: 20e6)")
    parser.add_argument("--amplitude", "-a", type=float, default=1.0,
                        help="Total output amplitude (default: 1.0)")
    parser.add_argument("--fft-size", type=int, default=4096,
                        help="FFT size for spectrum display (default: 4096)")

    # Hardware TX options
    parser.add_argument("--tx", action="store_true",
                        help="Enable hardware transmit via UHD")
    parser.add_argument("--tx-freq", type=float, default=915e6,
                        help="TX center frequency in Hz (default: 915e6)")
    parser.add_argument("--tx-gain", type=float, default=30,
                        help="TX gain in dB (default: 30)")
    parser.add_argument("--tx-args", type=str, default="",
                        help="UHD device args (default: \"\")")

    # Hardware RX options
    parser.add_argument("--rx", action="store_true",
                        help="Enable hardware receive via UHD")
    parser.add_argument("--rx-freq", type=float, default=915e6,
                        help="RX center frequency in Hz (default: 915e6)")
    parser.add_argument("--rx-gain", type=float, default=30,
                        help="RX gain in dB (default: 30)")
    parser.add_argument("--rx-args", type=str, default="",
                        help="UHD device args (default: \"\")")
    parser.add_argument("--rx-file", type=str, default=None,
                        help="Save RX samples to file (complex64 binary, e.g. capture.cf32)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Clamp subcarriers
    num_sc = max(1, min(args.num_subcarriers, 1800))

    # Enforce minimum spacing
    spacing = max(args.spacing, 10e3)

    # QT application
    qapp = Qt.QApplication(sys.argv)

    tb = MulticarrierTRx(
        num_subcarriers=num_sc,
        spacing=spacing,
        samp_rate=args.samp_rate,
        amplitude=args.amplitude,
        fft_size=args.fft_size,
        tx_enabled=args.tx,
        tx_freq=args.tx_freq,
        tx_gain=args.tx_gain,
        tx_args=args.tx_args,
        rx_enabled=args.rx,
        rx_freq=args.rx_freq,
        rx_gain=args.rx_gain,
        rx_args=args.rx_args,
        rx_file=args.rx_file
    )

    tb.show()
    tb.start()

    qapp.exec_()

    tb.stop()
    tb.wait()


if __name__ == "__main__":
    main()