# C-V2X Coexistence CW MicroDoppler Tools

This repository contains Python-based radio-frequency (RF) sensing pipelines built on top of GNU Radio and UHD. It is designed to run on a USRP X310 at 5.9 GHz and provides tools for micro-Doppler extraction, continuous-wave (CW) hopping, and C-V2X sidelink coexistence.

## 📡 Features & Workflows

This repo is split into four main workflows:

1. **Pure-Python Simulator (`simulator/`)** Validate processing chains, tune parameters, and generate synthetic micro-Doppler butterflies without needing SDR hardware.
2. **Hopped-CW Pipeline (`multihop_and_cv2x_over_the_air_spectrogram_px/`)** Transmits a phase-continuous CW tone that hops across four frequencies. Includes scripts for GPSDO-aligned TX/RX capture, de-hopping, and interactive spectrogram viewing.
3. **C-V2X & CW Virtual Sensing (`overtheair/`)** Capture real 3GPP Rel-14 C-V2X sidelink transmissions, inject a narrow CW sensing tone between LTE data subcarriers, and replay the signal. Allows for simultaneous C-V2X data decoding and micro-Doppler extraction.
4. **Narrowband Comb (`narrowband_paper/`)** A multi-subcarrier CW transceiver producing a comb of *N* parallel tones. Great for visual, real-time classroom demos with a live QT GUI.

## 🛠 Prerequisites

### Hardware
* **USRP X310** with dual daughtercards (e.g., SBX). 
  * TX channel: 30.72 MSps (full 20 MHz C-V2X channel)
  * RX channel: 192 kSps (narrow-band CW return)
* **GPSDO:** Absolutely required for sample-aligned hopped-CW captures and for preserving C-V2X DFN scrambler timing during replays.

### Software
* **radioconda** (or equivalent environment)
* **GNU Radio** >= 3.9
* **UHD 4.x** with RFNoC (required for high-performance FPGA-DRAM replay)
* Python packages: `numpy`, `scipy`, `matplotlib`, `PyQt5`

## 🚀 Quick Start (Golden Path)

We highly recommend starting in the simulator to verify your environment and understand the math before burning USRP time.

**1. Run the Simulator**
```bash
cd multihop_and_cv2x_over_the_air_spectrogram_px/simulator
python test_dehop_pipeline.py
```
*(You should see a clear micro-Doppler "butterfly" plot representing a walking pedestrian.)*

**2. Interactive Parameter Tuning**
```bash
python cwhopping_explorer.py
```
*(Use the GUI sliders to adjust STFT windows, overlap, and hop frequencies to find your ideal configuration.)*

**3. Hardware Capture (Hopped-CW)**
Once configured, capture real data with your USRP:
```bash
cd ..
python cwhop_tx.py --config cwhopping_config_POST.json \
    --center-freq 5.9e9 --tx-gain 20 --rx-gain 30 \
    --duration 20 --gpsdo \
    --args "serial=YOUR_USRP_SERIAL,master_clock_rate=184.32e6" \
    --output my_capture
```

**4. View Results**
```bash
python cwhop_viewer.py --input my_capture_rx.cf32
```

## 📖 Documentation

For a deep dive into every script, sidecar JSON formats, and troubleshooting tips, please refer to the comprehensive **Tools User Manual** (HTML) included in this repository. 
```
