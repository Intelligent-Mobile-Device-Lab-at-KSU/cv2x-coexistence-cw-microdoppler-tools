# microDopplerHopping_WORKING

Frozen working snapshot of the hopped-CW micro-Doppler pipeline.
**Do not edit these files.** Make copies before experimenting.

Captured result: rotating panel micro-Doppler at 5.9 GHz, 20 MHz BW,
USRP X310, 70 ms STFT window / 80% overlap / 4x zero-pad / hann.

---

## Files

| File | Role |
|------|------|
| `cwhop_tx.py` | GNU Radio flowgraph: generates phase-continuous hopped CW and captures RX simultaneously |
| `cwhop_rx.py` | Batch processor: de-hop → cascade FIR decimate (30.72 MHz → 10 kHz) → STFT → PNG/NPZ |
| `cwhop_viewer.py` | Interactive spectrogram viewer with live sliders (window, overlap, zero-pad, dyn range, notch, stitch) |
| `tx_sanity.py` | Hopped-CW sanity check: raw spectrogram of a .cf32 file with NO de-hopping, cyan sidecar overlay should sit on the tone |
| `iq_spectrogram.py` | Generic .cf32/.bin IQ spectrogram viewer (not hop-specific). Useful for inspecting any capture — raw C-V2X, CW-injected, narrow-band RX, etc. |
| `cv2x_microdoppler_cw2.py` | **Real-time live monitor** — single static CW tone, 3-panel display (TX baseband / RX averaged / RX waterfall). Use during an experiment to see micro-Doppler live, not for offline analysis. |
| `cwhopping_config_POST.json` | Config used for the working result (70 ms window, hann, 80% overlap, ±500 Hz Doppler range) |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `simulator/` | Pure-Python simulation of the hopped-CW pipeline (no hardware). Validate changes, tune parameters, regression-test. See `simulator/README.md`. |
| `overtheair/` | C-V2X capture/replay workflow. GPSDO/DFN-aligned record & playback of real 5.9 GHz sidelink, CW virtual-sensing injection, micro-Doppler extraction. See `overtheair/README.md`. |

---

## Workflow

### Step 0 — Simulate first (no hardware needed)

Before any hardware run, validate the processing chain with the simulator scripts in
`simulator/` (frozen copy of `new_update/python_version/`):

**Generators**
- `generate_hopped_cw.py` — synthesises a .cf32 hopped-CW TX file matching the config
- `generate_reference_cw.py` — plain (non-hopped) CW reference for baseline comparisons
- `generate_scfdma_data.py` — SC-FDMA payload generator (for co-channel interference tests)
- `make_taper_envelope.py` — dwell-transition taper envelope builder

**Processing**
- `dehop_and_stft.py` — reference de-hop + STFT implementation (simpler than `cwhop_rx.py`, good for debugging)
- `compute_quality_metric.py` — quantitative quality scoring (SNR, SFDR, phase continuity)

**Tests (run these to validate changes)**
- `test_dehop_pipeline.py` — full de-hop + decimation + STFT on a synthetic signal; checks
  for phase continuity and clean DC output
- `test_phase_continuity.py` — verifies TX phase accumulates correctly across dwells
- `test_taper_envelope.py` — verifies dwell transition envelopes are smooth
- `test_pedestrian_model.py` — injects a synthetic walking micro-Doppler signature and
  verifies the spectrogram shows the expected butterfly pattern

**Exploration**
- `cwhopping_explorer.py` — interactive parameter sweep for tuning dwell/hop settings
- `walking_micro_doppler.py` — pedestrian micro-Doppler model reference

**Config**
- `cwhopping_config.json` — original baseline config (stft_window_ms=10, ±1826 Hz).
  The tuned version `../cwhopping_config_POST.json` overrides to 70 ms / hann / 80% / ±500 Hz.

Run the simulator pipeline to confirm a clean butterfly before touching hardware:

```bash
cd simulator
python test_dehop_pipeline.py       # end-to-end validation (hopped vs reference)
python cwhopping_explorer.py        # interactive tuning GUI
```

See `simulator/README.md` for a full per-script tutorial. The `generate_*.py`
and `dehop_and_stft.py` files are **library modules** (imported, not run
directly) — the runnable scripts above wire them together.

---

### Step 1 — TX/RX capture (`cwhop_tx.py`)

Transmits a frequency-hopping CW tone and captures the RX simultaneously.
Both TX and RX are scheduled to the same GPS time via GPSDO.

```bash
python cwhop_tx.py \
    --config cwhopping_config_POST.json \
    --output my_capture \
    --duration 20 \
    --center-freq 5.9e9 \
    --tx-gain 15 --rx-gain 30 \
    --args "serial=33767A5,master_clock_rate=184.32e6"
```

**Outputs:**
- `my_capture.cf32` — TX baseband (complex64, 30.72 Msps)
- `my_capture_rx.cf32` — RX capture (complex64, 30.72 Msps)
- `my_capture_schedule.json` — TX hop schedule sidecar (dwell times, frequencies, sample indices)

---

### Step 2 — Sanity check raw TX (`tx_sanity.py` or `iq_spectrogram.py`)

For hopped-CW files with a sidecar, use `tx_sanity.py` (cyan schedule overlay).
For any generic .cf32 file (C-V2X captures, CW-injected files, narrow-band RX), use
`iq_spectrogram.py`:

```bash
python iq_spectrogram.py my_capture.cf32 -r 30.72e6
python iq_spectrogram.py overtheair_capture/cv2x_iq_cw.cf32 --sidecar overtheair_capture/cv2x_iq_cw.json
```

---

### Step 2a — Hopped-CW sanity check (`tx_sanity.py`)

Before processing, verify the TX file looks correct: the CW tone should be
visible jumping between the 4 hop frequencies (7.5 / 15 / 22.5 / 30 kHz).
The cyan sidecar overlay should sit exactly on top of the tone.

```bash
python tx_sanity.py --input my_capture.cf32 --sidecar my_capture_schedule.json
```

If the cyan overlay drifts off the tone, the sidecar schedule or TX timing is wrong.
Fix before proceeding.

---

### Step 3a — Interactive viewer (`cwhop_viewer.py`)

Best for exploratory analysis. Builds a decimated cache on first run (~30 s for a 20 s
capture), then lets you tune STFT parameters live with sliders.

```bash
python cwhop_viewer.py --input my_capture_rx.cf32
```

Force a cache rebuild if the code has changed or the cache looks wrong:

```bash
python cwhop_viewer.py --input my_capture_rx.cf32 --reprocess
```

**Recommended starting settings for micro-Doppler:**
- Window: 70 ms
- Overlap: 80%
- Zero-pad: 4x
- Window type: hann
- Dyn. range: 40 dB
- Doppler ±: 500 Hz
- Stitch: ON

---

### Step 3b — Batch export (`cwhop_rx.py`)

For scripted runs or paper figures. Uses parameters from the config file directly.
`cwhopping_config_POST.json` has the tuned settings (70 ms / hann / 80% / ±500 Hz).

```bash
python cwhop_rx.py \
    --input my_capture_rx.cf32 \
    --config cwhopping_config_POST.json \
    --plot --save \
    --dynamic-range 40
```

**Outputs:**
- `my_capture_rx_results_plot.png` — spectrogram figure
- `my_capture_rx_results_spectrogram.npz` — `(f, t, S_dB)` arrays
- `my_capture_rx_results_summary.json` — metadata

---

---

## Live monitoring during an experiment (`cv2x_microdoppler_cw2.py`)

For real-time visual feedback while moving a target in front of the antenna —
useful for positioning, antenna aiming, or quick qualitative demos. This is a
completely separate workflow from the offline hopped-CW pipeline: single
static CW tone, no hopping, no de-hopping, no config file.

```bash
python cv2x_microdoppler_cw2.py -r 1e6 --fft-size 1024 \
    --tx --tx-freq 5.915e9 --tx-gain 15 --tx-args "serial=33767A5" \
    --rx --rx-freq 5.915e9 --rx-gain 30 --rx-args "serial=33767A5" \
    -s 200e3 --avg-frames 20
```

Shows 3 panels:
1. TX baseband spectrum (confirms CW tone is transmitting)
2. RX averaged spectrum (Doppler bumps either side of CW peak)
3. RX waterfall (scrolling spectrogram, Doppler arcs left/right of center)

Key tuning: `--avg-frames 10` for fast gestures, `--avg-frames 50` for slow
vehicle motion. Pure software mode (no SDR): `python cv2x_microdoppler_cw2.py -s 200e3`.

---

## Processing chain overview

```
my_capture_rx.cf32   (complex64, 30.72 Msps)
        |
        v
  De-hop: multiply by exp(-j*2*pi*delta_f*t_global) per dwell
  (phase-continuous; t is GLOBAL sample time, never reset per block)
        |
        v
  Cascade FIR decimate 3072x: [4,4,4,4,4,3] stages
  Anti-alias cutoffs: 3.84 MHz → 960 kHz → 240 kHz → 60 kHz → 15 kHz → 5 kHz
  Block boundaries: zero-padded margins (NOT raw signal — avoids FIR ringing)
        |
        v
  10 kSps baseband  (all dwells at DC, ~80 KB per original second)
        |
        v
  STFT spectrogram  (70 ms hann window, 80% overlap, 4x zero-pad)
  Resolution: ~2.4 Hz/bin  |  Range: ±500 Hz
        |
        v
  Micro-Doppler spectrogram  (Doppler Hz vs time s)
```

---

## Key design decisions (why the pipeline works)

- **Phase-continuous TX**: `cwhop_tx.py` accumulates global phase across dwells with no reset,
  so the received tone has no phase jumps at hop transitions.
- **Global-time de-hopping**: de-hopping uses `t = n / Fs` with global sample index `n`,
  matching the TX NCO exactly.
- **Zero-pad margins (not raw signal)**: block-boundary FIR margins are zero-padded rather than
  reading the adjacent raw (undehopped) signal. The raw signal at 7.5–30 kHz passes through
  the first four decimation stages unattenuated (stage 4 cutoff = 60 kHz >> 30 kHz), which
  caused severe ringing at every block boundary when raw margins were used.
- **TX/RX cache separation**: `cwhop_viewer.py` uses `splitext` only for cache naming, so
  `my_capture.cf32` and `my_capture_rx.cf32` never share a cache file.
- **GPSDO timing**: TX and RX are both scheduled to the same GPS-locked time in `cwhop_tx.py`,
  ensuring sample-accurate alignment.
