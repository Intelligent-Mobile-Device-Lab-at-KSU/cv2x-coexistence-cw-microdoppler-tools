# simulator/ — Hopped-CW Micro-Doppler Pipeline Validation

Pure-Python simulation of the hopped-CW micro-Doppler pipeline, with no
hardware required. Use this folder to:

- Validate the processing chain before touching hardware
- Develop and tune parameters (dwell, taper, STFT window, etc.) interactively
- Regression-test changes to the pipeline
- Compare hopped-CW micro-Doppler quality against a continuous-CW ground truth

---

## Dependencies

```
numpy, scipy, matplotlib (TkAgg backend for interactive scripts)
```

All scripts assume they're run **from inside this `simulator/` directory**
(the modules import each other by bare name, not by package path).

```bash
cd microDopplerHopping_WORKING/simulator
```

---

## Two kinds of files in this folder

This folder mixes **runnable scripts** and **library modules**. Know which is which.

### Runnable (call these directly)

| Script | Type | Purpose |
|---|---|---|
| `cwhopping_explorer.py` | Interactive GUI | Tabbed slider UI — tune every parameter live (dwell, taper, SC-FDMA, STFT, pedestrian) |
| `test_dehop_pipeline.py` | Test / demo | End-to-end: generate hopped CW → de-hop → STFT → plot side-by-side with reference |
| `test_pedestrian_model.py` | Test / demo | Generate and plot the walking-pedestrian micro-Doppler butterfly |
| `test_phase_continuity.py` | Test | Verify TX global phase accumulates correctly across dwells (no phase jumps) |
| `test_taper_envelope.py` | Test | Verify dwell transition envelope shapes |

### Library modules (imported, not run)

| Module | What it provides |
|---|---|
| `generate_hopped_cw.py` | `HoppedCWParams`, `generate_hopped_cw()` — frequency-hopping CW synthesis |
| `generate_reference_cw.py` | `RefCWParams`, `generate_reference_cw()` — continuous-CW ground truth |
| `generate_scfdma_data.py` | `SCFDMAParams`, `generate_scfdma_interference()` — SC-FDMA coexistence model |
| `make_taper_envelope.py` | `make_taper_envelope()` — per-dwell amplitude ramp |
| `walking_micro_doppler.py` | `PedestrianParams`, `walking_micro_doppler()` — 5-scatterer walker model |
| `dehop_and_stft.py` | `STFTParams`, `dehop_and_stft()` — reference de-hop + STFT implementation |
| `compute_quality_metric.py` | `compute_quality_metric()` — correlation, sidelobe, MSE vs reference |

None of these library modules have a CLI — import them in your own script or use one of the runnable scripts above.

---

## Suggested learning path

Run in this order to understand the pipeline bottom-up:

1. **`test_taper_envelope.py`** — see what one dwell looks like in amplitude
2. **`test_phase_continuity.py`** — see that TX phase is glitch-free across hops
3. **`test_pedestrian_model.py`** — see the walking butterfly we're trying to recover
4. **`test_dehop_pipeline.py`** — see the full chain: hopped generation → de-hop → STFT
5. **`cwhopping_explorer.py`** — tune parameters interactively and watch quality change

---

## Script-by-script usage

### 1. `test_taper_envelope.py`

**What it does.** Plots the dwell amplitude envelope for several taper shapes
(`raised_cosine`, `hann`, `blackman`, `linear`, `none`) so you can see how
`taper_pct` and `min_amp` affect the ramp.

```bash
python test_taper_envelope.py
```

**Expected output.** A matplotlib window with several envelope curves overlaid.
Ramp regions should be smooth, flat top at 1.0, reaching `min_amp` at the
guard edges. No CLI args — edit the hardcoded parameters at the bottom of the
file to try different shapes.

---

### 2. `test_phase_continuity.py`

**What it does.** Generates a short hopped-CW signal, then verifies that the
instantaneous phase is continuous across every dwell boundary. Proves the NCO
model in `generate_hopped_cw` is implemented correctly (phase accumulator
never resets).

```bash
python test_phase_continuity.py
```

**Expected output.** Console: "PASS — max phase discontinuity = X.XX rad"
(should be near 0 for all hop boundaries). Optional plot of unwrapped phase
vs time — it should be a piecewise-linear staircase with no vertical jumps.

**How to tell it failed.** A large phase jump at a dwell boundary or a PASS
→ FAIL flip in the console output.

---

### 3. `test_pedestrian_model.py`

**What it does.** Generates a pure-CW signal modulated by the
`walking_micro_doppler` 5-scatterer model (torso + 2 arms + 2 legs) and plots
the spectrogram. This is what a walking human looks like to a perfect
continuous-CW sensor — the ground truth you want the hopped pipeline to
reproduce.

```bash
python test_pedestrian_model.py
```

**Expected output.** Spectrogram showing the classic micro-Doppler butterfly:
- Torso trace near `v_bulk` (bulk Doppler, steady band)
- Arm sinusoids at `arm_f_md` peak, modulating at `arm_f_rot` Hz
- Leg sinusoids at ~1.5× arm Doppler, anti-phase to the arms

Adjust `PedestrianParams` inside the script to vary walking speed, RCS mix,
carrier frequency.

---

### 4. `test_dehop_pipeline.py`  ← **most important test**

**What it does.** The end-to-end validation:

1. Generate a hopped-CW signal with a pedestrian micro-Doppler modulation
2. Generate a matching continuous-CW reference
3. De-hop the hopped signal
4. Compute STFT on both
5. Plot side-by-side and compute quality metrics (correlation, sidelobe, MSE)

```bash
python test_dehop_pipeline.py
```

**Expected output.**
- Two-panel figure: reference spectrogram (left) vs de-hopped spectrogram (right)
- Console: correlation ≈ 0.9+, sidelobe ≈ −30 dB or lower, MSE low
- Both spectrograms should show the same butterfly

**This is the script to run after any change to `generate_hopped_cw`,
`dehop_and_stft`, or `make_taper_envelope`.** If the quality metrics drop or
the spectrograms diverge, your change broke something.

**Parameters to tweak.** Edit the `HoppedCWParams` / `STFTParams` /
`PedestrianParams` at the top of `main()` to try different dwell lengths,
taper settings, STFT windows, SNR, etc.

---

### 5. `cwhopping_explorer.py`  ← **interactive tuning**

**What it does.** Full tabbed GUI with every parameter exposed as a slider:

- **Tab 1 — Hopping CW:** dwell_ms, hop_freqs, taper_shape, taper_pct, min_amp, transition_mode, gap_ms/jitter
- **Tab 2 — SC-FDMA Data:** enabled, data_activity, num_subch_max, data_power_dB, spectral_isolation_dB, colocated_tx, cw_inside_alloc
- **Tab 3 — Rx / Sim:** pedestrian params (v_bulk, arm/leg Doppler & rate, RCS mix), STFT (window_ms, window_type, overlap_pct, zero_pad), duration, SNR

Move a slider → signal regenerates → de-hop + STFT runs → spectrograms and
quality score update live.

```bash
python cwhopping_explorer.py
```

**Use for.** Finding good parameter combinations before committing them to
`cwhopping_config_POST.json` for hardware runs. Answering questions like:
- "What's the shortest dwell I can use before the spectrogram degrades?"
- "How much SC-FDMA interference can the sensor tolerate?"
- "Does a longer STFT window help at slow walking speeds?"

---

## Common workflows

### A. Validate a config change before a hardware run

1. Edit your config parameters (dwell, taper, STFT window) in `cwhopping_explorer.py` sliders until the quality score is acceptable
2. Mirror those settings into `../cwhopping_config_POST.json`
3. Run `test_dehop_pipeline.py` as a final automated check
4. Only then run `../cwhop_tx.py` on the X310

### B. Debug a de-hop artifact seen in a hardware capture

1. Reproduce the capture conditions in `cwhopping_explorer.py` (same dwell/taper/hop freqs)
2. If the artifact shows in simulation → it's a signal-design problem, fix in the config
3. If the artifact does NOT show in simulation → it's a hardware/capture issue (TX/RX timing, gain, LO leakage, GPSDO drift, etc.)

### C. Regression-test after editing a library module

After changing `generate_hopped_cw.py`, `dehop_and_stft.py`, or
`make_taper_envelope.py`:

```bash
python test_phase_continuity.py   # sanity
python test_dehop_pipeline.py     # end-to-end quality
```

Both should PASS / produce a clean butterfly. If either regresses, revert.

### D. Benchmark quality for a paper table

`compute_quality_metric.compute_quality_metric()` returns a `QualityMetrics`
dataclass with `correlation`, `peak_sidelobe`, `dynamic_range`, `mse_dB`.
Call it from your own script in a sweep over dwell_ms / taper_pct / etc.

---

## File outputs

Most scripts are interactive (show matplotlib windows) and do NOT write files
to disk by default. If you want to save PNGs, add
`plt.savefig('out.png', dpi=150)` before `plt.show()` inside the script.

---

## Relationship to the hardware scripts

The library modules here (`dehop_and_stft.py`, in particular) are the
**reference implementation**. The hardware script `../cwhop_rx.py` is a
memory-efficient, decimating, block-processing version of the same algorithm,
tuned for 30.72 Msps .cf32 files that are too large to hold in RAM.

If the simulator shows a clean spectrogram but `cwhop_rx.py` does not, the
bug is in the block-processing / decimation layer, not in the core de-hop math.
