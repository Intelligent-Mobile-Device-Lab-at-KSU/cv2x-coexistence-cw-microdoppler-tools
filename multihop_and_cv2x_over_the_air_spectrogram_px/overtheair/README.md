# overtheair/ — C-V2X Over-The-Air Capture, Replay, and Virtual Sensing

Frozen snapshot of the over-the-air C-V2X sidelink workflow. Captures a real
3GPP Rel-14 C-V2X sidelink signal (20 MHz, 5.9 GHz), optionally injects a CW
virtual-sensing tone between data subcarriers, replays it GPSDO-synchronised so
the PHY scrambler stays valid, and extracts micro-Doppler from a narrow-band
RX of the CW tone.

**Do not edit these files.** Copy them out to experiment.

---

## The DFN problem (why UTC/GPSDO sync matters)

C-V2X sidelink PHY scrambling is seeded by the **Direct Frame Number (DFN)**,
which cycles 0–1023 every **10.24 s** (1024 frames × 10 subframes × 1 ms).
DFN is derived from UTC — every GPSDO-locked radio on Earth agrees on the
current DFN at every instant.

If you capture an IQ file and later replay it at an arbitrary time, the DFN
at the first replayed sample will not match the DFN encoded in the scrambler
of the captured symbols → commercial C-V2X receivers fail to decode.

**Fix.** When capturing:
1. Lock TX/RX to GPSDO (UTC)
2. Record the DFN/subframe coordinate of the first sample into a JSON sidecar
3. Snap duration to an integer multiple of 10.24 s

When replaying:
1. Read the sidecar
2. Wait until the current UTC time matches the captured DFN/subframe
3. Start TX on that sample boundary

`cv2x_capture_replay.py` and `cv2x_rfnoc_replay.py` both handle this when `--gpsdo` is set.

---

## Files

| File | Role |
|------|------|
| `cv2x_capture_replay.py` | **Main workflow:** GNU Radio flowgraph, capture or replay, GPSDO/DFN-aligned |
| `cv2x_rfnoc_replay.py` | High-performance replay via X310 FPGA DRAM (RFNoC Replay Block) — no underflows |
| `cv2x_cw_inject.py` | Injects a CW virtual-sensing tone at `(k+0.5)·15 kHz` between captured C-V2X subcarriers |
| `cv2x_cw_txrx.py` | Simultaneous TX replay of CW-injected file + narrow-band RX of the CW tone (dual daughtercard) |
| `cv2x_microdoppler_extract.py` | Offline: extract micro-Doppler from the narrow-band RX capture |

---

## Hardware assumptions

- **USRP X310** with two SBX daughtercards (or similar dual-channel setup)
  - Radio#0 / subdev A:0 → TX at 30.72 Msps (full 20 MHz C-V2X channel)
  - Radio#1 / subdev B:0 → RX at ~192 kSps (narrow band around the CW tone)
- **GPSDO** providing 10 MHz + PPS + GNSS time
- **UHD 4.x with RFNoC** for the Replay Block path
- Commercial C-V2X radio (the OBU/RSU under test) on the same RF channel

---

## Workflow A — Plain capture and replay (no CW injection)

Useful for basic record/playback of a real C-V2X transmission.

### 1. Capture

```bash
python cv2x_capture_replay.py --capture --gpsdo --headless \
    --rx-freq 5.915e9 --rx-gain 30 \
    --rx-args "serial=33767A5,master_clock_rate=184.32e6" \
    --capture-file cv2x_iq.cf32 --duration 60 -r 30.72e6
```

**Outputs:**
- `cv2x_iq.cf32` — raw IQ (complex64, 30.72 Msps). Duration auto-snapped up to the next 10.24 s multiple.
- `cv2x_iq.json` — sidecar with the DFN/subframe of the first sample, UTC timestamp, sample rate, center freq

### 2. Replay (GNU Radio path)

```bash
python cv2x_capture_replay.py --replay --gpsdo --headless \
    --tx-freq 5.915e9 --tx-gain 15 \
    --tx-args "serial=33767A5,master_clock_rate=184.32e6" \
    --replay-file cv2x_iq.cf32 --loop -r 30.72e6
```

Sidecar is auto-read from `cv2x_iq.json`. TX starts on the exact UTC moment where the captured DFN/subframe matches current time — so the scrambler is valid from the very first sample.

### 2b. Replay (RFNoC / FPGA DRAM path — recommended for sustained replay)

Avoids host-side underflows that can break 1 ms subframe timing.

```bash
# Upload once (stays in DRAM until power cycle)
python cv2x_rfnoc_replay.py --upload \
    --file cv2x_iq.cf32 -r 30.72e6 \
    --args "serial=33767A5,master_clock_rate=184.32e6"

# Play (can be repeated without re-uploading)
python cv2x_rfnoc_replay.py --play --gpsdo --loop \
    --file cv2x_iq.cf32 -r 30.72e6 \
    --tx-freq 5.915e9 --tx-gain 15 \
    --args "serial=33767A5,master_clock_rate=184.32e6"

# Or combine both in one call:
python cv2x_rfnoc_replay.py --upload --play --gpsdo --loop \
    --file cv2x_iq.cf32 -r 30.72e6 \
    --tx-freq 5.915e9 --tx-gain 15 \
    --args "serial=33767A5,master_clock_rate=184.32e6"
```

---

## Workflow B — Full virtual-sensing pipeline (capture → inject CW → TX+RX → extract)

This is the complete micro-Doppler-via-C-V2X chain: piggyback a narrow CW
sensing tone inside the spectrum of a real C-V2X transmission, replay it so
that the OBU/RSU still decodes C-V2X data, and simultaneously extract
micro-Doppler from the CW return.

### 1. Capture a real C-V2X transmission

Same as Workflow A, Step 1. Produces `cv2x_iq.cf32` + sidecar.

### 2. Inject a CW tone between subcarriers

```bash
python cv2x_cw_inject.py --input cv2x_iq.cf32 --output cv2x_iq_cw.cf32 --plot
```

**What it does.** Detects active subframes in the capture, runs an FFT to find
occupied subcarriers, then inserts a CW tone at a midpoint offset
`(k + 0.5) × 15 kHz`. Because this falls exactly between two LTE subcarriers,
the commercial radio's FFT gridlines ignore it and still decodes the C-V2X
data cleanly.

**Outputs:**
- `cv2x_iq_cw.cf32` — modified IQ with CW injected
- `cv2x_iq_cw.json` — updated sidecar including `f_cw_hz` (the injected tone frequency)

### 3. Simultaneous TX replay + narrow-band RX

```bash
python cv2x_cw_txrx.py \
    --tx-file cv2x_iq_cw.cf32 \
    --rx-file cv2x_rx_doppler.cf32 \
    --args "serial=33767A5,master_clock_rate=184.32e6" \
    --tx-freq 5.915e9 --tx-gain 15 \
    --rx-gain 30 --rx-rate 192e3 \
    --duration 60 --gpsdo --loop
```

**What it does.**
- TX daughtercard replays `cv2x_iq_cw.cf32` at 30.72 Msps (full C-V2X channel), GPSDO/DFN-aligned
- RX daughtercard tunes to `tx_freq + f_cw_hz + rx_offset` and captures a narrow ~192 kSps band around the CW tone
- `rx_offset` (default 50 kHz) keeps the CW tone off DC to avoid LO leakage contamination

**Outputs:**
- `cv2x_rx_doppler.cf32` — narrow-band RX of the CW return (complex64, 192 kSps)

### 4. Extract micro-Doppler offline

```bash
python cv2x_microdoppler_extract.py \
    --input cv2x_rx_doppler.cf32 \
    --sidecar cv2x_iq_cw.json \
    --fft-size 19200 --overlap 0.9 \
    --doppler-range 500 --plot --save
```

**What it does.**
1. Coarse FFT to find the exact received CW peak (correcting for LO offset between TX/RX daughtercards)
2. Mix the CW tone to DC
3. Calibrate out the static LO offset
4. STFT spectrogram (Doppler vs time)
5. Phase tracking at the CW frequency for fine-grained Doppler
6. Subframe gating — only show windows where C-V2X was actually transmitting

**Outputs:**
- PNG spectrogram
- NPZ with STFT arrays and phase-Doppler time series
- JSON summary

---

## Quick inspection utility

For sanity-checking any of the .cf32 files produced below (raw capture,
CW-injected, or narrow-band RX), use the generic spectrogram viewer in the
parent folder:

```bash
# Raw C-V2X capture (wideband)
python ../iq_spectrogram.py cv2x_iq.cf32 -r 30.72e6

# CW-injected file with sidecar overlay
python ../iq_spectrogram.py cv2x_iq_cw.cf32 -r 30.72e6 --sidecar cv2x_iq_cw.json

# Narrow-band RX capture of the CW return
python ../iq_spectrogram.py cv2x_rx_doppler.cf32 -r 192e3 --fft 64 --step 16
```

---

## Troubleshooting

**Commercial radio won't decode after replay.**
- Verify `--gpsdo` was set on both capture and replay
- Verify capture duration is an integer multiple of 10.24 s (sidecar will confirm)
- Check GPSDO lock with `uhd_usrp_probe --args "type=x300"` before running
- On the RFNoC path, check for underflows (`U` in stdout) — if seen, upload to DRAM isn't complete or sample rate is misconfigured

**CW tone not visible in the RX.**
- Confirm `f_cw_hz` in `cv2x_iq_cw.json` matches what you'd expect
- Verify `rx_freq` was set to `tx_freq + f_cw_hz + rx_offset` (the script does this automatically from the sidecar)
- Check TX/RX gain balance — CW can be buried in C-V2X data if TX gain is too low or RX gain too high

**Micro-Doppler spectrogram looks like noise.**
- First run `cv2x_microdoppler_extract.py` with `--plot` and no target — you should see a clean DC line
- If no DC line: LO offset calibration failed, or CW tone detection missed
- Check `--fft-size` matches your expected Doppler resolution (19200 at 192 kSps = 10 Hz/bin)

**GPSDO query fails at script start.**
- Scripts look for `query_gpsdo_sensors` in standard radioconda paths. If your install uses a different layout, edit `_QUERY_GPSDO_PATHS` at the top of `cv2x_capture_replay.py` / `cv2x_rfnoc_replay.py`.

---

## Relationship to the hopped-CW pipeline (../)

Both this folder and the hopped-CW pipeline in the parent directory achieve
**CW-based micro-Doppler sensing**, but from opposite ends:

| | Hopped CW (parent folder) | OTA CV2X + CW inject (this folder) |
|--|--|--|
| Signal source | Synthesised frequency-hopping CW | Real captured C-V2X sidelink |
| CW placement | Sweeps across 4 sensing bins | Static, between two data subcarriers |
| Data co-existence | No data | Full commercial C-V2X data |
| Sync | GPSDO to align TX/RX | GPSDO + DFN to preserve scrambler |
| Use case | Clean sensing channel characterisation | Realistic coexistence study |

The offline processing differs too: `../cwhop_rx.py` de-hops across 4
frequencies, while `cv2x_microdoppler_extract.py` operates on a single static
tone.
