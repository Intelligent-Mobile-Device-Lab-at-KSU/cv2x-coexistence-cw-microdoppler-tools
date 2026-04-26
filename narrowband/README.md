# narrowband/

Multi-subcarrier CW transceiver for the narrowband paper study.
Separate project from the hopped-CW work in `../uncoordinatedpaper/`.

---

## Files

| File | Role |
|------|------|
| `mc.py` | GNU Radio flowgraph: N parallel CW subcarriers, TX/RX/simulate modes, live QT FFT display |
| `capture.bin` | Sample RX capture for playback |
| `capture2.bin` | Sample RX capture for playback |

---

## What `mc.py` does

Programmatically builds a GNU Radio flowgraph with **N native C++ `sig_source_c`
blocks summed via an Add block** — creating a comb of CW tones symmetric around
DC. Includes QT GUI FFT sinks for real-time spectrum display of both the TX
baseband and the RX capture.

**Subcarrier placement** (symmetric outward from DC, Nyquist-guarded):

| n | Placement |
|---|-----------|
| 1 | `[0]` |
| 2 | `[-spacing, +spacing]` |
| 3 | `[-spacing, 0, +spacing]` |
| 4 | `[-2·spacing, -spacing, +spacing, +2·spacing]` |
| 5 | `[-2·spacing, -spacing, 0, +spacing, +2·spacing]` |

Per-tone amplitude is `amplitude / N` so the sum never clips.

---

## Three operating modes

### 1. Software only (no hardware)

CPU-throttled signal generation with live TX spectrum display. Good for
confirming the subcarrier layout before touching an SDR.

```bash
python mc.py -n 9
```

### 2. TX only

Transmit through a USRP, watch the TX baseband spectrum live.

```bash
python mc.py -n 10 -s 100e3 -r 20e6 --fft-size 1024 \
    --tx --tx-freq 5.915e9 --tx-gain 15 --tx-args "serial=33767A5"
```

### 3. TX + RX (loopback or separate USRPs)

```bash
python mc.py -n 10 -s 100e3 -r 20e6 --fft-size 1024 \
    --tx --tx-freq 5.915e9 --tx-gain 15 --tx-args "serial=33767A5" \
    --rx --rx-freq 5.915e9 --rx-gain 30 --rx-args "serial=33767A5"
```

### 4. RX only (spectrum viewer)

```bash
python mc.py -n 1 --rx --rx-freq 5.915e9 --rx-gain 30 \
    --rx-args "serial=33767A5" --fft-size 1024
```

Optionally add `--rx-file capture.bin` to save the RX stream to disk for later
replay.

---

## CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `-n / --num-subcarriers` | 9 | Number of CW tones |
| `-s / --spacing` | 100 kHz | Spacing between tones |
| `-r / --samp-rate` | 20 MHz | Baseband sample rate |
| `-a / --amplitude` | 1.0 | Total signal amplitude (split across tones) |
| `--fft-size` | 4096 | QT FFT display size |
| `--tx` | off | Enable USRP TX |
| `--tx-freq` | 915 MHz | TX center freq |
| `--tx-gain` | 30 | TX gain (dB) |
| `--tx-args` | "" | UHD device args |
| `--rx` | off | Enable USRP RX |
| `--rx-freq` | 915 MHz | RX center freq |
| `--rx-gain` | 30 | RX gain (dB) |
| `--rx-args` | "" | UHD device args |
| `--rx-file` | none | Save RX stream to .bin |

---

## Relationship to the hopped-CW paper

Both projects use CW for RF sensing, but different signal designs:

| | narrowband_paper (`mc.py`) | uncoordinatedpaper (hopped-CW) |
|--|--|--|
| Signal | N simultaneous parallel tones | 1 tone hopping between 4 frequencies |
| Occupancy | N bins active always | 1 bin active per dwell |
| Processing | Live QT FFT display | Offline de-hop + STFT |
| Hardware | Single-channel USRP | USRP X310, GPSDO-locked |
| Data co-existence | Standalone | Designed to coexist with SC-FDMA C-V2X |
| Paper | Narrowband sensing | Uncoordinated ISAC / C-V2X |

If you want to look at a moving target in real time with a simple tonal comb,
use `mc.py`. If you want offline micro-Doppler extraction with no wasted
spectrum, use the hopped-CW pipeline.

---

## Dependencies

- GNU Radio 3.9+ (radioconda works)
- PyQt5
- UHD 4.x (only if `--tx` or `--rx` enabled)
- numpy
