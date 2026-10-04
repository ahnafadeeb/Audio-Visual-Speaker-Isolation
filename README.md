# Audio-Visual Speaker Isolation

Upload a video of two people talking, click a
face, hear only that person. Switching speakers is instant and does not
interrupt playback.

The full design rationale — including why the first three attempts at
suppressing bleed-through failed — is in `../ARCHITECTURE_V2.md`.

## Install

Python **3.12** specifically. `mediapipe==0.10.21` is the last release that
exposes `mp.solutions.face_mesh`, and it ships per-interpreter wheels only up
to cp312.

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install torch first, from the CUDA index. Plain `pip install torch` pulls the
CPU wheel from PyPI: everything imports, nothing errors, inference is ~30x
slower.

**Pick the CUDA index by GPU generation — `cu128` is not universal.**

| GPU | index | why |
|:--|:--|:--|
| RTX 30xx / 40xx / 50xx | `cu128` | native `sm_86` / `sm_89` / `sm_120` |
| GTX 10xx (Pascal, `sm_61`) | `cu126` | `cu128` dropped the Pascal tail |

Both indexes carry the same `torch 2.11.0` + `torchaudio 2.11.0` cp312 pair, so
only the CUDA build differs. Note that `cu118`/`cu121`/`cu124` do *not* have
2.11.0 at all (they stop at 2.7.1 / 2.5.1 / 2.6.0), so reaching for an older
CUDA to support an older card costs you the matched pair too — and a mismatched
torch/torchaudio raises `ImportError: DLL load failed` on Windows, which reads
like a CUDA fault and will cost you a day.

`torch.cuda.is_available()` does **not** tell you the wheel can run on your
card — it only proves a driver and runtime exist. A wheel missing your compute
capability imports fine, reports CUDA available, allocates memory, then dies at
the first real kernel launch with *"no kernel image is available for execution
on the device"*, minutes into a job, from inside SepFormer. Settle it before you
care:

```bash
.venv\Scripts\python.exe scripts\gpu_probe.py
```

That prints the wheel's arch list and then *launches* the kernel families
SepFormer actually uses (matmul, fp16, conv1d, SDPA), so a pass means the card
works rather than that the metadata looks right.

Then verify the machine before you need it to work:

```bash
run.bat --preflight
```

That checks the interpreter, numpy major version, CUDA with a real allocation,
the GPU arch list, ffmpeg, and — importantly — that the gate actually emits
bit-exact zeros. It also pre-downloads the ~400 MB of SepFormer weights so the
first job isn't waiting on conference wifi.

### VRAM

`chunk_s` is resolved per-machine at run time against **free** VRAM, so the same
checkout runs on a 4 GB card and an 8 GB one with nothing to edit before a demo.
Measure your own card with:

```bash
.venv\Scripts\python.exe scripts\bench_vram.py
```

Measured on a GTX 1050 Ti (4 GB): peak VRAM is **linear** in `chunk_s`
(`chunk_s^0.93`, ≈ `133 + 126 × chunk_s` MB), not quadratic — the quadratic
inter-segment attention term only dominates past ~16 s. A 10 s chunk peaks at
1385 MB and fits in 3.3 GB free with room to spare. That matters in the
non-obvious direction: **larger chunks are safer**, because every chunk boundary
is a permutation decision that can swap the speakers for the rest of the clip —
which sounds exactly like the bleed-through the gate exists to remove.

## Run

```bash
run.bat
```

Then open <http://127.0.0.1:8000> and drop in a video.

Bound to `127.0.0.1` on purpose. This is an unauthenticated
file-upload-and-transcode endpoint; on a shared network `0.0.0.0` would offer
it to the room.

### Check the UI without touching the models

```bash
python scripts/make_demo_job.py
```

Writes a synthetic job to `runs/demo` — two animated faces whose mouths are
driven by the same envelopes as their voices — with zero ML in the path. Start
the server and open <http://127.0.0.1:8000/?job=demo>. If the overlay, the
switch, and the drift readout behave here, any remaining fault is in the
models, not the frontend.

### Offline, no server

```bash
python -m app.pipeline input.mp4 --out runs/cli
python -m app.pipeline input.mp4 --out runs/cli --separator passthrough
```

## Before a live demo: adapt the model to your team

The released AV-TSE model does well on the team's recordings in a quiet room
and badly in a loud one. Fine-tuning its last layers on a calibration
recording of the same people, with noise mixed in, fixes most of that. Tested
on a take the adapter never saw (Whisper WER, face 0 / face 1):

| condition | released weights | adapted |
|:--|:--|:--|
| quiet | 3.5 / 5.4 % | 3.5 / 7.6 % |
| café noise, 15 dB SNR | 10.5 / 21.7 % | 3.5 / 5.4 % |
| café noise, 10 dB SNR | 37.2 / 34.8 % | 25.6 / 15.2 % |
| babble, 15 dB SNR | 37.2 / 25.0 % | 4.7 / 13.0 % |

It also leaks less of the other speaker (strict mode, ground-truth remix:
16.3 / 13.4 → 16.9 / 16.0 dB; in pauses −25.6 → −30.3 dB) and lets through
6–7 dB less room noise. An adapter only knows the people it was trained on,
and it makes strangers *worse*. So a job uses it only for faces that match
the adapter's own faces (SFace embedding), and every other face gets the
released weights. The diagnostics panel shows which faces used it.

**Record, in the room you will present in:**

1. **Calibration take, 1–2 min, quiet room.** The same people who will
   be in the demo. Faces fully visible, phone about 0.5–1 m away, landscape.
   Take turns and also talk over each other a little. Pseudo-targets come from
   this take, so the quieter the room, the better the adapter.
2. *(Optional)* **One solo take per person, ~60 s each.** These are the
   cleanest targets there are.
3. *(Optional)* **30 s of room tone**, meaning the venue with nobody
   speaking: fans, AC, crowd murmur. Pass it with `--room`.

**Train (15 min on an RTX 5060 in performance mode, 30 min power-capped;
nothing else on the GPU meanwhile):**

```bash
python -m app.adapt train team calib.mp4 --room room_tone.m4a
```

Sources can be video files, `runs/<job-id>` folders, or job ids already
processed in the web app. The new adapter becomes active immediately, and the
next upload uses it with no restart. Other commands:

```bash
python -m app.adapt list
python -m app.adapt use none
```

Babble noise for training comes from `adapters/noise/`. It must hold speech
from people *other* than the team.

**Recording the live clip.** Keep the phone within about 1 m of the
speakers. SNR drops 6 dB every time the distance doubles, and the jump from
15 dB to 10 dB SNR costs far more than anything the software can recover.
Keep both faces in frame and lit, and have speakers take turns where possible.

**Uploading straight from the phone.** Start the server with `HOST=0.0.0.0`
on a private hotspot. Then open `http://<laptop-ip>:8000` on the phone;
"Browse" offers the camera. Windows asks once whether Python may use the
network: allow it for private networks only. Anyone on that network can
upload, so don't do this on shared Wi-Fi.

```bat
set HOST=0.0.0.0
run.bat
```

## Two outputs, every run

| file | chain | use |
| --- | --- | --- |
| `stems_demo.wav` | A1 + A3 + A4 (hard-gated) | what the UI plays |
| `stems_raw.wav` | A1 + A3 (ungated) | SI-SDR / PESQ scoring |

They differ by one time-domain multiply, so writing both is nearly free — and
it means the objective score and the demo audio provably come from the same
run. Do not score `stems_demo.wav`: a hard gate zeroes frames the reference
still has signal in, and SI-SDR punishes that severely. The gate is a
*perceptual* choice, and it costs metrics to buy silence.

## How it works

**Silence (Objective A).** Four stages, in `app/dsp.py` and `app/separation.py`:

1. **A1** — `separate_batch()`, not `separate_file()`. The latter peak-normalises
   each source independently, which rescales a −25 dB residual to full scale.
   That was the ghost whisper: an amplitude bug, not a DSP shortfall.
2. **A3** — Wiener mask with exponent 2 applied to the *estimates*, floor
   `0.0`. A conventional −20 dB floor passes a fixed fraction of the interferer
   at all times, which *is* the whisper. Musical noise is controlled by
   cepstral smoothing of the log-gain instead.
3. **A4** — time-domain Schmitt gate with hysteresis, minimum on/off
   durations, 20 ms lookahead, and raised-cosine edges. Time-domain because
   ISTFT overlap-add smears a zeroed frame by `nfft - hop` samples at each
   edge; frequency-domain zeros do not survive resynthesis.

Mixture consistency projection is deliberately **not** implemented. `whamr16k`
is trained with reverberant input against anechoic targets, so `s1 + s2` does
not equal the mixture by design — projecting onto that constraint measurably
made leakage worse.

**Instant switching (Objective B).** One `AudioBufferSourceNode` playing one
multichannel buffer, fanned out by a `ChannelSplitterNode` into one `GainNode`
per speaker. One buffer and one source means the stems physically cannot drift
relative to each other. Switching is a 25 ms equal-power crossfade
(`setValueCurveAtTime`) on the audio thread — never a re-decode, never a seek.
The `<video>` element is muted and slaved to the audio clock, corrected via
`requestVideoFrameCallback` with a ±2% `playbackRate` authority and a 250 ms
hard-reseek threshold.

Level meters tap **post-gain**, so a muted speaker reads a flat zero. The
silence is visible, not merely inaudible.

**Coordinates.** `app/vision.py` maps FaceMesh landmarks from crop-normalised
space back to frame pixels before measuring anything. The prototype compared
lip-aperture thresholds against a ruler whose length changed with the crop —
which is why both earlier gates failed. Cosine similarity is scale-invariant,
so the *matcher* survived the bug and hid it.

## Layout

```
app/config.py      every tunable number
app/dsp.py         Wiener mask, Schmitt gate, leakage measurement
app/separation.py  SepFormer + chunking + permutation alignment; passthrough
app/vision.py      face tracking, lip aperture (the coordinate fix)
app/matching.py    stem <-> track assignment on envelope derivatives
app/media.py       ffmpeg, multichannel WAV
app/pipeline.py    orchestration, both outputs
app/adapt.py       speaker/room adapter training (python -m app.adapt)
app/jobs.py        job store, single-worker GPU admission gate
app/main.py        FastAPI, SSE progress
app/static/        index.html, app.js, style.css
```

## Keyboard

`Space` play/pause · `←`/`→` seek 5 s · `1`–`9` select speaker

## Notes

- `max_workers=1` in `app/jobs.py` is the GPU admission gate, not an
  oversight. Two concurrent jobs OOM an 8 GB card.
- Do not add `GZipMiddleware`. It buffers responses, so the SSE progress
  stream never flushes and the bar appears to hang.
- The gate constants in `GateConfig` were fitted on synthetic signals. Their
  relative ordering transfers; re-fit the absolute values against real
  SepFormer output on your clips.
