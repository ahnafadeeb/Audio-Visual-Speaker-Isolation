# Audio-Visual Speaker Isolation — v2 Architecture

**IEEE SP Cup 2026** · rebuild of the Colab prototype into a local interactive app
Status: plan · Target: Windows 11, RTX 4050/4060 laptop (6–8 GB VRAM), Python 3.12

---

## 0. Executive summary

The bleed-through is **not primarily a DSP failure**. It is caused by three
compounding defects in the existing pipeline, two of which are amplitude bugs that
make an inaudible residual loud. Fixing those is a ~20-line change and must happen
before any gate is tuned.

The rebuild therefore has two independent tracks:

| Track | Purpose | Risk |
|---|---|---|
| **Fix + harden the existing chain** | Working clean demo fast | Low |
| **Swap in audio-visual target-speaker extraction** | Best-in-class quality | Medium |

Both feed the same UI and the same artefact format, so the separation stage is a
**swappable interface**. That is the single most important structural decision in
this document: it lets us ship a demo early and upgrade quality without touching
the frontend, and it satisfies the requirement to emit an ungated stem for
SI-SDR/PESQ scoring alongside the hard-muted demo stem.

---

## 1. Root cause: why you hear ghosts

### Bug 1 — `separate_file()` peak-normalises every source independently

*Verified by reading `speechbrain/inference/separation.py` on the `develop` branch.*

`separate_file()` ends with:

```python
est_sources = est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
return est_sources
```

`separate_batch()` contains **no normalisation** — it returns raw decoder output.

Consequence: if channel 0 holds the target at full scale and channel 1 holds a
−25 dB residual, this line rescales **both to peak 1.0**. The residual is boosted
by ~25 dB into clear audibility. A genuinely silent channel becomes `0/0 → NaN`.

This explains the symptom precisely, including *"worst during pauses"* — during a
pause the target's own peak is small, so its normalisation gain is large.

### Bug 2 — Cell 4 normalises a *second* time

`full_pipeline_code_dump.md:410-412`:

```python
peak = np.max(np.abs(src)) + 1e-8
src = src / peak
```

Applied per source, in a loop. Even if Bug 1 were fixed, this re-introduces it.
**Both sources are forced to equal peak level regardless of their true relative
energy.** The `+ 1e-8` guard prevents the NaN but not the boost.

### Bug 3 — the lip signal is measured on a moving ruler

`full_pipeline_code_dump.md:1068-1071` computes lip aperture as the distance
between landmarks 13 and 14 in **crop-normalised** coordinates:

```python
dist = np.linalg.norm(
    np.array([landmarks.landmark[13].x, landmarks.landmark[13].y]) -
    np.array([landmarks.landmark[14].x, landmarks.landmark[14].y]))
```

MediaPipe returns coordinates normalised to the **crop** it was given, not the
frame. Since each face gets its own bbox crop, and the crop size varies per face
and per frame, the resulting value scales with the crop box rather than with the
face. Two speakers at different distances produce lip signals on different
scales; the same speaker's signal rescales whenever their bbox changes size.

Cosine similarity is scale-invariant, so **matching survived this** — which is
why the Hungarian assignment worked and masked the bug. But every *threshold*
(both failed gates) was comparing against a ruler that changes length. That alone
plausibly explains why the absolute-position and velocity gates behaved
erratically.

Cell 3's earlier version multiplied by `frame_w`/`frame_h` and did not have this
bug; the Cell 8 rewrite reintroduced it.

### Bug 4 (architectural) — SepFormer cannot output silence

*Verified from `hyperparams.yaml` and `speechbrain/lobes/models/dual_path.py`.*

- `num_spks: 2` is fixed — the model **always** emits two streams.
- The mask activation is `nn.ReLU()`, **unbounded and not softmax-normalised**
  across speakers. Masks do not sum to 1; nothing forces a zero output.
- SI-SNR training loss is **scale-invariant** and degenerate against an all-zero
  target, so PIT training applies *no gradient pressure toward silence*.

Therefore during a monologue, channel 2 necessarily contains *something* — a
scaled copy of the target, a dereverberation tail, or breath/onset fragments.
WHAMR! is fully-overlapped 2-speaker reverberant training data, so any long
single-speaker stretch is out-of-domain.

**Conclusion: no amount of gating makes a 2-output blind model emit silence. The
silence must be imposed downstream, in the time domain.**

---

## 2. Objective A — absolute silence

Silence is produced by a **three-stage chain**, in this order. Order matters: A3
must run in the time domain, after A2's ISTFT.

### Stage A1 — use `separate_batch()`, apply one global gain

```python
with torch.no_grad():
    est = model.separate_batch(mix)      # [1, T, 2]  — NO normalisation
```

Never call `separate_file()`. Do your own resampling (`torchaudio.functional.resample`
or `soxr`) and apply a **single global gain** at export — or better, `pyloudnorm`
to a fixed LUFS target — so relative source levels are preserved.

Delete the per-source `src / peak` loop.

### Stage A2 — ~~mixture-consistency projection~~ DO NOT USE (verified harmful here)

Mixture consistency (Wisdom et al., ICASSP 2019, arXiv:1811.08521) is the closed-form
projection onto `{Σ sₘ = x}`:

```python
ests = ests + (mix - ests.sum(0)) / ests.shape[0]     # DO NOT USE with whamr16k
```

It is a genuinely good technique and I had it in this chain. **It is wrong for this
model.** I checked SpeechBrain's WHAMR recipe data preparation directly
(`recipes/WHAMandWHAMR/prepare_data.py`, `create_wham_whamr_csv`, develop branch):

```python
mix_both = ("mix_both_reverb/" if task == "separation" else "mix_single_reverb/")
if dereverberate and (set_type != "tr"):
    s1 = "s1_reverb/";   s2 = "s2_reverb/"
else:
    s1 = "s1_anechoic/"; s2 = "s2_anechoic/"     # training set takes this branch
```

The input is `mix_both_reverb` — reverberant **and** noisy — while the training
targets are `s1_anechoic` / `s2_anechoic`. The model is trained to separate,
denoise, **and dereverberate** in one shot. Therefore:

> **`s1 + s2 ≠ mixture` by design.** The difference is exactly the environmental
> noise and the reverb tails the model was trained to throw away.

Forcing consistency would take that discarded noise-plus-reverb residual, split it
in half, and **add it back into both channels** — including the one you are trying
to drive to silence. It would re-inject the precise garbage the model removed, and
it would do so worst during pauses, where the target contributes least to the
residual. That is the ghost whisper, re-created by a technique meant to help.

The general lesson, worth keeping: mixture consistency is valid only when the
targets sum to the input. It holds for anechoic `sep_clean` tasks (wsj0-2mix); it
does **not** hold for WHAMR, WHAM, or any denoise/dereverb-in-the-loop separator.
Check the recipe before applying it to a new checkpoint.

*(Numbering below is kept as A3/A4 so the stage names stay stable in code and
notes; there are three active stages.)*

### Stage A3 — Wiener TF mask, p=2, applied to the *estimates*, floor 0.0

```python
P0, P1 = np.abs(S0)**2, np.abs(S1)**2
M0 = P0 / (P0 + P1 + 1e-12)
Y0 = M0 * S0          # <-- S0, the ESTIMATE. Not the mixture.
```

STFT via `scipy.signal.ShortTimeFFT`, `nfft=512`, `hop=128`, Hann — verified
COLA/NOLA true and round-trip invertible to 1e-6.

Two counter-intuitive findings here, both measured, both contradicting the
"obvious" choice:

- **Mask the estimate, not the mixture.** Mixture-domain masking is 12–17 dB
  *worse* on leakage and ~3 dB worse on distortion, because it discards
  SepFormer's work and degenerates into a crude binary separator driven by the
  same leaky magnitudes. (p=2: estimate −35.8 dB leak / 8.91 dB SI-SDR vs mixture
  −18.4 dB / 5.94 dB.)
- **Never floor the mask.** The conventional −20 dB floor collapses leakage
  suppression from −35.8 dB to −1.6 dB. A floor passes a fixed fraction of the
  interferer *at all times* — that is exactly the ghost whisper. Use floor `0.0`
  and control musical noise with cepstral smoothing of the log-gain instead.

p=2 is the measured knee (p=1: −19.4 dB, p=2: −35.8 dB, p=3: −41.1 dB but 0.6 dB
more distortion and a harder, musical-noise-prone mask).

### Stage A4 — time-domain Schmitt-trigger gate on the target's *own* energy

**This is the stage that produces true digital zeros.** Exact zeros are impossible
in the STFT domain: ISTFT overlap-add smears any zeroed frame block by
`nfft − hop = 384` samples = 24 ms at *each* edge (measured: zeroing 21 frames
gave a 2305-sample zero run, not the naive 2688). So the gate must multiply
samples *after* ISTFT.

```
ldb  = 10*log10(frame_mean_square + 1e-12)
ref  = percentile(ldb, 95)
open  when (ldb - ref) > -30 dB   and run >= min_off (30 ms)
close when (ldb - ref) < -40 dB   and run >= min_on  (60 ms)
ramp: 10 ms raised cosine, applied at sample rate
```

Measured: **−46.8 dB leak, 99.4 % of pause samples exactly 0.0, 8.60 dB active
SI-SDR** (vs 8.91 dB ungated). The gate buys 11 dB and true silence for 0.3 dB of
distortion.

**Gate on the target's own energy — never on dominance.** Measured dominance on
target-*active* frames has mean +3.5 dB but 10th percentile **−4.5 dB**: the
interferer is legitimately louder during 10 % of the target's own speech. Every
dominance-driven Schmitt configuration tested cut 40–70 % of genuine target
frames. Likewise a 1:20 downward expander reached −196 dB leak but **−4.65 dB**
SI-SDR: it reached numerical zero by deleting the speaker.

### What is *not* worth doing

- **Spectral subtraction.** 48 % of TF bins go negative and need half-wave
  rectification (the classic musical-noise generator), and reused phase is wrong
  by a mean 7.7°. Two speakers in one room are not uncorrelated additive noise.
- **Chaining a denoiser.** DeepFilterNet / resemble-enhance / sepformer-dns model
  *one speech source plus noise*. A second voice is classified as speech and
  **preserved**. The existence of a separate "personalized DeepFilterNet2" research
  line trained on target+interferer data is direct evidence stock DFN does not do
  speaker suppression. These clean noise; they leave the ghost untouched.
- **A mask floor** — see A3.

---

## 3. The upgrade path: audio-visual target-speaker extraction

The chain in §2 treats the symptom. The *principled* fix is to stop doing blind
separation and post-hoc matching at all, and instead condition the network on the
target's lip video so it emits **one** stream trained to suppress everything else.

This collapses three of your stages — blind separation, energy-envelope
computation, and Hungarian assignment — into a single model call. The lip track
*is* the selection signal.

### Verified availability (I checked these directly, not from memory)

| Fact | Status |
|---|---|
| `alibabasglab/AV_MossFormer2_TSE_16K` on HF | **exists**, `gated: false`, `private: false` |
| Licence | **apache-2.0** (card data + tags) |
| Checkpoint `last_best_checkpoint.pt` | **734,561,014 bytes** (~700 MiB, ≈183 M fp32 params) |
| `masknet_numspks` | **1** — single output, no source selection needed |
| pip package `clearvoice` | **0.1.2**, Apache-2.0, uploaded 2025-07-11, `requires_python >=3.8` |
| Repo `modelscope/ClearerVoice-Studio` | Apache-2.0, 4,395 stars, pushed 2025-08-14, default branch `main` |
| Smaller sibling `log_VoxCeleb2_lip_tfgridnet_2spk` | **exists**, ungated, apache-2.0, 160,433,102 B (~40 M params) |

Benchmarks (authors' own): AV-MossFormer2 **14.6 dB SI-SDRi** on VoxCeleb2 2-mix,
15.5 dB on LRS2 2-mix. The TFGridNet sibling gets 13.7 dB at **4.6× smaller** —
a strong trade, but it is *not* wired into the pip inference path, so it costs
real integration work.

### Known blockers — verified in source

**1. Hardcoded `.cuda()` breaks all CPU inference.** In
`clearvoice/clearvoice/models/av_mossformer2_tse/av_mossformer2.py`,
`overlap_and_add()` contains verbatim:

```python
frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
```

The comment claims device-agnostic handling; the code forces CUDA. Patch to
`.to(signal.device)`. This also means it cannot gracefully degrade to CPU if VRAM
runs out mid-run. On a 4050/4060 we will use GPU anyway, but the patch is needed
for a reliable fallback.

**2. 25 fps is hardcoded, not derived.** `crop_video()` writes
`cv2.VideoWriter(..., 25, (224,224))` and computes `audioStart = track['frame'][0]/25`.
`decode.py` uses `window_v = 25 * decode_window`. **Feed anything other than
25 fps and audio/video desync silently.** Force `fps=25` in the ffmpeg pre-pass.

**3. `numpy<2.0` pin.** `clearvoice` 0.1.2 requires `numpy>=1.24.3,<2.0`, plus
exact pins `librosa==0.10.2.post1`, `opencv-python==4.10.0.84`, `soundfile==0.12.1`,
`scenedetect==0.6.6`. This is the single biggest dependency-conflict risk in the
project and is why the AV-TSE stack gets **its own virtualenv** (§5).

**4. Duplicate checkpoints.** The HF repo holds `last_best_checkpoint.pt` *and*
`last_best_checkpoint_old.pt`, both 734 MB (2.2 GB total storage). Use
`huggingface_hub.hf_hub_download` for the single file — never `git clone`.

**5. SI-SDR still does not produce silence.** The training loss is scale-invariant,
so 14–16 dB SI-SDRi means the interferer is *attenuated*, not eliminated. **The
Stage A4 gate is still required.** What changes is that gating becomes *easy*:
the decision is made on a single high-SNR stream instead of a leaky one, which
removes the resting-mouth failure mode entirely.

### Licence note

Code and weights are Apache-2.0. LRS2-derived weights descend from BBC/Oxford VGG
data under a research-only agreement — Apache-2.0 on the weights file does not
launder the training-data restriction. For an IEEE competition this is almost
certainly fine; prefer the **VoxCeleb2**-trained checkpoints if commercial use is
ever contemplated. The generic `AV_MossFormer2_TSE_16K` card names no corpora at
all, so its provenance is unconfirmed.

### Bonus: it replaces MediaPipe

`clearvoice/clearvoice/utils/video_process.py` already ships
S3FD detect → `scene_detect` → `track_shot` → `crop_video`, producing **per-face
tracks with per-frame bounding boxes**. That is exactly the data structure the
click-a-face UI needs, and it needs only OpenCV — no MediaPipe, no protobuf, no
TensorFlow conflict. The `{'track': ..., 'proc_track': dets}` structure feeds
hit-testing directly.

If we go this route, the entire MediaPipe/protobuf dependency mess from Cell 1
disappears.

---

## 4. Objective B — real-time interactive UI

### Architecture: precompute stems, switch with Web Audio

The heavy inference **must not run at click time**. A 183 M-param model will not
hit real-time factor < 1 on a laptop, and even the SepFormer path is seconds per
minute of audio. The answer is to make the separation stage **precompute** K stems
(one per speaker) into WAVs, and let the frontend switch between already-decoded
buffers in ~20 ms.

```
backend (offline):  video → separation → K mono 16 kHz stems + bbox JSON track
frontend (online):  one <video>  (muted, or original audio)  +  K WebAudio stems
                    click face → crossfade gain between stems  →  instant switch
```

### One audio clock: multichannel WAV + ChannelSplitterNode

The critical invariant is **exactly one audio clock**. Do *not* run N independent
`AudioBufferSourceNode`s and fight drift between them — and above all do not use
N `<audio>` elements, which introduce N independent decoders each with its own
clock.

Recommended: the backend emits **one interleaved multichannel WAV**
(ch0 = speaker A, ch1 = speaker B) plus a **video-only MP4**. The frontend decodes
the WAV once into a single `AudioBufferSourceNode` and fans it out with a
`ChannelSplitterNode`:

```js
const buf = await ctx.decodeAudioData(await (await fetch(stemsUrl)).arrayBuffer());
const n   = buf.numberOfChannels;          // read it, never assume
const src = ctx.createBufferSource();
src.buffer = buf;
const splitter = ctx.createChannelSplitter(n);
src.connect(splitter);
const gains = Array.from({length: n}, (_, i) => {
  const g = ctx.createGain();
  g.gain.value = (i === active) ? 1 : 0;
  splitter.connect(g, i, 0);               // connect(dest, outputIndex, inputIndex)
  g.connect(master);
  return g;
});
const t0 = ctx.currentTime + 0.10;          // 100 ms scheduling cushion
src.start(t0);
video.muted = true;
video.play();
```

Because there is one buffer, the stems **cannot drift relative to each other**.
Only video-vs-audio needs correction.

Build the context as `new AudioContext({ sampleRate: 16000 })` — `decodeAudioData`
resamples to the context rate, so a default 48 kHz context would upsample every
16 kHz stem 3×, tripling memory and altering the samples relative to what the DSP
wrote.

> **Deliver the stems losslessly.** The entire §2 chain exists to produce exact
> time-domain zeros. Encoding those stems as AAC/Opus adds coding noise and
> pre-echo *precisely in the silent regions*, resurrecting the ghost whisper in the
> last mile. Use PCM WAV. This is the reason the stems ship separately from the
> video rather than muxed as AAC.

### Drift correction (video is the slave)

```js
const DEADBAND = 0.020;   // 20 ms — below this, do nothing
const MAX_RATE = 0.02;    // ±2% playbackRate authority
const HARD_SEEK = 0.250;  // 250 ms — give up and reseek
const KP = 0.5;

function onFrame(now, meta) {
  const ts = ctx.getOutputTimestamp();
  const audioPos = (ts.contextTime ?? ctx.currentTime) - t0;
  const drift = meta.mediaTime - audioPos;      // >0 => video ahead
  if (Math.abs(drift) > HARD_SEEK) {
    video.currentTime = audioPos; video.playbackRate = 1;
  } else if (Math.abs(drift) > DEADBAND) {
    video.playbackRate = 1 + Math.max(-MAX_RATE, Math.min(MAX_RATE, -KP * drift));
  } else {
    video.playbackRate = 1;
  }
  video.requestVideoFrameCallback(onFrame);
}
video.requestVideoFrameCallback(onFrame);
```

`requestVideoFrameCallback` is Baseline (Chrome 83, Safari 15.4, Firefox 132) and
its `metadata.mediaTime` is the on-screen frame's timestamp on the `currentTime`
timeline — far more precise than polling `video.currentTime`. Prefer
`ctx.getOutputTimestamp().contextTime`, which reports the frame *actually being
played*, since `ctx.currentTime` runs ahead by `baseLatency + outputLatency`.

**Never nudge audio `playbackRate`** — `AudioBufferSourceNode.playbackRate`
resamples with no pitch correction, so it audibly shifts pitch. Video rate changes
are silent because the video is muted.

rVFC does not fire in a hidden tab, so add a `visibilitychange` handler that does
one unconditional hard reseek on return rather than letting a 2%-capped corrector
crawl back across a multi-second gap.

*Alternative that removes the drift loop entirely:* mux the stems as channels of a
single **lossless** audio track inside the video (`ffmpeg -filter_complex
"[1:a][2:a]join=inputs=2:channel_layout=stereo[a]" -c:a flac`) and use
`createMediaElementSource` + splitter. One demuxer, one clock, sync exact by
construction. Costs container/codec-support work and, at N ≥ 3, risks browsers
downmixing a media element to stereo — safe at N=2. Worth a 5-minute smoke test if
you'd rather do container work than run a correction loop.

### Switching without a click

```js
const XF = 0.025, N = 64;                      // 25 ms, 64 control points
const up = new Float32Array(N), down = new Float32Array(N);
for (let i = 0; i < N; i++) {
  const x = i / (N - 1);
  up[i]   = Math.sin(x * Math.PI / 2);         // 0 -> 1
  down[i] = Math.cos(x * Math.PI / 2);         // 1 -> 0
}                                              // up² + down² === 1 exactly

function selectSpeaker(k) {
  const t = ctx.currentTime + 0.005;
  gains.forEach((g, i) => {
    g.gain.cancelScheduledValues(t);
    g.gain.setValueAtTime(g.gain.value, t);    // pin current value first
    g.gain.setValueCurveAtTime(i === k ? up : down, t, XF);
  });
  active = k;
}
```

`setValueCurveAtTime` runs the whole shape on the **audio thread** with
sample-accurate timing, immune to main-thread jank; a JS-driven fade stutters under
GC or layout. After the curve the param is guaranteed to hold `values[N-1]`, so the
faded-out gain rests at `cos(π/2) ≈ 6.1e-17` — −324 dB. Append
`g.gain.setValueAtTime(0, t + XF)` if you want bit-exact zero.

Three traps, all avoided by the code above:

- **Never `exponentialRampToValueAtTime`** — it cannot ramp to or from 0, which is
  fatal when the goal is silence.
- **Never write `.value` once automation is scheduled** — direct writes are
  ignored once the event list is non-empty.
- **Always `cancelScheduledValues` + `setValueAtTime` first** — overlapping
  `setValueCurveAtTime` events on one param throw, which a fast double-click on two
  faces would otherwise trigger mid-playback.

### The face overlay

Send the **full per-frame** bbox array, not keyframes. For a 3-minute 25 fps clip
with 2 faces that is ~9,000 objects — 200–350 KB raw, 60–90 KB gzipped, a single
cached fetch. Interpolating 2 fps keyframes costs you exactly the frames where
faces move fastest, and buys nothing measurable.

```json
{ "fps": 25.0, "w": 1280, "h": 720,
  "tracks": [ {"id": 0, "label": "Speaker A",
               "boxes": [[0.31,0.22,0.14,0.19], null, ...] } ] }
```

Normalise boxes to `[0,1]` at export so the frontend never needs the source
resolution to hit-test, and use `null` for frames where the track is absent —
`null` is a real state (occlusion, out of shot), and interpolating across it
invents a face that was never there.

- Frontend: a transparent `<canvas>` sized to the video's `clientWidth/Height`,
  redrawn in the same `requestVideoFrameCallback` loop that runs drift correction.
  Index with `Math.round(meta.mediaTime * fps)`; `mediaTime` is the frame actually
  on screen, so the box cannot lag the face.
- Hit-test in normalised space: convert the click to `(u, v)` via
  `getBoundingClientRect()`, then test against the raw box. No pixel maths, no
  resize handler, no dependence on CSS layout.
- Also render the tracks as real `<button>`s below the video. The canvas is a
  convenience; the buttons are the accessible, keyboard-reachable control, and they
  still work if a face is off-screen when the presenter wants to switch.

### Progress reporting during processing

The job takes tens of seconds; the page needs to show stage and percentage.
Server-Sent Events, not WebSockets — the traffic is one-way, it is plain HTTP, and
`EventSource` reconnects on its own.

FastAPI 0.135.0+ ships this natively:

```python
from fastapi.sse import EventSourceResponse   # no sse-starlette dependency needed
```

On older FastAPI, `pip install sse-starlette` and import from there; the API is the
same. One hard constraint: **SSE breaks under `GZipMiddleware`** — the middleware
buffers to compress and the stream never flushes. If you add gzip for the bbox
JSON, exclude the SSE route.

### Autoplay policy

`AudioContext` may start `suspended`. The upload → process → play flow provides a
gesture naturally; call `await ctx.resume()` **inside** that click handler, then
start playback. Because the video element is muted it never trips the policy
itself. Do not construct the `AudioContext` at page load and assume it runs —
check `ctx.state` before `src.start()`.

Two behaviours to know for a live demo: `requestAnimationFrame`/`rVFC` do not fire
in a hidden tab (so the drift corrector freezes — hence the `visibilitychange`
reseek above), while `AudioContext` in Chrome is *not* auto-suspended and keeps
playing. If the presenter tabs away and back, audio continues and video resyncs on
return.

---

## 5. Objective C — the local Windows backend

The failure mode to design against is not "the laptop is too slow". A 4060 with
8 GB will run this comfortably. The failure modes are **OOM from unchunked
attention**, **a blocked event loop that makes the UI look hung**, and **an
environment that silently breaks on a dependency pin**. All three are avoidable by
construction.

### 5.1 Process shape

```
uvicorn (1 worker, asyncio loop)          ← HTTP, SSE, static files. Never touches torch.
   └── ThreadPoolExecutor(max_workers=1)  ← the single inference lane
          └── ModelRegistry (warm singleton, loaded at startup)
```

Three rules, each preventing a specific observed pathology:

- **Warm-load models in the lifespan handler, not per request.** SepFormer
  construction + checkpoint fetch is seconds; doing it per job triples perceived
  latency and re-runs HF cache validation every time.
- **Run inference in a thread, not on the loop.** `torch` releases the GIL in its
  kernels, so a worker thread is sufficient and avoids Windows' spawn-based IPC
  (no `fork`: a process pool would re-import your module and re-load the model in
  the child). Wrap with `await asyncio.to_thread(...)` or an explicit executor.
- **`max_workers=1` is load-bearing.** It is the GPU admission gate. Two concurrent
  8 GB jobs OOM; serialising them is the entire concurrency policy you need for a
  single-user demo app.

```python
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

@asynccontextmanager
async def lifespan(app):
    app.state.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    app.state.models = ModelRegistry()      # lazy per-model, eager for the default
    app.state.models.warm("sepformer")
    yield
    app.state.pool.shutdown(wait=False, cancel_futures=True)
    app.state.models.release()

app = FastAPI(lifespan=lifespan)
```

Guard the entrypoint with `if __name__ == "__main__":` and call
`uvicorn.run("app.main:app", ...)` by import string. On Windows, `--reload` and any
multiprocessing spawn a fresh interpreter that re-imports the module; without the
guard you get a second model load, or a recursive spawn loop. Run **without
`--reload`** for the demo — the reloader doubles VRAM during the overlap window.

### 5.2 The job model

Processing is tens of seconds, so it cannot be one blocking HTTP request. Three
endpoints and a filesystem artefact store:

```
POST /api/jobs            -> {job_id}          multipart video upload, returns immediately
GET  /api/jobs/{id}/events -> SSE stream       {stage, pct, message} until "done"|"error"
GET  /api/jobs/{id}/artifact/{name}            static-ish file serving
```

```
runs/{job_id}/
  input.mp4
  video.mp4            # video-only, audio stripped  (-an -c:v copy)
  stems_demo.wav       # N-channel, gated  <- what the UI plays
  stems_raw.wav        # N-channel, ungated <- what the metrics script reads
  tracks.json          # per-frame bboxes + track ids
  meta.json            # fps, w, h, n_speakers, sr, durations, git sha, config hash
  log.jsonl
```

`job_id` is a server-generated UUID and the **only** path component derived from
anything client-side; the uploaded filename is never used to build a path. Reject
non-video content types and cap upload size — a demo box is still a web server
listening on a socket.

Bind to `127.0.0.1`, not `0.0.0.0`. On conference wifi, `0.0.0.0` exposes an
unauthenticated file-upload-and-transcode endpoint to the room. If the presenter
needs it on a phone, that is a deliberate, temporary `--host` change, not the
default.

### 5.3 Progress that reflects reality

Weight the SSE percentage by measured stage cost, not by stage count. Ordered by
typical share of wall clock: face detect + track (~35 %), separation (~40 %),
DSP chain (~10 %), matching (~5 %), encode/mux (~10 %). A progress bar that sits at
20 % for 40 s reads as a hang; one that moves proportionally reads as work.

Emit a heartbeat comment (`: ping\n\n`) every ~15 s so intermediaries don't reap an
idle stream, and remember from §4 that **`GZipMiddleware` will break SSE** — exclude
the events route if you enable compression.

### 5.4 Chunked inference (stability, not poverty)

You have the VRAM for whole-file inference on short clips; the reason to chunk is
that SepFormer's inter/intra-transformer attention is **quadratic in chunk count**,
so memory grows superlinearly with duration. A 30 s clip that fits and a 3 min clip
that OOMs are the same code path. Chunk unconditionally and duration stops being a
variable you have to think about.

- 10 s windows, 2 s overlap, at 16 kHz.
- **Align permutations across chunk boundaries.** Each `separate_batch` call is
  independently permuted — SepFormer has no cross-call speaker identity. Correlate
  chunk *k*'s overlap region against chunk *k−1*'s, both ways, and swap if the
  crossed pairing correlates better. Skipping this makes speakers trade places
  mid-sentence, which sounds exactly like the bleed-through bug and will send you
  debugging the wrong stage.
- Overlap-add the joins with a raised cosine over the 2 s region.
- `torch.inference_mode()` (not just `no_grad`), `model.eval()`, fp32.

Do **not** reach for fp16/autocast here. It saves memory you don't need and
introduces numerical risk in a chain whose entire purpose is a clean, exactly-zero
output. Save it for the day a 3-hour file appears.

Call `torch.cuda.empty_cache()` **between jobs, not between chunks** — per-chunk it
defeats the caching allocator and slows things down.

### 5.5 Environment: the two hard pins

This is where the build actually breaks. Two independent constraints, both verified
against current package metadata:

**(a) NumPy 2.x.** NumPy is at 2.5.2; `clearvoice` 0.1.2 pins `numpy<2.0,>=1.24.3`
(along with exact pins `librosa==0.10.2.post1`, `opencv-python==4.10.0.84`,
`soundfile==0.12.1`). SpeechBrain 1.1.0 requires only `numpy>=1.17.0` and is fine on
2.x. So the AV-TSE upgrade path (§3) and the SepFormer path have **mutually
incompatible dependency floors**.

> Do not try to satisfy both in one environment. Use **two venvs** — `.venv`
> (SepFormer + FastAPI, NumPy 2.x) and `.venv-avtse` (clearvoice, NumPy 1.26) — and
> have the backend invoke the AV-TSE separator as a subprocess against a file
> contract, not an import. The separator interface in §0 already makes this a
> drop-in: it takes a wav path and returns stem paths. This is the single highest-
> value structural decision for keeping the upgrade path open.

**(b) torchaudio is in maintenance and 2.9 removed APIs.** From the release notes:
2.8 deprecated the "Drop"-listed APIs, 2.9 removed most of them, and `load`/`save`
are now reimplemented on **TorchCodec**, which needs FFmpeg shared libraries present
at runtime. Only `forced_align`, `lfilter`, `overdrive`, `RNNT`, and `CUCTC` are
guaranteed preserved.

The robust move is to **stop depending on torchaudio for I/O**:

- Read/write audio with **`soundfile`** (0.14.0, BSD-3, ships a `win_amd64` wheel
  with libsndfile bundled — no system dependency).
- Resample with **`soxr`** (1.1.0) rather than `torchaudio.functional.resample`.
- Keep `torchaudio` installed only because SpeechBrain declares it
  (`torchaudio>=2.1.0`); don't build your own code on it.

This also removes the FFmpeg-shared-library question from the Python process
entirely. You still need the **ffmpeg binary** for demux/mux — install via
`winget install Gyan.FFmpeg` (or scoop/choco), verify with `ffmpeg -version` at
startup, and fail loudly with an actionable message if absent rather than at minute
three of a job.

**Versions to install** (verified present as Windows cp312 wheels):

```
python 3.12
torch 2.11.0+cu128   torchaudio 2.11.0+cu128    # --index-url .../whl/cu128
fastapi 0.141.1      uvicorn 0.52.1
speechbrain 1.1.0    soundfile 0.14.0   soxr 1.1.0   opencv-python
```

**Pin torch and torchaudio to the same minor version.** `torch` ships cp312 Windows
wheels up to 2.13.0, but `torchaudio` stops at **2.11.0** on every CUDA channel
(a consequence of the maintenance freeze above). Installing torch 2.12/2.13 leaves
torchaudio unsatisfiable at a matching version, and SpeechBrain imports it. **2.11.0
is the newest matched pair**, and it exists on cu126, cu128 and cu130 alike — pick
the channel matching the installed driver; cu128 is the safe middle. Ada
(RTX 40-series, `sm_89`) is fully supported by all three, so the old
`sm_61`/cu118 pinning advice does **not** apply to this machine.

Verify immediately after install, before writing any pipeline code:

```python
import torch, torchaudio
print(torch.__version__, torchaudio.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())        # expect sm_89 present
```

Set `HF_HOME` to a project-local path before first run. The default
`%USERPROFILE%\.cache\huggingface` puts a multi-GB model tree on C:, and the
AV-MossFormer2 checkpoint alone is 735 MB.

```
:: run.bat
set HF_HOME=%~dp0.cache\hf
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 5.6 Dual output: demo stems and metrics stems

The user requirement is explicit — aggressive muting for the demo, an unmuted path
for SI-SDR/PESQ. Objective scores are **destroyed** by gating: a hard gate zeroes
frames the reference still has signal in, and SI-SDR punishes that heavily. Scoring
the demo mix would understate the system's real performance.

So gating is a **flag on one function**, and both artefacts are written on every
run:

```python
def separate(video, *, gate: bool) -> list[np.ndarray]: ...

stems_raw  = separate(video, gate=False)   # A1 + A3 only  -> metrics
stems_demo = separate(video, gate=True)    # A1 + A3 + A4  -> UI
```

Emitting both always (they differ by one cheap time-domain multiply) means the
metrics number and the demo audio provably come from the same run — no "which
build produced that score?" ambiguity when a judge asks.

---

## 6. Build order — always keep a working demo

The ordering rule: **at no point should there be a day where nothing runs.** Each
step ends with something you could put on a projector. Steps 1–3 are the ones that
decide whether the rest of the plan is even necessary.

### Step 0 — Environment (half a day)

`.venv` on Python 3.12, torch 2.11.0+cu128 and torchaudio 2.11.0+cu128 (matched
pair — see §5.5), verify `torch.cuda.is_available()`, that
`torch.cuda.get_device_name(0)` reports the 4060, and that `get_arch_list()`
contains `sm_89`. Install ffmpeg, verify on PATH. Set `HF_HOME`. Warm the SepFormer
checkpoint once so the download isn't in the critical path later.

### Step 1 — The one experiment that could invalidate §2 ⚠️

Before building anything: take one real clip with a genuine monologue stretch, run
it through both paths, and measure.

```python
a = model.separate_file(path=wav)     # current prototype behaviour
b = model.separate_batch(mix)         # proposed
# for each, during a known single-speaker stretch:
#   10*log10(mean(ch2**2) / mean(ch1**2))     <- the leak ratio
```

The synthetic measurements behind §2 (formant-synthesis speech with artificial
leakage at α=0.12, *not* real SepFormer output) predict this jumps roughly 20+ dB
between the two calls. **If fixing the normalisation alone drops the ghost to
inaudible, the DSP chain is optional** and Objective A is essentially done on day
one. Build A3/A4 only if this measurement says you still need them.

This is also where you re-tune the A4 thresholds. The values in §2 (hi −30 dB,
lo −40 dB, min_on 60 ms, min_off 30 ms, p95 reference) come from synthetic signals;
the *relative orderings* should transfer, the **absolute numbers must be re-fitted
against real output**. Treat them as initial conditions, not constants.

### Step 2 — Offline CLI, no server (2–3 days)

`python -m app.pipeline in.mp4 --out runs/x/`, producing the full artefact set from
§5.2. No FastAPI, no browser. Debug the pipeline where the stack traces are short.
Write `stems_raw.wav` and `stems_demo.wav` from the start — retrofitting the dual
output later means re-plumbing every stage.

Fix the lip feature here (§1, Bug 3): compute landmarks on the **full frame**, or
map crop-normalised coordinates back through the crop offset and scale. Then use a
**scale-invariant** feature — inner-lip polygon area (shoelace over the inner-lip
ring) divided by squared inter-ocular distance — so the signal cannot drift with
bbox size or subject distance. Log it alongside the energy envelope and eyeball the
two before trusting any correlation.

### Step 3 — Static playback in the browser (1–2 days)

FastAPI serving one HTML page. Hard-code a `job_id` from step 2, no upload, no SSE.
Load stems + video, get the splitter/gain graph and the drift loop working, prove
the crossfade is inaudible. This is Objective B's actual risk — sync and switching —
isolated from all backend concerns.

### Step 4 — Wire the job pipeline (1–2 days)

Upload → SSE progress → artefacts → playback. Now it's an application.

### Step 5 — Face overlay and click-to-select (1 day)

Canvas, hit-testing, and the button fallback. Note that until this step you select
speakers from a hard-coded button, which is a perfectly good demo already.

### Step 6 — Metrics harness (1 day)

Score `stems_raw.wav` with SI-SDR / PESQ / STOI against references. Keep it a
separate script reading the artefact directory — never wire metrics into the
serving path.

### Step 7 — AV-TSE, only if steps 1–6 are solid (optional, 3–5 days)

Second venv, subprocess contract, same artefact format (§3, §5.5). If it works it
replaces separation + matching + gating in one move. If it doesn't, you delete a
directory and still have a working system. **Do not start this before step 6.**

### Timing sanity

At 16 kHz float32, a 3-minute 2-speaker job is ~23 MB of stems — a ~1 s
`decodeAudioData` and a trivial local fetch. Nothing about the payload needs
optimising at demo scale. Keep demo clips **under ~2 minutes**; that is a
presentation decision (attention span, upload+process time on stage), not a
technical limit.

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | Step 1 shows the ghost survives the `separate_batch` fix | Medium | High — §2's chain becomes mandatory, not optional | The chain is already designed; this converts optional work into planned work, not a redesign |
| 2 | A4 gate clips genuine speech onsets after re-tuning | Medium | High — worse than the ghost, and obvious on stage | Gate on the target's **own** energy, never dominance; 30 ms min_off; always compare against `stems_raw` by ear |
| 3 | Cross-chunk permutation swap | High if unhandled | High — sounds identical to the original bug | Correlation-align every boundary (§5.4); assert on it in step 2 |
| 4 | NumPy 1.x/2.x conflict if AV-TSE is added to the main venv | High if attempted | High — breaks a working system late | Two venvs, subprocess contract, decided up front (§5.5) |
| 5 | torchaudio 2.9 removed an API you depend on | Medium | Medium | Depend on `soundfile` + `soxr` for all I/O and resampling; torchaudio stays only as SpeechBrain's declared dep |
| 6 | Lossy encoding of stems reintroduces audible residue | Low (designed out) | High — undoes all of §2 in the last mile | PCM WAV only; never AAC/Opus for stems |
| 7 | Autoplay policy blocks the AudioContext on the demo machine | Medium | Medium — silent demo, looks broken | `ctx.resume()` inside the click handler; check `ctx.state` before `start()`; rehearse in the actual presentation browser |
| 8 | A/V drift over a long clip | Low | Medium | Bounded ±2 % `playbackRate` correction on the muted video; hard reseek past 250 ms and on `visibilitychange` |
| 9 | Model download at demo time | Low | High | Warm the cache in step 0; the app must run fully offline |
| 10 | Judges ask for metrics on the muted output | Medium | Low | `stems_raw.wav` exists for exactly this; explain that gating is a presentation layer |

### The two things most likely to actually bite

**Permutation instability across chunks (row 3)** and **a mis-tuned gate (row 2)**
both produce symptoms that sound like the original bug. If, mid-integration, the
ghost seems to "come back", check those two before touching the DSP chain — the
temptation will be to conclude §2 doesn't work, and that will cost a day.

---

## 8. What I'd tell you if you only read one paragraph

The bleed-through is, most likely, **an amplitude bug rather than a DSP failure**:
`separate_file()` peak-normalises each source independently, and Cell 4 then
normalises a second time, so a −25 dB residual channel gets rescaled to full scale —
loudest, relatively, exactly during pauses. Fix that first (step 1), measure, and
only then decide how much of the silence chain you need. Meanwhile the lip gate
never had a chance, because Cell 8 measures lip opening in **crop-normalised**
coordinates — a ruler whose length changes with the bounding box — so no fixed
threshold could ever have worked. Both are one-line-ish fixes to things that look
like deep problems, which is why the three previous attempts failed: they were
tuning gates on top of a broken measurement and a 25 dB gain error.



