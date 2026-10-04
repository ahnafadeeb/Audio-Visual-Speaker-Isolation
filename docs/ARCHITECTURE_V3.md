# Audio-Visual Speaker Isolation — v3 Architecture

**EEE 312 DSP-I Lab · Group 4 · Level-3 Term-I Section C2**
Status: plan · Supersedes the separation + matching stages of `ARCHITECTURE_V2.md`
Target: Windows 11, RTX 5060 Laptop (8 GB, Blackwell `sm_120`), Python 3.12

---

## Context

The v2 system works and is well-engineered, but it fails four of the five
must-have requirements, and `docs/DIAG_MATCHER.md` proves the failure is
**architectural, not a tuning problem**:

> The matcher scored **0.0679** on the real clip. The same statistic on **pure
> white noise** scored **0.0683**. Fifteen alternative audio features all invert
> the pairing. A 400-shift circular null gives **p = 0.71–0.96**.

The face→voice matcher is a random number generator. No feature engineering
repairs it, because on a talk show both mouths move constantly, so a scalar
lip-motion feature tracks *total* speech activity — measured `r ≈ +0.18` for
**both** faces, to three decimals — and the one component that survives the
landmark noise floor is precisely the component that cannot discriminate.

v3 therefore replaces blind separation + post-hoc matching with **audio-visual
target speaker extraction (AV-TSE)**: condition the network on one face's mouth
video, get that speaker's voice out. Matching stops being a stage.

**Do not rebuild from scratch.** The measurements in `dsp.py`, `separation.py`
and `scripts/diag_*.py` are the most valuable asset in this repository and took
months to accumulate. The defect is localised to two stages. Replace those.

---

## 1. Verdict on each known problem

| # | Problem | v3 resolution | Mechanism |
|:--|:--------|:--------------|:----------|
| 1 | Bleed-through | **Fixed, and measured** | Identity gate (§4a) composed with the acoustic VAD. Every scored pause is bit-exact zero and silent at +40 dB, on both paths (§9.2) |
| 2 | Voice alteration | **Fixed** | Root cause is the *checkpoint*, not chunking (§3) |
| 3 | 2-speaker limit | **Fixed** | One model call per face; N is unbounded |
| 4 | Face tracking loss | **Not started** | Embedding re-ID is designed (§5) but unbuilt; the resolution floor it implies is already live and the reference clip's track 1 (126 px) trips it on every run |
| 5 | Chunk swapping | **Eliminated** | One output per face — there is no permutation to get wrong |
| 6 | Matcher unreliable | **Eliminated** | There is no matcher |

Problems 3, 5 and 6 do not get *better*; they stop existing. That is the
argument for the architecture change over further tuning. Problem 1 changed
shape rather than closing: the leak this document was written to chase was
never the main one (§4a).

---

## 2. The headline decision: vendor the model, use ONE venv

`ARCHITECTURE_V2.md` §5.5 mandates two virtualenvs and a subprocess file
contract, because `clearvoice==0.1.2` pins `numpy<2.0`, `opencv-python==4.10.0.84`,
`librosa==0.10.2.post1`, `scenedetect==0.6.6`.

**Those pins belong to clearvoice's packaging, not to the model.** Verified by
walking the full import closure of `AV_MossFormer2_TSE_16K`:

| File | Size |
|:-----|-----:|
| `av_mossformer2.py` | 7.1 KB |
| `visual_frontend.py` | 6.1 KB |
| `mossformer/utils/one_path_flash_fsmn.py` | 22.6 KB |
| `mossformer/utils/Transformer.py` | 15.0 KB |
| `mossformer/utils/conv_module.py` | 3.5 KB |
| `mossformer/utils/fsmn.py` | 3.3 KB |
| `mossformer/utils/normalization.py` | 2.6 KB |
| **Total** | **~60 KB** |

Complete third-party import set across all seven files:

```
torch  torchaudio  numpy  einops  rotary_embedding_torch
```

`einops` 0.8.2 declares **no dependencies**. `rotary-embedding-torch` 0.9.1
declares only `einops>=0.8, torch>=2.4`. Neither pins numpy.

**Vendoring ~60 KB of Apache-2.0 PyTorch deletes:** the second venv, the
subprocess contract, the file-based IPC, the `.cuda()` monkey-patch, the 25 fps
hardcode, and risk-register row 4. It also hands us direct control of chunking
and VRAM, which the subprocess path could never have.

Vendor into `app/avtse/` with the upstream Apache-2.0 notice and a
`VENDORED.md` recording the source commit. Fetch weights with
`hf_hub_download("alibabasglab/AV_MossFormer2_TSE_16K", "last_best_checkpoint.pt")`
— never `git clone`, the repo carries a duplicate 734 MB `_old` checkpoint.

Two edits to the vendored copy, both recorded in `VENDORED.md`:
- `av_mossformer2.py:177` — `.cuda()` → `.to(signal.device)`
- Drop the `Mossformer` wrapper's checkpoint-path logic; we load the `state_dict` ourselves.

---

## 3. Why the voice sounds wrong today — and why AV-TSE fixes it

The stated hypothesis (overlap-add phase discontinuities at chunk boundaries) is
**not the cause**. SepFormer is a time-domain model with a learned
encoder/decoder; it does not multiply the input STFT, so there is no STFT phase
to discontinue.

The actual cause is already recorded in `ARCHITECTURE_V2.md` §2 A2, without the
perceptual consequence being drawn. The WHAMR recipe trains:

```
input:  mix_both_reverb        (reverberant + noisy)
target: s1_anechoic            (dry)
```

**`sepformer-whamr16k` is trained to dereverberate.** Stripping the room
signature *is* "sounds different from the original" — drier, closer, thinner.
It is the checkpoint's designed behaviour and no post-processing restores it.

AV-MossFormer2 is trained on VoxCeleb2/LRS2 — real in-the-wild recordings where
the target keeps its room. Changing checkpoint is the fix.

**Secondary contributor, worth fixing regardless.** The current 10 s chunk / 2 s
overlap means **20 % of every output sample is a blend of two independent
inferences**. Those two estimates differ slightly in phase, so the blend
comb-filters — audible as coloration. Fix by decoupling the two uses of overlap:

- **alignment/context window** — keep 2 s of evidence
- **audio crossfade** — shorten to ~128 ms

Cheap, strictly better, and applies to either separator.

---

## 4. Zero bleed-through: reframe the problem

> **Superseded in part — read this section with §4a.** The two-regime split and
> the target-VAD conclusion below are correct and shipped. The *discriminator*
> proposed in "v3 gate" — a stream-to-mixture energy ratio — was never built,
> because the clip turned out to contain a speaker this section does not
> account for, against whom the ratio provably cannot work. §4a records the
> argument and what replaced it. The reasoning here is kept because the failure
> is instructive: it is the second time on this project that an energy
> statistic was proposed for an identity problem.

Split the requirement into its two perceptual regimes, because they have
different answers and conflating them is why this has resisted three attempts:

**Regime A — target active, interferer active.** Leakage is masked by the
target. The measured −46.8 dB is *already* inaudible here. Chasing it further
costs fidelity and buys nothing. Stop optimising this.

**Regime B — target silent, interferer active.** Leakage is fully exposed. This
is the only place the ghost whisper lives, and the correct output is exactly
`0.0`, which is achievable.

So **"zero bleed-through" is not a masking problem, it is a target-VAD
problem.** The mask cannot deliver it and should stop trying.

### The specific defect in the current gate

`gate_mask()` decides from **the stem's own frame energy** — a signal
contaminated by the very leakage it is trying to remove. During a target pause
with the interferer loud, leakage lifts the energy floor above `close_db` and
the gate cannot distinguish "quiet target" from "loud leak". That is the ghost,
and it is structural.

### v3 gate: decide from evidence independent of the leak

Fuse three signals, each covering the others' blind spot:

| Signal | Covers | Fails when |
|:-------|:-------|:-----------|
| AV-TSE stream energy vs its own p95 | normal speech | very quiet target |
| **Stream-to-mixture energy ratio** | exposed pauses | both silent (harmless) |
| Visual activity from the 112×112 mouth ROI | acoustic ambiguity | face occluded |

The second is the new one and it is the important one: during a target pause the
stream/mixture ratio collapses even though the stream's *absolute* energy does
not. That is exactly the discriminator the current gate lacks.

Keep everything else in `dsp.py`: hysteresis, `min_on`/`min_off`, lookahead,
raised-cosine ramps, exact-zero output. The machinery is right; only its input
was wrong.

One refinement: **erode the open region by `ramp_ms` before ramping**, so the
fade happens over the tail of real speech rather than over silence. Today the
ramp is partially-open *inside* the silent region — a 10 ms window where the
ghost is audible at every boundary.

> **Retired — see §9.2.** Measured: erosion improves the boundary peak by 14 dB
> and costs 0.19 % of speech. But `_advance()` / `lookahead_ms` already put the
> ramp there *on purpose*, to avoid eating onsets, and the region it improves is
> already inaudible and outside the requirement. Not worth reversing a settled
> decision for.

---

## 4a. Correction: there is a third regime, and energy cannot see it

§4 assumes only two voices exist. The reference clip has at least three: two
tracked faces and **a man who is never on screen**. That breaks the regime
split, because it admits a case neither regime covers:

**Regime C — target silent, an *untracked* person active.** AV-TSE is an
extractor, not a detector. Conditioned on a face, it emits a voice on *every*
frame; when that face is silent it returns the most speaker-like thing
available, and if the only person talking is off-screen, that is what comes
out. Measured on `runs/862bd92a01ac`: ~8 s of 32 on face 0.

Every discriminator in §4's table passes this case:

| Signal | On an off-screen intruder |
|:-------|:--------------------------|
| stream energy vs own p95 | **passes** — it is loud |
| stream-to-mixture ratio | **passes** — the intruder *is* in the mixture, so the ratio is high, not collapsed |
| visual activity | **passes or abstains** — the tracked mouth is closed, but this is the one signal that could veto, and it is also the one that fails on a 126 px face |

The intruder is a loud, clean, perfectly good voice that is genuinely present in
the room. No energy statistic distinguishes it from the target, because energy
is not the axis on which it differs. **What differs is who it is.**

### What shipped instead: `app/identity.py`

Ask the identity question directly, with ECAPA-TDNN embeddings over 2 s windows,
and score each window *relatively*:

    s(w) = cos(w, ref_mine) − max over other refs of cos(w, ref_other)

Absolute similarity does not separate — measured on this clip it gives a
*negative* margin, because the on-screen man and the off-screen intruder sound
alike (centroids at cosine +0.74). The difference separates cleanly: margin
+0.183. Comparing two similar voices *to each other* is a far easier question
than describing either one absolutely.

Three things about the design are load-bearing:

* **The mixture is the arbiter of how many people exist.** Cluster the stems
  *and the mixture* jointly, then grow `k` only while every cluster still holds
  ≥ `min_share` of the **mixture's** windows. A cluster that owns part of a stem
  but no part of the room cannot be a person, because the stem was derived from
  the room. This is a conservation argument, not a fitted threshold, and on this
  clip it is not close: at `k=4` the spurious cluster holds 48% of face 0's stem
  and **0.0%** of the mixture, against 5% for the real off-screen speaker.
  Two alternatives were measured and rejected — a centroid-cosine threshold has
  to fit inside a 0.044-wide gap, and split-half stability is degenerate (it
  scores `k=2` at 1.000 and would detect no intruder at all).
* **Faces claim clusters exclusively**, via Hungarian assignment, so two faces
  cannot both claim one voice. Unclaimed clusters are the intruders.
* **Unverified audio closes, it does not inherit a verdict.** Windows below the
  loudness gate were never measured, so holding the nearest verdict outward
  would report an extrapolation as a decision — in either direction. `covered()`
  makes that explicit, and the report splits muted time into "wrong identity"
  and "never verified" rather than claiming the total as a detection rate.

### The identity gate does not replace the VAD

They answer different questions — *"is this someone else?"* versus *"is anybody
speaking?"* — and the AV path needs both. With identity gating alone, face 1's
channel never reached a digital zero: its quietest 1% of frames sat 44 dB below
its own speech and were plainly audible at +40 dB. `Pipeline._run_avtse`
therefore composes `dsp.apply_gate` with the identity envelope.

The composed gate runs **pure-acoustic, without visual fusion**, and the margin
is the reason. `open_db`/`close_db` are relative to each stem's own p95. v2
needed lip motion because SepFormer's residual interferer sat at p50 = −31 dB
relative to p95 — *inside* the −20/−23 dB hysteresis band, where no threshold
can separate it from quiet speech. AV-TSE has already removed the interferer, so
what survives a pause is 44 dB down, roughly 20 dB clear of the band. Adding lip
motion buys nothing measurable there and introduces an unmonitored way to clip
real words — unmonitored because the veto that detected exactly that failure is
one of the stages this path deletes. Measured retention outside the intruder
spans: **99.6% / 99.7%**, against the v2 fit's own 97.9%.

---

## 5. The hard constraint AV-TSE imposes: face resolution

Exact preprocessing contract, read from `video_process.py` and `visual_frontend.py`:

```
face box → pad by cropScale=0.40 → resize 224×224 → CENTER-CROP 112×112
        → cv2.COLOR_BGR2GRAY → /255.0 → (x - 0.4161) / 0.1688
tensor (B, T, 112, 112);  frontend unsqueezes to (B, 1, T, 112, 112)
25 fps and 16 kHz are a DATA contract, not just a code hardcode
frame count is derived from audio: int(n_samples / 16000 * 25), edge-padded
```

112×112 is pinned by `nn.AvgPool2d(kernel_size=(4,4))` after four stride-2
stages — any other resolution breaks the final `reshape(B, -1, 512)`.

**Consequence:** the mouth ROI is the central half of the padded face box. For
it to carry real viseme detail the native face box wants to be **≥ 150 px**.

- `runs/862bd92a01ac` has faces at 162×162 and 127×127 px — **workable**.
- Problem 4's 640×360 clip with ~45 px faces — **AV-TSE will fail there.**

This is a data requirement, not a code one. **Demo clips must be ≥ 720p with
faces ≥ 150 px.** Enforce it in preflight and fail loudly, do not degrade
silently.

### Why AV-TSE succeeds where the correlation matcher failed

`DIAG_MATCHER.md` concludes "the lip signal cannot tell who is speaking". That
verdict is about **a scalar**, and must not be generalised to the modality:

- The matcher compressed the mouth to **one number per frame** (inner-lip area
  ratio), then correlated it against **one number per frame** (energy in dB).
  On a 162×162 px face the inner-lip ring is ~20×11 px and its shoelace area is
  *quadratic* in landmark noise. What survived was the common component.
- AV-TSE feeds the **full 112×112 grayscale ROI at 25 fps** into a 3D-CNN +
  ResNet-18 lipreading frontend → 512-d per frame. That encodes **visemes** —
  *what* is being said — not "is the mouth moving".

Viseme sequence is discriminative against a specific waveform in a way a motion
scalar can never be. The information was always in the pixels; the scalar threw
it away.

---

## 6. Module plan

### Keep unchanged
`app/static/app.js` (single audio clock, ChannelSplitter, equal-power
`setValueCurveAtTime`, rVFC drift control — correct as written) ·
`app/main.py` · `app/jobs.py` · `app/media.py` · `app/serialization.py` ·
the dual `stems_demo` / `stems_raw` contract · every `scripts/diag_*.py`.

### Replace
| File | Change |
|:-----|:-------|
| `app/avtse/` | **new** — vendored model (§2) |
| `app/separation.py` | add `AVTSESeparator`; one `separate_one(mix, mouth_roi)` call per face. Keep `SepformerSeparator` as fallback |
| `app/vision.py` | emit 112×112 normalised mouth ROIs (§5) alongside boxes; add embedding re-ID across occlusion gaps |
| `app/dsp.py` | ~~`gate_mask()` gains a `reference` input for the stream/mixture ratio; erode-before-ramp~~ — **unchanged in the end.** Both refinements were retired (§4a, §9.2); the v2 gate transfers as-is because its thresholds are relative |
| `app/identity.py` | **new, unplanned** — the off-screen intruder was not known when this document was written (§4a) |
| `app/matching.py` | **off the critical path.** Retain only as telemetry — it is a useful honesty check, not a decision-maker |
| `app/channels.py` | collapses to identity: face *i* → channel *i* by construction |
| `app/config.py` | add `AVTSEConfig`; `n_speakers` stops being a constant |

### Interface (unchanged shape, so the frontend never learns about any of this)

```python
class AVTSESeparator:
    def separate_for_faces(
        self,
        mixture: np.ndarray,          # (n_samples,) 16 kHz mono
        rois: list[np.ndarray],       # per face: (T, 112, 112) float32, normalised
        progress: ProgressFn | None = None,
    ) -> np.ndarray:                  # (n_faces, n_samples) float32
```

---

## 7. Environment (RTX 5060, Blackwell)

One correction to the brief: the RTX 5060 Laptop is **`sm_120` only**.
`sm_100` is datacenter Blackwell (B100/B200) and will not appear.

Verified available as matched Windows `cp312` wheels:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install einops rotary-embedding-torch huggingface_hub
.venv\Scripts\pip install -r requirements.txt
```

`torch` ships wheels to 2.14.0 but **`torchaudio` stops at 2.11.0** on every
channel, so 2.11.0 remains the newest matched pair. Confirmed present:
`torch-2.11.0+cu128-cp312-cp312-win_amd64.whl` and
`torchaudio-2.11.0+cu128-cp312-cp312-win_amd64.whl`.

**Python 3.12 is no longer forced by mediapipe.** The brief says "mediapipe
0.10.21 tops out at cp312"; mediapipe is now at **1.0.1**, shipped as
`py3-none-win_amd64` — no CPython ABI tag at all. Stay on 3.12 anyway because
that is what the matched torch/torchaudio pair is verified against, but record
the real reason.

⚠️ mediapipe 1.0 very likely **removed the legacy `solutions.face_mesh` API** in
favour of MediaPipe Tasks `FaceLandmarker`. Pin `mediapipe==0.10.21` until the
migration is done, or drop mediapipe entirely — §6 needs boxes and a mouth ROI,
not 468 landmarks.

```bat
:: run.bat
set HF_HOME=%~dp0.cache\hf
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Machine reality check
`nvidia-smi` on the current box reports **GTX 1050 Ti, 4 GB, Pascal `sm_61`**,
and `.venv` holds `torch 2.11.0+cu126`. The 5060 is not this machine. Keep the
device/VRAM probe (`resolve_chunk_s()`) so one codebase runs on both, and treat
every 5060 number as unverified until measured there.

**This venv cannot run on the 5060 at all — fix before the demo.** The
installed wheel is a **cu126** build and its compiled architectures are

    sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90

with **no `sm_120`**. Blackwell is `sm_120`, so the 5060 has neither a matching
binary nor a forward-compatible PTX fallback here: CUDA init fails outright
rather than running slowly. §7 specifies cu128 and that is the reason — the
requirement is not a preference about speed, it is the difference between the
GPU working and not. Re-install from the cu128 index on the target machine and
re-run `bench_vram.py` there; `resolve_chunk_s()` will pick a much larger
`chunk_s` off 8 GB than the 9.31 s it resolved here, so the 3.9× RTF in §9.1 is
the pessimistic end of the range, not a prediction.

---

## 8. Build order

Each step ends with something demonstrable.

**Step 1 — AV-TSE spike (highest risk, do first).** Vendor the 7 files, load the
checkpoint, run one hand-cropped face from `runs/862bd92a01ac` through it,
write a WAV. No pipeline, no server. *Gate: is the extracted voice the right
speaker, and does it retain the room?* If this fails, nothing downstream
matters.

**Step 2 — ROI extraction.** `vision.py` emits `(T, 112, 112)` matching §5
byte-for-byte. Verify by writing the ROI stack back out as an MP4 and watching
it — a misaligned crop is instantly visible and silently fatal.

**Step 3 — Wire into the pipeline.** `AVTSESeparator` behind the existing
interface, per-face loop, identity channel map. Keep `stems_raw` / `stems_demo`.

**Step 4 — Re-point the gate.** ~~Add the stream/mixture reference (§4)~~ — see
§4a: that signal cannot see an off-screen intruder, and what shipped is
`app/identity.py` composed with the existing acoustic gate. The v2 constants
were expected not to transfer and **did** transfer unchanged, because
`open_db`/`close_db` are relative to each stem's own p95 rather than absolute:
measured retention is 99.6 % / 99.7 %, above the 97.9 % of the v2 fit itself. No
re-fit was needed and `scripts/fit_gate.py` was not re-run against AV-TSE output.

**Step 5 — Verification harness.** **Done for silence:** `scripts/ghost_test.py`
implements the +40 dB ghost test against a real run directory — it belongs in
its own script rather than inside `redteam_silence.py`, which is a synthetic
harness taking no input and would have had to grow a second personality. Both
paths pass at 100.000 % (§9.2). **Still open:** SI-SDR / PESQ / STOI on
`stems_raw`, which measure a different thing — fidelity, not silence.

**Step 6 — Face robustness.** Embedding re-ID across occlusion gaps, profile
handling. Only after 1–5 are solid.

---

## 9. Verification

| Requirement | Test | Pass condition | Status |
|:------------|:-----|:---------------|:-------|
| Zero bleed-through | `ghost_test.py --write` | scored pauses are **bit-exact 0.0**; nothing audible at +40 dB | **pass** — 100.000 % on both channels, both paths (§9.2) |
| No voice alteration | A/B `stems_raw` vs mixture on a solo stretch | room character preserved; no dereverb "drying" | not yet run |
| No mid-clip swapping | `diag_chain.py` per 5 s window | binding constant for the whole clip | **met by construction** — face *i* conditions channel *i*, so the assignment is the identity permutation and there is no per-chunk decision to drift |
| Reliable matching | identity purity per face | every face's channel is majority that face | **pass** — 0.74 / 1.00 (§9.1) |
| Clean silence | `ghost_test.py` | ≥ 99.9 % of pause samples exactly `0.0` | **pass** — 100.000 % of 3.59 s / 2.05 s scored pause, zero nonzero samples |
| Crossfade click-free | existing `check_transport.py` | unchanged | pass |

`runs/862bd92a01ac` is the regression clip: v2 scores **0 of 2 faces correct**
and is reproducible bit-for-bit (`runs/pair_v2` is byte-identical). Any v3 build
that does not score 2/2 on it has not fixed the problem.

### 9.1 Measured, AV path end to end (`runs/v3_e2e`, `--separator avtse`)

| | purity | nearest rival | identity-muted | of which wrong identity | acoustic retention¹ | total bit-zero |
|:--|:--|:--|:--|:--|:--|:--|
| channel 0 | 0.737 | +0.747 | 33.2 % | 25.45 % | 99.6 % | 50.4 % |
| channel 1 | 1.000 | +0.311 | 0.0 % | 0.0 % | 99.7 % | 20.9 % |

¹ share of frames *outside* the known intruder spans (0.00–5.24 s, 13.26–18.74 s)
that the acoustic gate passes. Channel 0's headline 72 % retention looks like the
VAD eating speech and is not: restricted to frames where the target is the only
person talking it is 99.6 %. The rest is the identity gate removing the
intruder, which is the point.

`_choose_k` selected **k = 3 unforced**: mixture shares `[0.24, 0.69, 0.06]` at
k=3, all ≥ `min_share`; `[0.21, 0.69, 0.00, 0.10]` at k=4, so it stopped. Claims
`{face 0 → c2, face 1 → c1}`, unclaimed `[c0]` — one off-screen intruder,
recovered without being told how many people to look for.

Two numbers are worth reading together. Face 0's nearest rival sits at **+0.747**
— the intruder genuinely sounds like him — and its purity is **0.737**, meaning a
quarter of that channel was somebody else before gating. Face 1's rival is
+0.311 and its purity 1.000: nothing to confuse, nothing muted. The gate did work
on exactly the face that needed it and left the other alone.

Cost on the development GPU (GTX 1050 Ti, `chunk_s` resolved to 9.31 s from free
VRAM): **126 s for a 32 s clip, ≈ 3.9× RTF.** Not real-time, and the nice-to-have
target of < 1× RTF is not met on this hardware.

### 9.2 The ghost test, and the two ways it was measured wrong first

`scripts/ghost_test.py` takes a run directory, finds the stretches where the
target demonstrably is not speaking, amplifies them by 40 dB, and writes the
result out to listen to. On both paths, every scored pause is **bit-exact
digital silence**:

| | scored pause | bit-exact zero | +40 dB peak | acoustic damage to speech |
|:--|:--|:--|:--|:--|
| v3 ch 0 | 3.59 s in 13 pauses | **100.000 %** | −∞ (no nonzero sample) | 0.01 % |
| v3 ch 1 | 2.05 s in 3 pauses | **100.000 %** | −∞ | 0.00 % |
| v2 ch 0 | 4.83 s in 15 pauses | **100.000 %** | −∞ | 0.43 % |
| v2 ch 1 | 11.89 s in 43 pauses | **100.000 %** | −∞ | **3.68 %** |

`ghost_ch0.wav` / `ghost_ch1.wav` are 32.3 s files containing zero nonzero
samples. The requirement is met.

The last column is an unlooked-for result and the clearest quantitative case
for the architecture: **v2's gate removes 3.68 % of channel 1's real speech;
v3's removes 0.01 % and 0.00 %.** That is the margin argument in §4a showing up
as a measurement. v2's gate has to operate with the interferer's residual
sitting inside its hysteresis band, so it cannot help clipping speech; v3's has
20 dB of clearance and never has to decide anything difficult.

Getting to that table took two wrong statistics, both worth recording because
both looked authoritative.

**Wrong once — a window pinned to the threshold it audits.** `whisper_db` reads
−23.9 / −20.7 dB here, and cannot mean anything: its window is defined by
`active_db = 20.0`, the *same* 20 dB point as the gate's `open_db`. It averages
exactly the frames the gate is deliberately undecided about and lands near
−20 dB regardless. It is left in `meta.json` unchanged — redefining a published
field to make a row go green is the wrong direction — but it does not answer
this. `ghost_test.py` instead keeps every scored frame ≥ 10 dB clear of the
band on either side and reports the band itself as unscored.

**Wrong twice — clearance in level but not in time.** The first version of the
script scored any quiet frame and reported 99.04 % / 93.93 % zero with peaks at
−38.6 / −32.9 dBFS, which reads as a leak. It was not one. Every nonzero sample
sat within 25 ms of a zone edge, and the interior of all 111 quiet runs — 9.9 s
— was bit-exact zero with not one nonzero sample. Two deliberate mechanisms
live at those edges:

* the raised-cosine ramp, which `lookahead_ms` intentionally places in the
  silence rather than over the onset (`dsp.py:412`);
* `min_on_ms`, which holds the gate open through short dips — and on channel 1,
  a purity-1.000 channel, those dips are stop closures and inter-syllable gaps
  *inside words*. Zeroing them would be the defect, not the fix.

So a pause needs a duration, not just a level. The script scores quiet runs
≥ `--min-pause` (default 150 ms = 2.5 × `min_on_ms`), excluding a `ramp_ms +
lookahead_ms` margin at each edge, and prints everything it excluded along with
that region's peak. The verdict does not depend on the choice: 150, 250 and
400 ms all read 100.000 %. At 600 ms channel 0 has no run long enough to score,
which the script reports as *cannot test* rather than as a pass.

**This also retires §4's erosion refinement.** "Erode the open region by
`ramp_ms` before ramping" was measured: it improves the boundary peak on
channel 0 by 14 dB (−38.6 → −52.9 dBFS) and costs 0.19 % of speech-zone
samples. But it would reverse a deliberate, documented decision — `_advance()`
exists precisely to put the ramp in the silence instead of eating the onset,
and `lookahead_ms == ramp_ms` for that reason. Since the region it improves is
already inaudible and outside the requirement, the trade is not worth
re-opening a settled question for.

---

## 10. Honest risks

| # | Risk | Mitigation |
|:--|:-----|:-----------|
| 1 | Vendored `state_dict` keys mismatch the upstream wrapper | **Retired** — Step 1 passed; the model loads and extracts |
| 2 | Faces below ~150 px → AV-TSE underperforms | **Live and firing.** The reference clip's track 1 is 126 px and warns on every run. It still extracts cleanly (purity 1.000), so the floor is currently a warning rather than a measured failure — do not read that as headroom, read it as one clip |
| 3 | Preprocessing subtly wrong (crop offset, the `0.4161/0.1688` constants) | Silent quality loss, not a crash — hence the Step 2 visual check |
| 4 | N model calls for N faces | Linear, not quadratic; 8 GB is ample at N ≤ 4. Cache the audio encoder across faces if it bites |
| 5 | LRS2-derived weights carry a research-only training-data restriction | Fine for coursework. Prefer VoxCeleb2 checkpoints if that ever changes |
| 6 | 5060 is unverified hardware | **Worse than unverified: the current venv cannot start CUDA there** — cu126 wheel, no `sm_120`. Re-install cu128 on the target box, then re-measure (§7) |
| 7 | Everything in §9.1 is one clip, with one intruder, and the identity gate's thresholds were set on it | The structural choices are defensible without fitting (zero is the natural decision point; mixture attestation is a conservation argument, not a threshold). The *numbers* are not evidence of generalisation. A second clip with a different speaker count is the cheapest test left undone |

---

## 11. What I would tell you if you read one paragraph

The matcher is measurably a random number generator — 0.0679 on your clip,
0.0683 on pure noise — so face→voice assignment cannot be fixed by tuning, and
three of your six problems disappear only if separation becomes
face-conditioned. Do that with AV-MossFormer2, but **vendor the 60 KB of model
code instead of installing `clearvoice`**: the `numpy<2` conflict that forced
the two-venv design lives in that package's metadata, not in the model, which
needs only torch, einops and rotary-embedding-torch. Your voice-alteration
complaint is not chunk artefacts, it is `sepformer-whamr16k` doing the
dereverberation it was trained to do, and it goes away with the checkpoint. And
stop trying to reach silence with a mask — but not, as this document first
argued, by gating on the stream-to-mixture ratio. That was wrong for a reason
worth keeping: your clip has a third man who is never on screen, an AV-TSE
extractor emits *somebody* on every frame, and an off-screen intruder is loud,
clean and genuinely present in the mixture, so every energy statistic waves him
through. The question is not how much energy this is, it is **whose voice this
is** — ask it directly, with speaker embeddings, and let the mixture arbitrate
how many people exist (§4a).

---

## 12. Addendum 2026-09-24 — the team's phone recording, on the RTX 5060

The showcase clip is an iPhone recording (4K HEVC, 60 fps, mono 48 kHz) of two
people **reading two different texts at the same time, continuously**, with
similar voices. It broke three assumptions this document relied on. Every
number below is Whisper `large-v3-turbo` WER of each channel against the read
script (face 0 / face 1), on `runs/a49b343d1d24` (20 s) and
`runs/9f06afeda8fc` (37 s), plus "intruders": words from the *other* script.

| output | 20 s clip | 37 s clip |
|:--|:--|:--|
| SepFormer, v2 web default | 80 / 100 %, both texts interleaved | 110 / 128 %, 8 / 14 intruders |
| AV-TSE, 12 s chunks, identity gate (v3) | 54–78 % of each channel **muted** | same |
| AV-TSE, 12 s chunks, ungated | 0 / 24 %, 4 intruders | 2.3 / 15.2 %, 6 intruders |
| **v4: 2 s windows + refinement + word-safe gate** | **4.3 / 24.1 %, 0 intruders** | **3.5 / 5.4 %, 0 intruders** |

(The 20 s clip's face-1 script is partly reconstructed, so its floor is not 0.)

1. **Blind separation cannot split these two voices.** SepFormer on the whole
   clip in one call still interleaves both texts in both channels. It is not a
   chunk-alignment bug; only the lips can tell these speakers apart. The web
   default is now `separator="avtse"`.
2. **Window length was out of distribution.** Upstream decodes in 3 s windows
   (`one_time_decode_length: 3`); 12 s windows made both faces' outputs
   near-copies of the mixture on continuous overlap (ECAPA same-instant cosine
   between channels 0.86, where the talk show reads 0.12). 2 s windows
   (`chunk_s=1.0, context_s=0.5`, frame-aligned, batched) are best on every
   clip, including the talk show.
3. **One refinement pass** re-extracts each face from the mixture minus the
   other faces' estimates, protecting the part of those estimates that is
   phase-coherent with this face's own (its own voice, leaked across). Plain
   subtraction cancelled the target (face 0: 2.3 → 30.2 % WER).
4. **The identity gate is off by default.** ECAPA could not separate the two
   on-screen voices on this recording (leave-one-out 45–53 %), picked k=5 and
   muted most correct speech; on the talk show its k flips with chunk length
   (4 vs 0 mixture windows against a threshold of 3). `--identity-gate` still
   enables it for clips with an off-screen speaker. **Off-screen speakers are
   therefore not removed by default** — a lips-still veto was tried and never
   fires on the talk show's intruder spans.
5. **The v2 silence gate cut words** (12–18 % of word time muted). The AV path
   now gates at −30/−40 dB with a 120 ms hold: 0.05–1.6 % of word time muted.
   Between-word gaps on this clip are not silence — they hold the other reader
   about 7–10 dB below this one — so no gate can zero them without cutting
   speech, and a coherent leak canceller that lowered them 2–4 dB raised WER
   on every channel, so it is not shipped.

Runtime on the RTX 5060 after capping the working video at 1080p and TF32 for
AV-TSE: 55 s for the 20 s clip, 89–97 s for the 37 s clip.

---

## 13. Addendum 2026-09-24 (later) — strict isolation, re-identification, other videos

**Ground-truth leak bench.** Leakage cannot be measured on the showcase clip
directly (both people read throughout, so there is no clean reference), and
Whisper WER hides it (Whisper follows the louder voice). So a bench was built
from the clip itself: face 0's separated channel plus face 1's, shifted 18 s so
the words are unrelated, each paired with its own lips; and a "same speaker"
case mixing face 0 with himself. It measured the refined AV-TSE output at only
**10.4 / 8.1 dB** channel-to-interferer — the "huge leaks" the team heard.

**What was tried and measured on it:**

| approach | result |
|:--|:--|
| Dolphin (ICLR 2026, 11 M params, weights on HF) | worse: 6.5–8 / 0–3 dB. Its landmark-aligned mouth crop assumes a frontal face; face 1 reads looking down |
| third AV-TSE pass on masked input | no gain |
| averaging over mouth-crop augmentations | no gain |
| **coherent leak cancellation** (`dsp.strict_isolation`) | **16.3 / 13.4 dB**; WER cost small (37 s clip 3.5→5.8 %) |
| + cross-face ratio mask | +2 dB on the bench, large WER cost, and *worse* on the talk show — not used |

**Strict / natural.** Both versions ship in one multichannel WAV (strict in
channels [0, N), natural in [N, 2N), `tracks.json.modes`), and the player's
"Isolation" button switches between them with the usual crossfade. The silence
gate always listens to the natural channels (gating the strict ones on their
own audio cut 2–17 % of word time).

**Other videos.** The IoU tracker stopped creating tracks after four and could
not follow a face across a camera cut. `app/reid.py` adds SFace/YuNet
appearance re-identification (OpenCV, ~39 MB of weights): a 196 s edited talk
show went from 0.2–1.6 % coverage per person to 44 / 18 / 8 %, with identities
consistent across cuts. Faces off screen for more than 0.48 s get a frozen
mouth, are muted in their channel, are kept out of refinement, and their
windows are not run at all (196 s clip: 436 s → 210 s). A CNN split-screen
debate (man and woman talking over each other) separates cleanly: channel
cosine 0.08, and the man's channel recovers words that are inaudible in the
mixture.

**Honest limit.** Two similar voices reading continuously over each other is
the hardest case there is. Strict mode leaves the other reader roughly 13–16 dB
down during speech (much lower in pauses). No available open model did better
on this clip.

## 14. Addendum 2026-09-25 — noise, and adapting the model to the team

**The problem.** The final demo may be recorded live, on a phone, in a noisy
room. Real noise was mixed into the 37 s showcase take (café = babble + room
noise, babble = other people talking, room = fan rumble, hiss and mains hum),
and the released AV-TSE model degraded fast. Whisper WER, face 0 / face 1:

| condition | WER |
|:--|:--|
| clean | 3.5 / 5.4 % |
| café 20 / 15 / 10 / 5 dB | 9.3 / 9.8 · 12.8 / 16.3 · 34.9 / 32.6 · 75.6 / 62 % |
| babble 15 / 5 dB | 26.7 / 27.2 · 72 / 62 % |
| room 5 dB | 17.4 / 21.7 % |

**What did not work,** measured on a noisy ground-truth bench (the §13 remix
plus known noise) and on WER:

| approach | result |
|:--|:--|
| speech enhancement *after* separation (SepFormer-DNS4, MossFormer2-SE-48k, FRCRN) | best SIR/SNR on the bench, but WER rose on every channel |
| enhancement *before* separation | destroyed the separation (the model needs the voices' fine structure) |
| strict canceller in noise | WER rose |
| lip-motion voice activity from the mouth crops | AUC 0.5–0.6 on the phone clips: no signal |
| DeepFilterNet | needs a Rust toolchain on Windows; not installable here |

**What worked: speaker/room adaptation** (`app/adapt.py`). Fine-tune the last
4 of 24 transformer layers plus the output heads (10.1 M of 68.5 M params) on
a calibration recording of the same people:

- targets: the model's own separated channels (pseudo-targets);
- interferer: the other face's channel from a different moment;
- noise: babble, synthetic room noise, recorded venue tone, 0–25 dB SNR;
- bf16 autocast, AdamW 3e-5 with OneCycle, SI-SNR loss, batch 4, 800 steps;
- about 15 min at the GPU's 55 W default; 30 min at the 24 W cap this laptop
  was running at.

The result is a 40 MB adapter that loads over the released weights. Trained
on the 20 s take and tested on the 37 s take it never saw, WER through the
same path:

| condition | released | adapted |
|:--|:--|:--|
| clean | 3.5 / 5.4 % | 3.5 / 7.6 % |
| café 15 dB | 10.5 / 21.7 % | 3.5 / 5.4 % |
| café 10 dB | 37.2 / 34.8 % | 25.6 / 15.2 % |
| room 5 dB | 17.4 / 22.8 % | 11.6 / 17.4 % |
| babble 15 dB | 37.2 / 25.0 % | 4.7 / 13.0 % |

On the ground-truth remix of the held-out take (babble here comes from voices
never used in training), adaptation also leaks less:

| metric (face 0 / face 1) | released | adapted |
|:--|:--|:--|
| SIR natural | 10.4 / 8.1 dB | 11.2 / 10.4 dB |
| SIR strict | 16.3 / 13.4 dB | 16.9 / 16.0 dB |
| other voice in pauses, strict | −25.6 dB | −30.3 dB |
| room noise 5 dB: output SNR | 14.5 / 12.0 dB | 20.7 / 19.3 dB |

**It is speaker-specific.** On three strangers (sports clip, same bench) the
adapter *lowered* strict SIR from 24.0 / 25.8 to 20.3 / 17.9 dB. So an adapter
stores the SFace embeddings of its training faces. A job applies it per face,
and only to faces with cosine ≥ 0.5 to one of them (`adapter_face_match`).
Across seven clips, the same person in two takes measured 0.86–0.96 and
different people at most 0.34. Every other face runs on the released weights
in the same job (`AVTSESeparator.separate_for_faces(adapted=)`).

**A recipe detail that mattered.** A variant that also trained on 15 %
target-only examples, meant to mimic turn-taking, taught the model to pass
whatever voice it hears. Pause leak on the held-out take got worse than the
released model: −17.3 dB against −20.5. Every example now has an interferer.

**Throughput note.** During this work the laptop's power mode capped the RTX
5060 at 24 W (default 55 W): SM clock 750 MHz under load, and training at half
speed. Inference is affected the same way. Set the laptop to its performance
mode, on mains power, before a live demo.
