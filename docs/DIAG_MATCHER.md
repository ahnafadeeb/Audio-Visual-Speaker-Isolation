# Diagnosis: the perceived speaker swap

Status: **DONE** — 2026-08-11. Attribution of the four-item field bug report
against `runs/862bd92a01ac` (32.30 s, 25 fps, 1920×1080, two-speaker Bengali
talk show: a man screen-left, a woman screen-right).

Reproduce, in order:

```bash
PYTHONPATH=. python scripts/diag_sel.py runs/862bd92a01ac
PYTHONPATH=. python scripts/diag_feat.py runs/862bd92a01ac
PYTHONPATH=. python scripts/diag_rigid.py runs/862bd92a01ac
PYTHONPATH=. python scripts/diag_win.py runs/862bd92a01ac
```

Then the two scripts written later, which check this document's own bookkeeping
rather than the clip — `diag_chain.py` closes the whole stem → channel → track
chain by measurement, and `diag_sep_repro.py` establishes that the separator is
deterministic, which is what makes any of it reproducible:

```bash
PYTHONPATH=. python scripts/diag_sep_repro.py runs/pair_v2
PYTHONPATH=. python scripts/diag_chain.py runs/pair_v2
```

`runs/pair_v2` is a fresh full-pipeline run of the same upload and is
**byte-identical** to `runs/862bd92a01ac` in `video.mp4`, `stems_raw.wav` and
`stems_demo.wav`; either job answers to either script.

**Verdict: the symptom is real and reproducible, but three of the four
diagnoses in the report are wrong, and the actual defect is in neither
`separation.py` nor the visual tracking.** The stem-to-face matcher is operating
at chance on this clip, it landed on the wrong side of the coin, and it reported
a confidence that cannot detect this because the confidence grows with smoothing.

| # | reported cause | verdict |
|:--|:---------------|:--------|
| 1 | chunk-boundary permutation inversion at ~10 s | **refuted** — `flipped: 0`, no per-chunk pitch swap |
| 2 | profile-face / yaw > 45° tracking dropout | **refuted** — boxes 809/809 and 755/809, head motion explains R² 0.022 |
| 3 | AV-VAD can drive a hard mute | **refuted** — the lip signal cannot tell who is speaking (p 0.49–0.96) |
| 4 | female stem collapsed onto the male channel | **symptom confirmed, mechanism wrong** — the separator did split the two voices; the *matcher* inverted them |
| — | *(not reported)* | **the real defect** — matcher at chance + uncalibrated confidence |

---

## The symptom, stated as a measurement

Three premises, each closed independently.

**P1 — track 0 is the man.** Box centroid x = 0.301 (screen-left) against track
1's 0.755, and the annotated frame `runs/_diag/862bd92a01ac_id0.png` shows the
man in glasses inside track 0's box at t = 4.72 s.

**P2 — stem 0 carries the man's voice.** Labelled from the *mixture's own* f0
with a 300 Hz ceiling, so neither the separator nor the matcher is in the loop
(a 400 Hz ceiling admits the octave-doubled band and was what wrecked every
earlier attempt at this; at 300 Hz only 7.7% of frames are octave-ambiguous).
38 LOW frames (median 146.1 Hz) and 106 HIGH frames (median 221.3 Hz), with the
165–190 Hz overlap band excluded:

| signal | LOW dB | HIGH dB | contrast |
|:-------|-------:|--------:|---------:|
| mixture (baseline) | −14.5 | −12.0 | −2.44 |
| stem 0 | +0.3 | −25.8 | **+26.10** |
| stem 1 | −3.5 | +5.4 | −8.91 |

Paired and scale-free: stem 0 is the louder of the two in 55.3% of LOW frames
and 11.3% of HIGH frames, difference **+0.439, 95% CI [+0.270, +0.609]**; the
level swing is **+21.52 dB, 95% CI [+11.33, +31.02]**. Both intervals exclude
zero, so the separation genuinely splits the two speakers.

**P3 — the shipped pairing is inverted.** `assignment: [1, 0]` gives track 0
stem 1, so channel 0 — the man's button — selectively carries the HIGH voice
(contrast −16.83 dB, paired difference −0.439, CI [−0.609, −0.272]). Correct is
`[0, 1]`.

That is verbatim the field report: *"selecting Male played Female audio"*, and
*"the female speaker's stem collapsed onto the male channel"* is literally true
— the female stem *is* on the male channel. `"selecting Female gave complete
silence"` follows: channel 1 carries stem 0, which is 8–9% voiced, holds 13–38%
of the two stems' joint energy, and is 73.9% exact zero after gating.

---

## Why: the audio-visual correlation carries no identity information

`matching.py` scores `corr(|Δ lip_i|, env_dB_j)` and resolves it with the
Hungarian algorithm. The obvious suspicion — that a dB energy envelope rewards a
stem for being loud whenever *anyone* speaks — is correct as far as it goes, but
it is not the fix, because nothing on the audio side repairs it.

**Fifteen audio features all invert** (`diag_feat.py`). Seven absolute: dB
(shipped), linear RMS, dB minus the cross-stem mean, power share
`10log₁₀(p_j/Σp_k)`, share on loud frames only, share on the pre-mask stems, dB
on the pre-mask stems. Eight differential — `corr(lip₀−lip₁, feat₀−feat₁)`,
which cancels common mode on the audio side *and* the lip side. Every one
returns `[1, 0]`, with |r| ≤ 0.09.

**Rigid head motion is not the contaminant** (`diag_rigid.py`). Translation,
scale and yaw together explain **R² = 0.022** of the shipped lip signal on both
tracks. No motion proxy follows either voice above 1.4σ in a
difference-in-differences on per-track percentile ranks. Four alternative lip
features — vertical inner-lip gap normalised by interocular; the same normalised
by a vertical face span, which does not foreshorten under yaw; a Procrustes
alignment to a per-track rigid reference; and the shipped feature with head
motion regressed out — all still invert.

**Time scale flips the answer, and then a null kills it** (`diag_win.py`).
Smoothing the lip feature over 200 ms *before* differencing does produce the
correct `[0, 1]`. But a circular-shift null — roll the lip signal, recompute —
gives **p = 0.888–0.963** for every such cell, the shifted-lip baseline lands on
the correct pairing **≈50%** of the time, and split-half disagrees in **all
eight** configurations tested. The correct answer at 200 ms is a coin landing
the other way up, not a signal.

**What the visual side does carry**, and it is exactly the wrong thing:

| target | r (track 0, man) | p | r (track 1, woman) | p |
|:-------|-----------------:|--:|-------------------:|--:|
| mixture | **+0.1804** | 0.070 | **+0.1867** | 0.036 |
| stem 0 (man) | −0.0967 | 0.389 | −0.0237 | 0.766 |
| stem 1 (woman) | +0.1448 | 0.089 | +0.0309 | 0.812 |

Both faces track *total speech activity* — to three decimal places equally —
and neither tracks its own voice. On a talk show both mouths move constantly:
reacting, interjecting, back-channelling. The faces are 162×162 px and 127×127
px in a 1920×1080 frame, so the inner-lip ring is roughly 20×11 px and its
shoelace *area* is quadratic in landmark noise. What survives that noise floor
is the common component, which is the one component that cannot discriminate.

---

## The second defect: `confidence` is not calibrated

The margin `matching.py` reports grows monotonically with smoothing while the
answer stays wrong, because the statistic divides by `n_frames` as though
smoothed frames were independent samples:

| integration window | reported confidence | answer |
|-------------------:|--------------------:|:-------|
| 40 ms (the shipped window) | 0.0679 | wrong |
| 200 ms | 0.1870 | wrong |
| 600 ms | 0.4125 | wrong |
| 1600 ms | **0.6654** | wrong |

(Recomputed by `scripts/diag_win.py` on the pipeline's own decode, from
landmark-derived lip features against post-gate stems. `meta.json` stored 0.0376
for the same job: the pipeline correlates against *raw* stems and uses
`tracks.json`'s lip signal. Both are the 40 ms window; the 1.8× spread between
two reasonable implementations of the same statistic is itself a comment on how
much weight it can bear. What matters is the 9.8× rise across the sweep while the
answer never changes.)

`min_confidence = 0.05` therefore cannot protect the user: a wider kernel would
have shipped the same wrong pairing with 13× the confidence. The shipped 0.0376
*was* below the threshold and `reliable: false` was stored correctly — and the
user still heard a confident wrong answer, because nothing downstream acted on
it.

### How little that margin is worth, measured

The statistic was then run on **pure noise** — two z-scored white-noise "lip"
signals against two z-scored white-noise "envelopes", 800 frames, 12 independent
draws (`scripts/check_fusion.py: check_null_calibration`):

| smoothing | median margin on PURE NOISE | median p | p ≤ 0.05 |
|----------:|----------------------------:|---------:|---------:|
| 1 frame (40 ms, shipped) | **0.0683** | 0.296 | 1/12 |
| 41 frames (1640 ms) | **0.1922** | 0.662 | 1/12 |

Pure noise at the shipped window scores **0.0683**. The real clip scored
**0.0679**. The confidence the pipeline has been reporting is numerically
indistinguishable from a random number generator, and it sits *above*
`min_confidence` in both cases — so the threshold admits noise by construction,
not by bad luck on this clip.

The p-value survives the same test: its false-positive rate stays at 1/12 ≈ 8%
against a nominal 5% at **both** windows, i.e. it does not inflate when the
margin inflates 2.8×. A positive control (irregular turn-taking, each lip signal
tracking its own envelope) reaches the p floor of 1/(n+1) and recovers the true
pairing, so the null is not simply refusing everything.

One number that must NOT be over-read: `null_agreement`. For a 2×2 problem a null
that has destroyed the alignment picks either pairing about half the time, so
~50% is the null's own generic value — the positive control above also scores 51%
while being perfectly correct. It is diagnostic only when *high*, which would mean
the pairing survives destroying the alignment and is therefore decided by some
shift-invariant asymmetry rather than by who is speaking. An earlier draft of this
document read the real clip's ~50% as evidence of chance; it is not evidence
either way, and `p ≈ 0.9` is what carries that verdict.

### The clip, re-measured on the decode the pipeline actually ships

Two corrections to how the numbers above were obtained, both found while wiring
the p-value in.

**The diagnostics were reading the wrong video.** A job directory holds
`input.mp4` (the upload, 29.98 fps) and `video.mp4` (what `normalize_video`
produces at `vision.target_fps = 25.0`, and the only one `FaceAnalyzer` ever
sees). `diag_match.py`, `diag_visual.py` and `fit_gate.py` all preferred
`input.mp4` — 968 frames instead of the shipped 809. Fixed in
`scripts/_job.py: pipeline_video()`. The attribution numbers are unaffected:
`diag_pose.py`, which caches the landmarks that `diag_feat.py` and `diag_win.py`
consume, always read `video.mp4`.

It also settles a discrepancy this document previously left as "input-side
differences": the harness reporting 0.0679 against `meta.json`'s 0.0376 was two
different decodes of one clip.

**Vision itself is bit-reproducible**, so nothing here is detector noise.
`scripts/diag_repro.py` runs the identical analysis three times in one process:
landmark counts, SHA-256 of every lip signal, assignment, margin and p are
identical across all three.

> **Correction (2026-08-12).** Two claims that stood here have been withdrawn,
> and they failed the same way — a number from one index space compared against a
> number from another. See *The index-space audit* below for the full accounting.
>
> * **Withdrawn: "the two decodes return opposite pairings — `[1, 0]` at 25 fps,
>   `[0, 1]` at 29.98,"** offered as "the most compact demonstration in this
>   document that the pairing is a coin flip." It is not a demonstration of
>   anything. `meta.assignment` is track → **stem**; `diag_match.py` reads
>   `stems_raw.wav`, which `plan_channels` has already reordered, so its output is
>   track → **channel**. The two numbers were never comparable. Re-measured with
>   both decodes in **one** space (`diag_match.py runs/pair_v2 --video
>   …/input.mp4`, same stems file), they agree: **both give `[0, 1]`.** The frame
>   rate of a transcode does not decide who is speaking.
> * **Withdrawn: "`meta.json` says `[1, 0]`, current code gives `[0, 1]`, so the
>   artifact predates a code change."** Same error. A fresh full-pipeline run
>   (`runs/pair_v2`) is **byte-identical** to the stored one in `video.mp4`,
>   `stems_raw.wav` and `stems_demo.wav`, and stores the same `assignment: [1, 0]`
>   and `confidence: [0.0376, 0.0376]`. There was no code change and no drift.
>
> What the two decodes *do* show is worth more than the claim it replaces. The
> **decision** is stable; the **confidence attached to it** is not:
>
> | decode | frames | pairing | margin | p |
> |:-------|-------:|:--------|-------:|----:|
> | `video.mp4`, 25 fps (shipped) | 809 | `[0, 1]` | 0.0438 | 0.7132 |
> | `input.mp4`, 29.98 fps | 968 | `[0, 1]` | **0.1765** | **0.1646** |
>
> A transcode moves the margin by 4× and the p-value by more than 4×. At 29.98 fps
> the margin alone clears `min_confidence = 0.05` and would have been reported as
> trustworthy; only the null-calibrated p-value still declines. That is the
> strongest argument in this document for the p-value guard being load-bearing
> rather than decorative — and it survives the withdrawal above intact.
>
> (`p` also carries Monte-Carlo error: the pipeline stored 0.7307 and the harness
> 0.7132 for the same decode, from 400 random shifts. Two decimal places are
> meaningful; three are not.)

Authoritative current numbers, `video.mp4`, `stems_raw.wav`, current code. **Every
row names its index space**, because the rows above this one did not and that is
how two wrong claims got in:

| quantity | value | space | verdict |
|:---------|------:|:------|:--------|
| `meta.assignment` | `[1, 0]` | track → stem | **inverted** (P3) |
| `tracks[].channel` | `[0, 1]` | track → channel | the same decision, renumbered |
| margin | 0.0438 | — | below `min_confidence = 0.05` |
| p-value | **0.7307** | — | 73% of shifted-lip nulls do as well |
| null agreement | 49.3% | — | the baseline; not evidence either way |

Those first two rows are the trap in one place: `[1, 0]` and `[0, 1]` are the same
decision, and it is the **wrong** one. `plan_channels` reorders the export so
channel *i* belongs to track *i*, which makes the shipped binding always look like
the identity no matter what the matcher decided. A reader who compares
`assignment` against `channel` and finds them "opposite" has learned nothing about
the audio.

So the matcher lands on the wrong answer and correctly declines to claim it: both
guards fire, `reliable` is false for both tracks, and the pipeline logs that the
channel labels should not be trusted on this clip. That is the honest outcome for
a coin flip — not a fix, and it is why the swap control exists.

---

## The index-space audit

Three claims in earlier drafts of this document were wrong, and all three failed
the same way, so the failure is worth naming once and for all rather than
correcting three times.

There are **three** index spaces and **two** permuting hops between them:

```
stem     what the separator emitted             0..n_stems-1
  |
  |  hop 1 — plan_channels()  (app/channels.py:131)
  |          reorders the export so channel i belongs to track i.
  |          Applied at app/pipeline.py:242-246 to stems_raw, stems_demo, veto.
  v
channel  a column of stems_demo.wav / stems_raw.wav   0..n_ch-1
  |
  |  hop 2 — tracks.json's `channel` field
  |          what the browser binds its buttons to.
  v
track    a detected face, the order the buttons are drawn in  0..n_tracks-1
```

Hop 1 is the trap. Because it reorders the export *by track*, the shipped binding
looks like the identity no matter what the matcher decided. So on this clip:

| artifact | value | space |
|:---------|------:|:------|
| `meta.assignment` | `[1, 0]` | track → stem |
| `tracks[].channel` | `[0, 1]` | track → channel |

**These are the same decision printed as opposite permutations.** Comparing one
against the other looks like finding a disagreement and is finding nothing. It
produced:

1. This document's *"the two decodes return opposite pairings"* — withdrawn above.
2. This document's *"`meta.json` says `[1, 0]`, current code gives `[0, 1]`"* —
   withdrawn above. `diag_match.py` reads `stems_raw.wav` **from disk**, which is
   post-`plan_channels`, while the pipeline matches *pre*-reorder. The harness even
   labels its columns "stem 0 / stem 1" while reading a channel-ordered file.
3. A claim in `scripts/check_swap.py` and `app/static/app.js` that the digit
   shortcut *"picked the wrong face on this clip"*. It did not: `tracks.json` ships
   `[0, 1]` whenever every track is matched, so a channel-indexed shortcut is
   correct as the clip ships. Both files now state the sharper true version — it
   breaks the instant the user presses Swap, which on this clip they must.

`scripts/diag_pose.py:235` hardcodes `TRUTH = [1, 0]`, and that one is **correct**:
it is in channel space (the man belongs on channel 1). Numerically identical to
`assignment`, semantically opposite, and nothing in either output said so.

### The chain, measured end to end

`scripts/diag_chain.py` exists so this never has to be argued again. It recovers
hop 1 by **cross-correlating the separator's live output against `stems_raw.wav`**
rather than reading `assignment`, so the report cannot inherit the bookkeeping
error it exists to catch, and it attributes each voice with the octave-safe YIN f0
that `diag_f0.py` established (plain autocorrelation and HPS both misled earlier
analyses by an octave). On `runs/pair_v2`:

| hop | measurement | result |
|:----|:------------|:-------|
| faces | box centroid x, confirmed by eye on frame 402 (t = 16.08 s) | track 0 = the man in glasses, 0.2966; track 1 = the woman in a pink hijab, 0.7569 |
| stems | YIN median f0 | stem 0 = **149.3 Hz**, 9.7% voiced → MALE; stem 1 = **225.5 Hz**, 34.4% voiced → FEMALE |
| hop 1 | cross-correlation | stem 0 → ch1 (r = +0.9355), stem 1 → ch0 (r = +0.9652); **`[1, 0]`**, consistent with `assignment` |
| channels | YIN median f0 | channel 0 = **213.9 Hz** → FEMALE; channel 1 = **147.4 Hz** → MALE |
| hop 2 | `tracks.json` | track 0 → channel 0 → the woman's voice |

**Verdict: `PAIRING INVERTED: 2 of 2 faces hear the wrong voice.`** Which is
verbatim the field report, now reproduced and attributed at every hop, and it
confirms P1–P3 were right all along. Independently corroborated by `diag_f0.py`
over the whole clip: channel 0 is 30.7% voiced at median 214.0 Hz, channel 1 is
7.9% voiced at 146.3 Hz.

### The separator is deterministic, so this is reproducible

The retracted idea that stem order might be *a property of the run* is worth
killing explicitly, because if it were true the swap control would be the only
honest interface and no fix could ever be verified. It is false.
`scripts/diag_sep_repro.py` builds a **fresh separator per pass**:

| quantity | pass 1 | pass 2 |
|:---------|:-------|:-------|
| mixture (32.30 s, 516783 samples) | `a687aed79036b467` | identical, and identical to the run's own `_probe.wav` |
| stem 0 | `891043cfa39ada75`, rms 1.153556 | identical |
| stem 1 | `355dce7e21d8652e`, rms 2.500978 | identical |
| cross-correlation | diagonal +1.000000, off-diagonal −0.033043 | **max abs diff 0.000e+00** |

Bit-identical across two passes in one process **and** across two separate
processes. And `runs/862bd92a01ac` and `runs/pair_v2` share `video.mp4`
`ab8cf8576804a171`, `stems_demo.wav` `9b517d7634fbec21`, `stems_raw.wav`
`48dc80721fd1bdc7` and the same `assignment: [1, 0]` — they are the same run.

So the inversion on this clip is a **fixed, reproducible** wrong answer, not a
coin that lands differently each time. One press of Swap always lands correctly
here. That is compatible with the matcher being *at chance* in the statistical
sense: the coin is weighted by nothing, but it is not re-flipped.

### The rule, for whatever is written next

Every number about pairing states its space, or it is not evidence. `diag_chain.py`
is the reference implementation of that rule, and lifting its attribution into the
pipeline (tracked separately) would make the pipeline self-diagnosing — it detects
this inversion with no ground-truth constant beyond which face is male.

---

## Items 1 and 2, closed

**Item 1 — chunk-boundary permutation.** `meta.alignment` records
`flipped: 0` across all 3 boundaries. Per-chunk median f0 of each stem over the
four chunk interiors, with the crossfade zones skipped and the octave-safe
300 Hz ceiling, shows no swap — a flip would trade the two columns:

| chunk interior | stem 0 | share | stem 1 | share |
|:---------------|-------:|------:|-------:|------:|
| 0.0–8.0 s | 132.4 Hz (n=77) | 13.0% | 242.2 Hz (n=109) | 87.0% |
| 10.0–16.0 s | 175.1 Hz (n=10) | 12.8% | 273.6 Hz (n=33) | 87.2% |
| 18.0–24.0 s | 150.5 Hz (n=13) | 37.8% | 187.9 Hz (n=80) | 62.2% |
| 26.0–32.3 s | too few | 18.8% | 194.9 Hz (n=121) | 81.2% |

The perceived "swap at ~10 s" is the *global* inversion plus the male stem
thinning out: stem 0 has 77 voiced frames in the first chunk and 10–13
thereafter, so male leakage inside the dominant stem becomes the loudest male
audio present, on the wrong button.

**Item 2 — profile-face dropout.** Boxes present 809/809 (track 0) and 755/809
(track 1); landmarks 774 and 755; lip NaN 4.1% and 6.7%. The *woman* — the one
whose channel went silent — has **less** yaw than the man (median 18.8° vs
28.4°) and **0%** of her frames exceed 45°. Bounding-box interpolation and a
pose fallback are worth having as robustness and are tracked separately, but
they cannot be the cause of a symptom that occurs while both boxes are present.

### The stitcher's margin was broken anyway — fixed, with no effect on this clip

`_align_permutation`'s margin was `best − trace`, which is **0.0 by construction
whenever the identity order wins** — and identity wins at nearly every boundary —
so `min_margin: 0.0` and `held_boundaries: []` could not distinguish a decisive
hold from a coin flip. Worse, `confident` was true *at* margin 0.0, so the
warning it exists to raise could never fire.

Fixed: the margin is now winner-minus-best-*feasible*-alternative (Murty's first
step, the same construction `matching.pairing_margin` uses), and `confident` is
`margin >= PERM_MARGIN` regardless of direction. Re-measured on this clip's own
stems at its own chunk bounds:

| boundary | waveform margin | decided by |
|---------:|----------------:|:-----------|
| 8.0 s | 1.9883 | waveform |
| 16.0 s | 1.9775 | waveform |
| 24.0 s | 1.9932 | waveform |

All three boundaries are decisive by a factor of 13 over `PERM_MARGIN = 0.15`.
Under the old formula all three reported **0.0**. So item 1 is refuted a second
way, from the stitcher's own side: it was never short of evidence here.

A cross-chunk **spectral identity anchor** was added for the case the waveform
cannot cover — an overlap window landing on a shared pause, where correlation
preserves identity in only 16/40 trials and one bad call propagates to the end of
the clip. It accumulates a level-invariant low-order real cepstrum per channel
over loud frames only, and may act only when *both* a pairing margin and an
absolute cosine floor clear thresholds calibrated against a null.

Two statistics are required because **neither works alone**, and the way the
first attempt failed is worth keeping:

* A margin threshold alone admits noise. Profiles are L2-normalised, so a profile
  accumulated from noise is a unit vector in a random direction and its 2×2
  margin is large *by amplification*: null median 0.287, p90 0.670, **max 1.383**
  over 400 draws — larger than the **0.573** a genuinely distinct pair of voices
  scores. No value of a margin threshold separates them. This is the same failure
  the AV matcher has, arriving from a new direction.
* A cosine floor alone admits acoustically **similar** speakers, whose cosine
  matrix is high everywhere. Measured on same-sex voice pairs: cosine clears any
  floor (0.52–0.76) while the margin collapses to 0.00–0.02.

Jointly (60 seeds × 4 SNRs × 3 similarity conditions): distinct voices →
**accuracy 1.00 in every cell where it acted**; similar and near-identical voices
→ **abstains 100%**; pure noise → **0/400 false actions**. At 0 dB residual SNR it
abstains 53% of the time and is still 100% correct when it acts.

**Its measured effect on this clip is nil, in both directions**, and that is
stated rather than glossed: the waveform is decisive at all three boundaries so
the rung is never reached, and if it *were* reached it would abstain — the two
stems' accumulated profiles give cosine `[[1.000, 0.929], [0.962, 0.995]]`,
margin **0.1040**, below the 0.15 threshold. The two stems on this clip are not
cepstrally distinct enough to anchor on, which is consistent with the separation
being lopsided (stem 0 holds 13–38% of joint energy, so much of it is leakage of
the same content). The mechanism is built, calibrated and tested; it is not a fix
for anything observed here.

One case it cannot detect, stated rather than papered over: two speakers the
anchor has never heard (a scene cut) score margin 0.292 and cosine 0.721 and it
will pick an order with no basis. Left alone because at such a boundary there is
no prior identity to preserve — holding is exactly as arbitrary as flipping.

---

## What follows

The matcher cannot be repaired by choosing a better correlation feature; there
is no signal on this clip to correlate. Three things follow, in value order:

1. **Calibrate the confidence against the circular-shift null** and let the
   pipeline act on it, instead of reporting an uncalibrated margin that
   smoothing inflates.
2. **Give the user a swap control.** Two speakers plus an at-chance matcher is a
   coin flip, so exactly one click fixes it, every time.
3. **The real fix is architectural** — AV-TSE conditioned on one face, emitting
   one stream, with no assignment step to get wrong (`ARCHITECTURE_V2.md` §3).
   `matching.py`'s own module docstring already anticipated this failure mode.
