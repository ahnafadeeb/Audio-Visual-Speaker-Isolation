# Research: Gating & Fusion

Status: **DONE** — 2026-08-11. Objective A (hard mute).

Every number below is produced by `scripts/gating_knee.py` on this box and
reproducible with `PYTHONPATH=. python scripts/gating_knee.py`. The signals are
synthetic voiced-speech surrogates, not real SepFormer output — so the
*mechanism* and the *ordering* are what transfer; the exact dB at which the knee
falls will shift on real audio.

---

## 1. Why the three previous fixes failed — one shared cause

The absolute lip-position gate, the lip-velocity gate, and trusting raw
SepFormer all failed for the same structural reason: **each used a single
modality to answer a question that modality cannot answer.**

- Absolute lip position: a speaker resting with their mouth open is
  indistinguishable from one mid-vowel. (Also measured against the
  crop-normalised ruler — see `vision.py`.)
- Lip velocity alone: nodding, chewing, and tracker jitter all produce motion.
- Raw SepFormer: the residual is *speech*, so no spectral test separates it
  from the target.

The pattern: every one of them had to choose a threshold that traded
bleed-through against cutting real speech, on an axis where the two overlap.

---

## 2. The knee, and proof it is not a tuning problem

`scripts/gating_knee.py` instruments what the Schmitt trigger actually sees.
`rel max` is the residual's peak frame energy relative to the stem's **own**
95th-percentile — the exact quantity compared against `open_db` (−30 dB).

| input leak | residual | rel max | > open? | acoustic zeros% |
|-----------:|---------:|--------:|:-------:|----------------:|
| −18 dB     | −68.6    | −50.6   | no      | **100.0**       |
| −15 dB     | −59.7    | −41.6   | no      | **100.0**       |
| −12 dB     | −51.0    | −32.6   | no      | **100.0**       |
| −9 dB      | −42.5    | −23.6   | **YES** | **70.0**        |
| −6 dB      | −34.4    | −14.6   | **YES** | **23.4**        |
| −3 dB      | −27.0    | −5.6    | **YES** | **3.4**         |

Two things this establishes:

1. **The collapse coincides exactly with `rel max` crossing `open_db`.** Not
   approximately — the first row where the comparison flips is the first row
   where zeros leave 100%. The gate is not misbehaving; it is doing precisely
   what it was told, on evidence that has become ambiguous.
2. **The slope is ~3:1.** 3 dB more input leakage → ~9 dB more residual, because
   a p=2 Wiener mask on estimates suppresses roughly as the cube of leak
   amplitude. So the residual chases `open_db` three times faster than the
   separator degrades.

**Why no threshold fixes it.** Lowering `open_db` to −45 dB would restore the
zeros at −9 dB leakage — and would then also close on genuine quiet speech,
because real sibilants and utterance tails sit 20–30 dB below vowel peaks. That
is the *same* trade the three earlier attempts made, arriving from a different
direction. The acoustic axis has no operating point where both hold.

---

## 3. The fix: shift the threshold, don't lower it

Vision is independent of acoustic level, so it can break the tie that gain
staging cannot. The fusion does **not** hard-AND a visual VAD with the gate —
that would make a tracking failure into a mute. It biases the thresholds:

```
effective_open  = open_db  + visual_veto_db * (1 - v)
effective_close = close_db + visual_veto_db * (1 - v)
```

Four properties, each deliberate:

- **Both thresholds shift together**, so the hysteresis band width is preserved
  and the anti-chatter behaviour is unchanged.
- **`v` is continuous, not binary.** Weak evidence of motion produces a partial
  shift rather than a hard decision.
- **Where vision is invalid the shift is zero** — the gate degrades exactly to
  its pure-acoustic behaviour. Verified bit-identical (`check_fusion.py`).
- **`visual_veto_db = 25`** is sized off the measurement: it must exceed the
  ~15 dB by which the residual overshoots `open_db` at −6 dB leakage, with
  margin.

`v` is built from the **derivative** of lip aperture — same rationale as the
matcher, and immune to the resting-open-mouth failure that killed attempt #1.

### The dilation that made it free

The first working version held silence at 100% but cost 3.6% of the target's own
speech (97.7% → 94.1% retention). That is the old trade reappearing, and it
would have been the fourth failure.

Cause: the mouth is briefly still *mid-utterance* — sustained vowels, nasals —
and at 25 fps one still frame is 40 ms of audio. Fix: symmetric 250 ms dilation
of the activity signal (`visual_hold_ms`), the visual analogue of `min_off_ms`.
Backward dilation is also physiologically correct — lips begin moving *before*
voicing (anticipatory coarticulation), so the veto must lift ahead of the onset.

**Result — the fusion is free:**

| input leak | acoustic zeros% | acoustic kept% | **AV zeros%** | **AV kept%** |
|-----------:|----------------:|---------------:|--------------:|-------------:|
| −12 dB     | 100.0           | 97.7           | **100.0**     | **97.7**     |
| −9 dB      | 70.0            | 97.7           | **100.0**     | **97.7**     |
| −6 dB      | 23.4            | 97.7           | **100.0**     | **97.7**     |
| −3 dB      | 3.4             | 97.7           | **100.0**     | **97.7**     |

Silence restored to 100% across a 9 dB range where the acoustic gate collapses
to 3%, at **exactly zero** cost to target retention — identical to the
pure-acoustic gate, digit for digit.

---

## 4. Failure modes — vision abstains, it never vetoes on absence

The governing rule: **"face not visible" is not "not speaking."** Conflating
them mutes a speaker who turns their head. Measured at −9 dB leakage:

| case                    | zeros% | kept% | behaviour |
|:------------------------|-------:|------:|:----------|
| vision ok               | 100.0  | 97.7  | fused |
| A's face gone 1–3 s     | 100.0  | 97.7  | holds ≤200 ms, then abstains |
| A's face never found    | 70.0   | 97.7  | **falls back to acoustic** |
| A's track frozen        | 70.0   | 97.7  | **falls back to acoustic** |
| lips swapped (mismatch) | 70.0   | 57.3  | detected — see §5 |

Rows 3 and 4 lose the benefit and keep the old behaviour. That is the correct
direction to fail: a frozen tracker and a silent face are indistinguishable, so
the code reports *invalid*, not *silent*.

Enforced at three levels, each independently checked:
- `resample_hold` returns a **validity mask** alongside values. NaN gaps ≤200 ms
  hold; longer go invalid.
- Extrapolation past the last video frame is invalid — `np.interp` pads with
  0.0, which reads as "mouth perfectly still", so a video one frame shorter than
  its audio would otherwise have vetoed the last word.
- `visual_activity` abstains wholesale when a track's motion never exceeds
  `visual_min_dynamic_range`.

---

## 5. Bonus: a runtime mis-assignment detector

`matching.py` documents a failure with "no principled runtime detector": a
mis-assignment means the user clicks face A and hears speaker B. The fusion
produces one for free.

`dsp.veto_cost()` measures what fraction of the *acoustic* gate's own output the
visual veto removed. Correct pairing → the veto only acts where the speaker is
already silent, so it removes ~nothing. Wrong pairing → vision and audio
disagree constantly and it eats the target.

| leak | correctly paired | swapped |
|-----:|-----------------:|--------:|
| −12  | 0.000            | — |
| −9   | 0.015            | **0.405** |
| −6   | 0.102            | — |
| −3   | 0.227            | — |

**One confound had to be removed.** The first version read 0.256 at −6 dB and
0.307 at −3 dB on a *correct* pairing — it would have cried wolf on a healthy
clip. Cause: when the separator is bad the acoustic gate is itself open on the
other speaker's residual, and the veto correctly kills it — the fusion working,
counted as loss. Excluding samples where a competing stem is strong drops those
to 0.102 / 0.227 while the swap rises to 0.405.

`visual_disagree_alarm = 0.30` sits in the middle of that gap. It logs a
warning and clears the track's `reliable` flag, so the UI shows a caveat instead
of being confidently wrong. **Caveat:** the −3 dB row (0.227) is close to the
threshold. That is a separator which has essentially failed, and the detector's
margin there is genuinely thin.

This is a better signal than the matcher's Hungarian margin because it is
computed on the *final* signals rather than on the statistic the assignment was
chosen to maximise.

---

## 6. What changed, and the ordering constraint

- **`app/dsp.py`** — `resample_hold`, `visual_activity`, `_dilate`, `veto_cost`;
  `gate_mask` takes per-frame threshold arrays; `apply_gate` takes `lips`.
- **`app/channels.py`** *(new)* — `lips_by_stem` / `plan_channels`. Extracted
  from `pipeline.py` specifically so they are testable with **no** heavy
  dependencies. Three index spaces meet here (track / stem / channel) and
  mixing them never raises — it just plays the wrong voice.
- **`app/vision.py`** — `resample_lip` is now a thin wrapper over
  `resample_hold`. One implementation of the NaN contract, not two.
- **`app/config.py`** — `visual_fusion`, `visual_veto_db`, `visual_smooth_ms`,
  `visual_hold_ms`, `visual_min_dynamic_range`, `visual_disagree_alarm`.
- **`app/pipeline.py`** — **`match` now runs before `gate`**, because the gate
  needs to know whose face is whose. Matching still consumes `stems_raw`
  (ungated) — correlating against gated audio would feed the matcher a signal
  the gate already shaped, which is circular. `meta.json` gains `visual.*`.
- **`scripts/check_fusion.py`** *(new)* — 31 contract checks, numpy+scipy only.

**`stems_raw.wav` is untouched by any of this.** It is written pre-gate, so
SI-SDR/PESQ scoring is unaffected — the metrics path and the demo path still
provably come from the same run, and `visual_fusion: false` reverts to the pure
acoustic gate.

---

## 7. Answered by `redteam:silence-claim`

These were deliberately not claimed here, because the bench is synthetic. They
have since been attacked directly — see **`docs/REDTEAM_SILENCE.md`** for the
numbers.

| open question | outcome |
|:--------------|:--------|
| Sibilants vs `open_db = −30` on a **global** `ref_percentile` | **survived** — 100% kept down to −35 dB |
| The 90th-percentile normalisation vs dropout-heavy tracks | **partly** — 8 hostile inputs survived; realistic tracker noise remains untested |
| `min_on_ms = 60` vs turn-taking and backchannels ("mhm") | **survived** — 150–400 ms at −12…−20 dB, 100% kept |
| Reverb tails vs the 250 ms visual hold | **survived** — 2 closings for 2 utterances at RT60 0.3/0.6/1.0 s |

One real finding came out of it, and it sharpens §2 rather than contradicting
it. The global reference bounds intra-clip dynamic range at exactly
`open_db`: a passage 30 dB below the clip's loudest moment *and* surrounded by
silence is lost. Both fixes measured worse — a windowed reference re-opens the
gate on pure residual (0% → 100% leak at −35 dB), and lowering `open_db` trades
the two failures 1:1, flipping quiet-speech survival and residual rejection
within one 2 dB step of each other. **§2's "no operating point where both holds"
is now measured, not argued.** No config changed.

