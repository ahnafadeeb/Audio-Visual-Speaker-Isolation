# Red Team: the hard-mute claim

Status: **DONE** — 2026-08-11. Adversarial review of Objective A.

Reproduce with `PYTHONPATH=. python scripts/redteam_silence.py`. The script also
takes `--open` / `--close` so any candidate threshold pair can be run through the
whole bench, not just the attack it was proposed for.

**The claim under attack**, stated so it can fail:

> The unselected speaker is digitally silent, **and the selected speaker is not
> damaged.**

Both halves. A gate that reaches 100% silence by cutting the target's consonants
has not succeeded — it has moved the failure somewhere less measurable. So every
attack reports both numbers.

**Verdict: the claim survives, with one bounded and now-quantified limit.** Ten
attack groups, 47 assertions. The three failures all belong to one finding
(F-1), which is a property of level-threshold gating rather than a defect, and
§2 shows the two candidate fixes are both worse. No config change is justified.

---

## Summary

| # | attack | result |
|:--|:-------|:-------|
| 1 | quiet fricatives (/s/, /f/) at −20…−35 dB | **survived** — 100% kept at every level |
| 2 | global `ref_percentile` vs a shout | **F-1** — quiet speech below `open_db` is lost |
| 2b | would a *local* reference fix F-1? | **survived** — the fix is strictly worse |
| 2c | does fusion buy headroom to lower `open_db`? | **survived** — the trade is conserved |
| 3 | plosive onsets vs `min_on_ms` + lookahead | **survived** — 10–80 ms bursts, 100% kept |
| 4 | backchannels ("mhm") vs `min_on_ms` | **survived** — 150–400 ms at −12…−20 dB, 100% |
| 5 | speech the mouth does not advertise | **survived** — all 5 cases 100% kept |
| 6 | hostile vision muting a live speaker | **survived** — all 8 inputs 100% kept, none raised |
| 7 | is the silence exact or merely quiet? | **survived** — peak `0.000e+00` |
| 8 | reverb tails vs gate chatter | **survived** — 2 closings for 2 utterances, all RT60 |

---

## F-1 — the only finding: dynamic range is bounded by `open_db`

`ref_percentile = 95` is computed over the **whole clip**. One shout raises the
reference for everything, so a quiet passage elsewhere is measured against the
shout. Sweeping the quiet level finds where that bites:

| quiet level | headroom (`quiet − open`) | alone | after a shout | +vision |
|------------:|--------------------------:|------:|--------------:|--------:|
| −18 dB | +12 | 100.0% | 100.0% | 100.0% |
| −22 dB | +8 | 100.0% | 100.0% | 100.0% |
| −26 dB | +4 | 100.0% | 100.0% | 100.0% |
| −28 dB | +2 | 100.0% | 100.0% | 100.0% |
| −30 dB | **0** | 100.0% | **0.0%** | **0.0%** |
| −34 dB | −4 | 100.0% | **0.0%** | 0.0% |
| −40 dB | −10 | 100.0% | **0.0%** | 0.0% |

**Severity: low, and bounded by arithmetic.** The cliff lands exactly where
`quiet_db = open_db`; there is no soft region and no clip-dependent surprise.
The usable intra-clip dynamic range is 30 dB, which comfortably covers normal
conversational variation — attack [1] confirms fricatives 20–35 dB below vowel
peaks survive intact, because they sit inside a *word* whose surrounding vowels
hold the gate open. F-1 needs a passage that is *both* 30 dB down *and*
surrounded by silence — a genuine whisper after a shout, in one clip.

**The `+vision` column is the important one.** It never rescues anything, and
that is by construction: `visual_veto_db` only shifts thresholds **up**.

```
effective_open = open_db + visual_veto_db * (1 - v)      # shift >= 0, always
```

Vision can add suppression. It can never restore sensitivity. This is a
deliberate asymmetry — a fusion that could *lower* the threshold on visual
evidence would mute nothing when the tracker failed but would open the gate on
residual whenever a face happened to be moving, which is the failure mode that
killed the absolute-lip-position attempt. F-1 is the price of that asymmetry and
it is the right price.

---

## §2 — both obvious fixes are worse than the finding

### 2b. A local (windowed) reference

The textbook answer to a global percentile. Measured on the one span this
project exists to protect — a stretch containing nothing but the interferer's
residual:

| residual level | global ref | local ref |
|---------------:|-----------:|----------:|
| −35 dB | **0.0% leaks** | **100.0% leaks** |
| −45 dB | **0.0% leaks** | **100.0% leaks** |

A window containing only residual makes the residual *its own* 95th percentile,
so `rel ≈ 0 dB` and the gate cannot distinguish it from speech. The fix trades a
quiet-speech limit for the exact ghost whisper the project exists to remove.
**F-1 is not a bug in the design; it is the cost of the design.**

### 2c. Lower `open_db`, now that fusion supplies suppression

The plausible idea: pre-A4b `open_db` was the only thing suppressing residual,
so it could not go low; now the visual veto helps, maybe it can. `AV0`/`Ac0` are
exact-zero percentages on a residual-only span with and without vision; `r-35%`
is the leak from a residual sitting at −35 dB.

| open/close | quiet% | kept% | **r-35%** | AV0@−12 | Ac0@−12 | AV0@−9 | Ac0@−9 | AV0@−6 | Ac0@−6 |
|-----------:|-------:|------:|----------:|--------:|--------:|-------:|-------:|-------:|-------:|
| **−30/−40** | 0.0 | 100.0 | **0.0** | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 0.0 |
| −34/−44 | 0.0 | 100.0 | **0.0** | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 0.0 |
| −36/−46 | 100.0 | 100.0 | **100.0** | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 0.0 |
| −38/−48 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| −40/−50 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |

`quiet%` and `r-35%` are **the same axis read from opposite ends**, and they flip
within one row of each other: −36 buys quiet speech at −34 dB and pays for it by
passing a residual at −35 dB. The trade is conserved to within the 2 dB step of
the sweep.

This is `RESEARCH_GATING_FUSION.md` §2's claim — *"the acoustic axis has no
operating point where both hold"* — measured rather than argued. **`open_db`
stays at −30**, which keeps 6 dB of margin above the −36 flip.

> **A near-miss worth recording.** The first version of this sweep omitted
> `r-35%` and reported −36/−46 as a free win: quiet-speech survival 0 → 100%
> with every silence column unchanged. It looked like a clean config
> improvement. It was an artefact of measuring silence only on residuals that
> sat far below the threshold in the first place. The `Ac0` (no-vision fallback)
> column had been added for the same reason — to stop a vision-only measurement
> justifying a change that regresses every unmatched face. **Do not read the
> `AV0`/`Ac0` columns without `r-35%`.**

---

## Attacks the claim survived

**[1] Quiet fricatives.** /s/ and /f/ sit 20–30 dB below vowel peaks and are the
first thing a level gate destroys. 100% kept at −20, −25, −30 and even −35 dB —
below `open_db` — because `min_off_ms` and the hysteresis band hold the gate open
across the consonant rather than re-deciding per frame.

**[3] Plosive onsets.** Bursts of 10, 20, 40, 80 ms before a vowel: 100% kept at
every duration. `lookahead_ms = 20` opens the gate before the triggering frame,
so the raised-cosine ramp lands in the silence *ahead* of the onset. A 10 ms
burst surviving a 60 ms `min_on_ms` confirms the dwell requirement gates the
*state*, not the audio.

**[4] Backchannels.** "mhm" / "yeah" — short, quiet, and the listener's mouth
barely moves, so they are simultaneously an acoustic and a visual edge case.
150 ms @ −12 dB, 250 ms @ −12, 150 ms @ −20, 400 ms @ −18: **100% kept in all
four.** Notably the 150 ms @ −20 dB case passes with vision saying "still".

**[5] Speech the mouth does not advertise.** Mumbling, a hand over the mouth,
extreme head pose, a beard, and tracker latency. All five — vision agrees,
half-amplitude mouth, mouth barely moves, vision lags 200 ms, vision leads
200 ms — **100% kept**. The ±200 ms tolerance is the 250 ms symmetric dilation
(`visual_hold_ms`) doing its job; it was added for mid-utterance stillness and
buys A/V desync tolerance for free.

**[6] Hostile vision — the worst outcome, a speaker selected and silent.**
Eight adversarial lip signals: constant zero, constant nonzero, all-NaN, a
single spike, negative values, `1e6`, one frame, and empty. **All eight kept
100%, none raised.** Constant-zero and all-NaN are the important pair: both mean
"no usable evidence", and both correctly abstain instead of vetoing. Empty and
one-frame confirm the length-mismatch paths do not throw on a degenerate track.

**[7] Is the silence exact?** A −60 dB floor is still an audible ghost on
headphones, so "quiet" is not the claim.

- ch0 during B's turn: **100.0% of samples exactly `0.0`**
- peak of that span: **`0.000e+00`** — not small, *zero*
- the gate envelope reaches exact `0.0` and exact `1.0`, so the raised-cosine
  ramp terminates rather than asymptotes

**[8] Reverb tails.** whamr16k's input is reverberant by construction, and a
decaying tail crosses `close_db` slowly — the classic chatter trigger. RT60 0.3,
0.6 and 1.0 s each produced **exactly 2 closings for 2 utterances**. Zero
chatter, including at RT60 = 1.0 s where the tail outlasts the utterance.

---

## §3 — the leak on real audio does not arrive through `open_db`

Added 2026-08-11, after a user report on a real clip: *"if the sound of the
selected speaker is not loud but the other non selected person is loud then the
loud person's voice leaks"*, together with *"the selected speaker's sound is a
bit suppressed in the process of reducing leaks"*.

Both symptoms point at a threshold, and F-1 above makes that reading look
obvious — the residual clears `open_db` when the rival is loud. It is wrong, and
the measurement that settles it is an **attribution**, not a sweep. For every
leaking frame on the real clip (rival clearly active, this stem not), record
which mechanism was holding the gate open:

| mechanism | stem 0 | stem 1 |
|:----------|-------:|-------:|
| opened by `open_th` | **0** | **0** |
| held open by hysteresis / `min_off_ms` | 54 | 34 |
| added by `lookahead` dilation | 26 | 18 |

Zero frames. The gate opens legitimately on real target speech, the target
stops, the rival keeps talking, and the residual settles **inside the hysteresis
band** — `rel` p90 = −20.6 dB against `close_db = −23`, `open_db = −20`. Too low
to re-open, too high to ever satisfy `min_off`. So the leak is a **dwell**
defect, and no threshold change can reach it.

### What was tried first, and refuted

A **rival-relative leak floor**: keep the gate shut where this stem sits more
than M dB below the loudest rival. It is a strictly weaker test than the
dominance gating already refuted in `dsp.gate_mask` (dominance asks the target to
*win*, at p10 = −4.5 dB; a floor only asks it not to lose badly), and an offline
sweep endorsed it — at M = 20, 64.2% of leak frames satisfied the condition while
0.37% of genuine target speech did, with loss scored over every target-active
frame including overlap.

End-to-end it changed **nothing**: silence/retain identical to disabled, with and
without the visual veto active. The attribution table says why, and says it for
*every* M: a constraint on the open threshold cannot close a gate that `open_th`
never opened. Applying the same floor to `close_th` did move silence, but was
dominated on both axes by plain `min_off_ms`, which uses no cross-stem
information at all (M = 15 → 80.2%/92.5%; `min_off_ms = 10` → 82.4%/96.4%). The
knob was removed rather than left at 0, so it cannot be "re-enabled" later
without re-reading this section.

### The fix, and why it is not a fit

`lookahead_ms: 20 → 10` is a **correction**. `_run_schmitt` already opens
retroactively across the `min_on` frames that justified the decision, so onsets
do not depend on `_advance` at all; the lookahead only has to cover the ramp, and
`ramp_ms` is 10. The extra 10 ms bought no onset protection and leaked.

`min_off_ms: 30 → 10` is one frame, the minimum. It is safe for a structural
reason, though **not** the tempting one: short `min_off` does not "refill" brief
dropouts — a frame genuinely below `close_db` closes the gate, and should. What
makes it cheap is the same retroactive fill: re-opening costs no `min_on` delay.
Without the fill, a 1-frame `min_off` would lose (`min_on` − 1) = **50 ms of
speech at every re-onset**, and retain would collapse. That claim is now pinned
by `check_fusion.check_gate_dwell`, which failed the first version of this
paragraph and is why it says what it says.

Measured end-to-end through `apply_gate` with the visual veto active, on the real
clip:

| `min_off` / `look` | silence | retain | whisper dB | gate events/s |
|:-------------------|--------:|-------:|-----------:|--------------:|
| 30 / 20 (was) | 66.7% | 98.1% | −23.2 / −25.6 | 22.1 / 7.2 |
| 30 / 10 | 71.2% | 98.1% | | |
| 20 / 10 | 75.1% | 98.1% | −23.8 / −26.9 | 12.3 / 5.7 |
| **10 / 10 (shipped)** | **80.5%** | **97.9%** | **−25.6 / −27.8** | **11.1 / 4.6** |

Better on every axis at once, chatter included. That last column is the
counter-intuitive one and it is the reason this is not a silence/retain trade:
closing promptly produces **fewer** gate transitions than hanging open inside the
band, because a gate held open by `min_off` re-closes and re-opens around every
dip in the residual. Exact-zero fraction rose 28.5%/31.8% → 35.2%/36.7%.

The §2c warning still stands and is untouched: `open_db` and `close_db` were not
changed, and this section is not a licence to revisit them. The F-1 dynamic-range
results are **byte-identical** under the new dwell values, as expected — the
change moves dwell and dilation, not thresholds.

Fitted on ONE clip. Prefer `min_off_ms = 20` if a second clip shows word holes;
it holds retain at the 98.1% baseline and still gains 8.4 points of silence.

---

## What this changes

**Nothing in the config.** `open_db = -30`, `close_db = -40`,
`visual_veto_db = 25`, `visual_hold_ms = 250` all stand, and §2c is now the
recorded evidence for why `open_db` should not be "improved" later.

> **Superseded in part, 2026-08-11.** The threshold conclusion stands exactly as
> written — `open_db`/`close_db` were not touched. But `min_off_ms` and
> `lookahead_ms` *were*, on evidence this bench could not produce, because every
> attack here is threshold-shaped. See §3.

**Two things in the bench**, both of which caught a wrong conclusion:

- `--open` / `--close` overrides, so a candidate threshold runs the *whole*
  bench. A change proposed against one attack has to survive the other nine.
- The `Ac0` (no-vision) and `r-35%` columns in §2c. Without them the sweep
  endorsed a config change that would have regressed the fallback path.

**One correction to an earlier result.** The first run of attack [2] reported
`quiet speech survives a later shout: alone 76.2% -> with shout 76.2%` as a
failure. The two numbers being identical proves the shout had no effect; the
missing 23.8% was a 0.25 s digital-silence gap deliberately placed inside the
measurement span (0.25/1.05 = 23.8%). The gate was correct and the test was
wrong. It now measures speech spans only, and an inline comment records the
mistake so it is not reintroduced.

---

## Residual risk — what this bench still cannot tell you

The signals are synthetic surrogates. The *mechanisms* and the *orderings*
transfer; the exact dB values will shift on real SepFormer output.

1. **Real lip tracking is noisier than `lips_for()`.** Attack [6] covers hostile
   *values*, but not realistic MediaPipe jitter statistics — intermittent
   dropout during head rotation, per-frame landmark noise correlated with pose.
   The 90th-percentile normalisation in `visual_activity` is the untested part.
2. **Multi-speaker overlap.** Every attack here is one target and one
   interferer. Genuine simultaneous speech — where both mouths move and both
   voices are present — is not attacked at all, and it is the case where the
   veto has no discriminating evidence.
3. **Non-speech mouth motion:** chewing, laughing, yawning. `visual_activity`
   reads these as "active", which lifts the veto and returns the gate to its
   pure-acoustic behaviour — the safe direction, but unmeasured.
4. **F-1 on real audio.** A 30 dB intra-clip range is generous for
   conversational speech but the SP Cup evaluation set may contain deliberately
   hard material. If it does, the symptom is a lost whisper, not a ghost — and
   §2c says the only lever trades it straight back for bleed-through.
