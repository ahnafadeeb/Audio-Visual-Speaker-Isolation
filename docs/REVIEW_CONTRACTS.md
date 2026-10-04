# Review: cross-module contracts

Status: **DONE** — 2026-08-11. Static + empirical review of the contracts
between `pipeline`, `dsp`, `channels`, `matching`, `media`, `main`, and the
browser.

Reproduce with `PYTHONPATH=. python scripts/check_fusion.py` (numpy + scipy
only — no torch, no ffmpeg, no GPU, so it runs on any machine).

**Scope.** Not "is the DSP correct" (that is `REDTEAM_SILENCE.md`) but "do two
modules that never call each other still agree about what a value *means*". This
class of defect never raises. It plays the wrong voice, hangs a progress bar, or
500s an endpoint — while every individual function passes its own tests.

**Verdict: six findings, all fixed.** Two are high severity and both were
reachable in the shipped demo path. One of them is triggered *by the project's
headline success condition*, which is why no amount of testing the failure cases
would have found it.

---

## Summary

| # | finding | severity | status |
|:--|:--------|:---------|:-------|
| F-1 | Non-finite floats break all three JSON boundaries; **perfect silence is the trigger** | **high** | fixed |
| F-2 | Duplicate stem claim: `lips_by_stem` keeps the LAST claimant, `plan_channels` the FIRST | **high** | fixed |
| F-3 | `stems_demo` and `stems_raw` normalised independently → 2.9 dB scale mismatch | medium | fixed |
| F-4 | `meta["silence"]` computed before the export gain, so `peak` described an array that was never written | medium | fixed |
| F-5 | UI meters indexed by button position against per-channel levels | medium | fixed |
| F-6 | `make_demo_job.py` wrote a `meta.json` missing three keys `pipeline.py` emits | low | fixed |

---

## F-1 — the success case is the one that cannot be serialised

**High.** Reachable on every clean job. Breaks the deep-link boot path.

Two functions in `dsp.py` legitimately returned `-inf`:

- `silence_stats(x)["nonzero_floor_db"]` — a channel with **no nonzero samples**
  has no residual floor. `20*log10(min(|nonzero|))` over an empty set is `-inf`.
- `measure_leakage_db(target, other)` — a target that **never goes silent** has
  no measurement window, so the old code returned `-inf` as a sentinel.

`whisper_db(out, own, other)` was added later and follows the same `None`
contract, for the same reason: a channel whose rival never holds the floor alone
has no window to measure in. It returns `None` there, and `None` again when the
window exists but `out` is bit-exact zero across it — perfect silence has no
finite dB value. Both read as "not measurable" rather than "measured and
perfect", and the UI labels them that way.

It exists because `measure_leakage_db` needs the **ground-truth interferer
component**, which only the synthetic bench has. Passing it the neighbouring
output channel instead — as `pipeline.py` did — measures how loud the other
speaker is while this one is silent, and that number *rises* as separation
improves. Real jobs reported `+4.4 dB` from that inversion while the gate was
working correctly. `measure_leakage_db` is now bench-only
(`scripts/check_fusion.py`); runtime code calls `whisper_db`.

`-inf` is the correct real-number answer and an invalid JSON one. RFC 8259 has
no infinity literal, and the three consumers disagreed about it in three
different ways. Measured:

```
silence_stats(all-zero)  -> {'exact_zero_fraction': 1.0, 'nonzero_floor_db': -inf, 'peak': 0.0}
measure_leakage_db(...)  -> -inf
json.dumps(...)          -> {"nonzero_floor_db": -Infinity, ..., "leakage_db": -Infinity}
strict reparse           -> REJECTED -- non-standard constant: -Infinity
json.dumps(allow_nan=False) -> RAISES: Out of range float values are not JSON compliant
```

### The three boundaries

**1. `GET /api/jobs/{job_id}` → 500.** The handler returned `job.snapshot()`,
which embeds `meta`. FastAPI wraps a bare dict in Starlette's `JSONResponse`,
whose `render()` is:

```python
json.dumps(content, ensure_ascii=False, allow_nan=False,
           indent=None, separators=(",", ":"))
```

`allow_nan=False` **raises** on non-finite values (verified against the
Starlette source). This is exactly the endpoint that `/?job=demo` boots from —
the way the demo is opened at a presentation.

**2. SSE `done` event → the progress bar hangs on success.** `_sse()` used bare
`json.dumps`, which defaults to `allow_nan=True` and happily emits the
non-standard `-Infinity` token. The browser's `JSON.parse(e.data)` in
`es.onmessage` **throws** on it, so the terminal `done` event is never applied:
the job completed, the artefacts are on disk, and the UI sits at 99% forever.

**3. `meta.json` on disk is not valid JSON.** Python's `json.loads` *accepts*
`-Infinity`, so `adopt_existing()` round-tripped it happily and the corruption
was invisible from inside the server. No other tool would read the file.

### Why this survived testing

The trigger is success. A channel that achieves this project's headline
guarantee — 100% bit-exact zeros — is precisely the channel with no nonzero
sample, and therefore precisely the channel whose statistics cannot be
serialised. A job with audible bleed-through serialises fine. **The better the
gate performs, the more likely the endpoint is to 500.**

### Fix

Three layers, because one is not enough:

1. **Producers report honestly.** `silence_stats` returns `None` for
   `nonzero_floor_db` when there is no floor; `measure_leakage_db` and
   `whisper_db` return `None` when there is no measurement window. `None`
   (JSON `null`) is used
   rather than a sentinel like `-400.0` because a large negative number *can* be
   mistaken for a measurement — and for leakage it would be mistaken for the
   **best possible result** while actually meaning "the gate never closed".

2. **A boundary net.** New `app/serialization.py`: `json_safe()` recursively
   maps non-finite floats to `None`, unwrapping numpy scalars and arrays on the
   way (`np.float32('-inf')` is *not* an instance of `float` and would otherwise
   slip past the check straight into the encoder). `dumps()` wraps it and sets
   `allow_nan=False` deliberately — after scrubbing there is nothing left to
   reject, so if it ever raises, that is a real bug in `json_safe` and we want
   it loud in the server log rather than silently emitting a token the browser
   will choke on.

3. **All three call sites converted.** `main.job_status` returns
   `JSONResponse(json_safe(...))`; `main._sse` and both `meta.json` writers use
   `serialization.dumps`.

The UI renders `null` as `"none (perfect silence)"` / `"n/a (no measurement
window)"` rather than a bare `null` that reads like a bug.

Verified end-to-end on a job with one perfectly muted channel:

```
[1] GET /api/jobs/{id}  (Starlette allow_nan=False)  -> 200 OK
[2] SSE 'done' event -> browser JSON.parse OK (progress bar completes)
[3] meta.json on disk -> valid strict JSON
rendered floor for the muted channel: {"exact_zero_fraction": 1.0, "nonzero_floor_db": null, "peak": 0.0}
```

A genuinely tiny floor is still reported as a number, not nulled — `-300.0 dB`
survives as `-300.0`. Only the absent measurement becomes `null`.

---

## F-2 — two modules, two different answers to "who owns this stem?"

**High.** Silent wrong-voice failure: the user clicks face A and hears B.

`assignment` is indexed by **track** and holds **stem** values. Two consumers
read it, and they disagreed about a duplicate claim:

```python
# channels.lips_by_stem -- assigns into a list in track order: LAST wins
for track_i, stem_j in enumerate(assignment):
    lips[stem_j] = tracks[track_i].lip

# channels.plan_channels -- dedups with a seen-set: FIRST wins
matched = [(i, j) for i, j in matched if not (j in seen or seen.add(j))]
```

Measured with `assignment = [0, 0]`:

```
lips_by_stem  = ['t1', None]              <- LAST claimant wins
plan_channels = order [0, 1] ch {0: 0, 1: None}   <- FIRST claimant wins
  UI labels track 0 as channel 0; that channel was gated with t1's lips
```

So channel 0 is **gated with track 1's lip signal** while being **labelled and
drawn as track 0**. The visual veto then fires against the wrong mouth, which
degrades the gate on top of the mislabelling.

### Why the existing check missed it

`check_fusion.py` already asserted `plan_channels([0, 0], …)` stays a
permutation — and it does. `order` was valid the entire time. The defect was
never in either function alone; it was in the *disagreement between them*, which
no single-function assertion can see.

### Where duplicates come from

The Hungarian solver is one-to-one, so this looked unreachable. But
`matching.match_stems_to_tracks` had a scipy-unavailable fallback:

```python
cols = np.argmax(score, axis=1)[: len(rows)]
```

`argmax` is not an assignment — it picks each row's favourite independently.
Measured over 4000 random square matrices (n ∈ 2..4), **it produced a collision
in 72.2% of them**. `separation._align_permutation` had the identical fallback
with a worse blast radius: its result is used as `cur[order]`, so a duplicate
copies one source onto two channels and **drops the other entirely** — one
speaker heard on both buttons, the other gone.

### Fix

- **One arbiter.** New `channels.claims(assignment, n_tracks, n_stems)` resolves
  `assignment` into `(track, stem)` pairs, first claimant wins, invalid and `-1`
  entries filtered. Both `lips_by_stem` and `plan_channels` now route through
  it, so the lips a stem is gated with always belong to the track the UI labels
  it as. It is structurally impossible for them to diverge again.
- **Do not manufacture the ambiguity.** New `channels.greedy_one_to_one()` —
  pure Python, no scipy — replaces both `argmax` fallbacks. It repeatedly takes
  the highest-scoring free (row, col) pair, so it is *always* a valid one-to-one
  assignment. Measured: **0 collisions in 2000 random shapes**, versus argmax's
  72%.

Greedy is not always Hungarian-optimal (83% of random 2×2 matrices, 54% at 4×4),
and that trade is stated in the docstring rather than hidden. It only runs when
scipy is missing, and scipy is a hard requirement — `dsp.py` imports it at
module scope — so this is a guardrail, not a working code path. It is fixed so
the guardrail cannot itself be the bug.

---

## F-3 — the demo and raw exports were on different scales

**Medium.** Affects objective scoring, which is the whole reason `stems_raw`
exists.

`write_multichannel_wav` computed its own peak normalisation per file, and
`pipeline` called it twice:

```python
media.write_multichannel_wav(stems_demo, job_dir / "stems_demo.wav", sr)
media.write_multichannel_wav(stems_raw,  job_dir / "stems_raw.wav",  sr)
```

The demo stems are *quieter* than the raw ones by construction — the gate
removed the interferer's energy. So the two files got different gains. Measured
on a representative case:

```
OLD independent gains -> demo x1.0000, raw x0.7136  (+2.9 dB mismatch)
NEW shared gain       -> both x0.7136  ( 0.0 dB)
```

SI-SDR is scale-invariant and would never have shown this. **PESQ is not.** A
raw export normalised on its own peak scores differently from the same audio
normalised with the demo — so the number reported to judges depended on an
implementation detail of the file writer.

**Fix.** New `media.peak_gain(*stem_sets)` returns the single gain that keeps
every set under full scale; `write_multichannel_wav` takes an explicit `gain`.
`pipeline` and `make_demo_job` compute it once and pass it to both writes.
Returns exactly `1.0` when nothing clips, so the common case is a bit-exact
no-op. A positive scalar cannot turn an exact zero into a nonzero one and PCM_16
maps `0.0` to sample 0, so **the hard-mute guarantee survives the scaling** —
verified.

---

## F-4 — the reported peak described an array that was never written

**Medium.** Diagnostics that quietly describe the wrong thing.

`meta["silence"]` was computed on the in-memory `stems_demo` *before*
`write_multichannel_wav` applied its normalisation. When the gain was not 1.0,
the reported `peak` was the pre-normalisation peak — a number for an array that
never reached disk. Anyone checking "does the shipped WAV clip?" against
`meta.json` got the wrong answer.

**Fix.** `pipeline` now applies the shared gain in memory *first*, then computes
the stats, then writes with `gain=1.0`. The reported numbers describe the bytes
that shipped. `export_gain` is recorded in `meta["silence"]` so the
transformation is auditable rather than implicit.

---

## F-5 — meters indexed by position, levels indexed by channel

**Medium.** Same track/channel confusion as F-2, on the browser side of the wire.

```js
[...el.speakers.children].forEach((b, i) => {
  bar.style.width = `${Math.min(100, levels[i] * 320)}%`;   // i is a POSITION
});
```

`el.speakers.children` has one button per **track**, including unmatched ones
(`channel === null`, rendered disabled with a "no stem" tag). `engine.levels()`
has one entry per **channel**. The two lists are the same length only when every
detected face got a stem. As soon as one does not — a third person, a poster, a
reflection in a window — every button after it reads its neighbour's meter and
the last one reads `undefined`.

**Fix.** Index by `dataset.channel`, which is the channel index the button was
built with; unmatched buttons read 0.

---

## F-6 — the walking skeleton did not exercise the real meta shape

**Low**, but it undermines the purpose of the script.

`scripts/make_demo_job.py` exists to exercise every contract the browser depends
on without loading torch. Its `meta.json` was missing three keys `pipeline.py`
emits: `visual`, `alignment`, and `config`. `app.js` tolerates the gap via its
`meta && meta.x` guards, so nothing broke — but a consumer that assumes those
keys exist would pass against the demo job and fail on the first real clip,
which is the exact inversion of what the skeleton is for.

**Fix.** The demo job now emits every key the pipeline does, and takes the same
shared-gain export path rather than a shortcut the real exporter does not take.

---

## Not a defect

**Unclaimed-stem backfill leaves `confidence` at 0.0.** `matching` assigns any
stem no track claimed to the highest-scoring free track, without setting a
confidence. This is correct and deliberate: a backfilled assignment has no
margin to report, `0.0 < min_confidence`, so `reliable()` returns False and the
UI marks it "low confidence". The audio is still playable — it is labelled
honestly rather than hidden. Documented here so it is not "fixed" later.

---

## `confidence` is a property of the pairing, not of a track

Two wrong versions of this shipped in sequence, both of which *looked* like
per-track confidence and neither of which measured the decision being made.

1. **Row top-2 gap.** Hungarian optimises globally, so it can place a track on
   its second-choice stem. The top-2 gap then reports a margin belonging to the
   stem the track did **not** get. On the test clip Speaker B was assigned stem
   1 (`0.0463`) while preferring stem 0 (`0.0529`), and this returned `+0.0067`
   — positive, above `min_confidence`, and arguing against its own assignment.
2. **Assigned stem's margin over the best alternative in that row.** Honest
   about direction, still per-track. But with two faces and two stems the
   pairing is not decided per track: Speaker A preferred stem 0 by `0.35`,
   which *forces* B onto stem 1 no matter how flat B's own row is. This
   reported B at `0.0` and flagged the only consistent pairing unreliable.

The shipped measure is the total score of the winning one-to-one assignment
minus that of the best **feasible** alternative — one where the stems are still
distinct. "Feasible" is load-bearing: moving a single track to its preferred
other stem is not an alternative assignment, because that stem is taken, and
the relaxed total can exceed the optimum (collapsing the margin to 0). The true
second best differs from the optimum in at least one pair, so `matching`
forbids each winning pair in turn and re-solves — the first step of Murty's
algorithm, at most `max_faces` solves of a ≤4×4 matrix.

Every track in the pairing therefore shares one confidence value. On the test
clip both faces read `0.2916` and `reliable=True`, where the per-track measures
read `0.2741 / 0.0048` and `0.3536 / 0.0`.

`scripts/check_fusion.py` verifies the implementation against a brute-force
enumeration of all pairings. That enumeration walks row **combinations** against
column **permutations**: permuting both emits the same assignment `k!` times,
making the top two totals identical and every margin `0.0` — which is how the
first version of the check managed to fail a correct implementation.

---

## Regression coverage added

`scripts/check_fusion.py` grew from 31 to 51 assertions (counted as emitted at
run time, not as `check(` call sites — several sit inside loops). The new ones:

**Index arithmetic** — that a duplicate claim keeps `lips_by_stem` and
`plan_channels` agreeing on the owner, across `[0,0]`, `[1,1,0]`, `[0,0,0]`;
that `claims` dedups first-wins and filters `-1`/out-of-range/short input; that
`greedy_one_to_one` is one-to-one and complete over 2000 random shapes, and on
the specific matrix where argmax collides.

**Serialization** — that `silence_stats` on an all-zero channel is `None`, not
`-inf`; that `measure_leakage_db` with no window is `None`; that a realistic
payload survives `json.dumps(..., allow_nan=False)` (the exact call Starlette
makes) **and** reparses under a strict parser with `parse_constant` rejecting
(browser-equivalent); that `json_safe` scrubs every non-finite shape including
numpy scalars, nested containers, tuples, and arrays, while leaving the *string*
`"-inf"` alone; and that a genuinely tiny floor (−300 dB) is preserved rather
than nulled.

```
$ PYTHONPATH=. python scripts/check_fusion.py
...
all checks passed
```

`scripts/redteam_silence.py` re-run after these changes: unchanged. The same
three F-1 findings from `REDTEAM_SILENCE.md` (the bounded intra-clip
dynamic-range limit), no new ones. **No gate config was changed by this review.**

---

## Files touched

| file | change |
|:-----|:-------|
| `app/serialization.py` | **new** — `json_safe`, `dumps` |
| `app/dsp.py` | `silence_stats` / `measure_leakage_db` return `None`, not `-inf` |
| `app/channels.py` | **new** `claims` (single arbiter) and `greedy_one_to_one` |
| `app/matching.py` | argmax fallback → `greedy_one_to_one` |
| `app/separation.py` | argmax fallback → `greedy_one_to_one` |
| `app/media.py` | **new** `peak_gain`; `write_multichannel_wav` takes explicit `gain` |
| `app/main.py` | `job_status` → `JSONResponse(json_safe(...))`; `_sse` → safe dumps |
| `app/pipeline.py` | shared gain applied before stats; safe dumps for both artefacts |
| `app/static/app.js` | meters indexed by channel; `null` rendered readably |
| `scripts/make_demo_job.py` | full meta shape; shared-gain export; safe dumps |
| `scripts/check_fusion.py` | +20 assertions (31 → 51) |

---

## Follow-on

The browser half of these contracts — the transport, the gain automation, and
the A/V clock — was reviewed separately in **`docs/REVIEW_WEBAUDIO.md`**. Two
findings above have a direct sequel there:

- **F-1's symptom class** (an async failure that hangs the progress bar with no
  message) recurs from four *frontend* paths, none of which involve non-finite
  floats. See W-9.
- **F-5's channel-vs-track indexing** is preserved verbatim through a rewrite of
  `ui.meters()` and re-verified. See W-11.
