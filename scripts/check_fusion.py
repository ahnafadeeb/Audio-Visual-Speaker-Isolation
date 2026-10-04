"""Contract checks for the audio-visual gate and its index arithmetic.

Runs on numpy + scipy alone -- no torch, no mediapipe, no ffmpeg, no soundfile,
no GPU -- so it is the check you can run on any machine before a demo.

Two families of failure are covered, because they fail in opposite directions:

  * **Index arithmetic** (``app.channels``).  Three index spaces -- track,
    stem, channel -- and using one where another belongs never raises.  It just
    plays the wrong person's voice.
  * **Abstention** (``resample_hold`` / ``visual_activity``).  The gate must
    treat "no face" as *no evidence*, not as *not speaking*.  Confusing the two
    mutes a real speaker who turned their head.

Run::

    PYTHONPATH=. python scripts/check_fusion.py
"""

from __future__ import annotations

import inspect
import json

import numpy as np

from app import dsp, matching
from app.channels import claims, greedy_one_to_one, lips_by_stem, plan_channels
from app.config import CONFIG
from app.serialization import dumps as json_dumps
from app.serialization import json_safe

SR = CONFIG.audio.sample_rate
FPS = CONFIG.vision.target_fps

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


class T:
    """Minimal stand-in for FaceTrack: the checks only need ``.lip``."""

    def __init__(self, tag):
        self.lip = tag


def make_audio() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, clearly-separated two-stem test audio.

    Alternating bursts of distinct pitches, one stem leading (A) then the other
    (B), so the envelopes correlate strongly with their own track's lip signal
    and weakly with the other's.  Used by the matcher checks only.
    """
    t = np.arange(SR) / SR
    n_on = SR // 4
    stem_a = np.zeros(SR, dtype=np.float64)
    stem_b = np.zeros(SR, dtype=np.float64)
    for i, start in enumerate(range(0, SR - n_on, n_on)):
        seg = 0.3 * np.sin(2 * np.pi * (440.0 if i % 2 == 0 else 660.0) * t[:n_on])
        if i % 2 == 0:
            stem_a[start:start + n_on] += seg
        else:
            stem_b[start:start + n_on] += seg
    return stem_a, stem_b


# --------------------------------------------------------------------------- #

def check_planning() -> None:
    print("\nchannel planning -- track/stem/channel index spaces")

    # The ordinary case.
    order, ch = plan_channels([1, 0], n_tracks=2, n_stems=2)
    check("2 faces, swapped assignment", order == [1, 0] and ch == {0: 0, 1: 1})

    # More faces than stems: the case the old naive filter got wrong.
    order, ch = plan_channels([0, -1, 1], n_tracks=3, n_stems=2)
    check("3 faces / 2 stems: unmatched track gets None",
          order == [0, 1] and ch == {0: 0, 1: None, 2: 1}, f"channel_of={ch}")

    # Every channel index must address a row that exists in the WAV.
    for a, nt, ns in ([0, -1, 1], 3, 2), ([-1, -1], 2, 2), ([1], 1, 3), ([], 0, 2):
        order, ch = plan_channels(a, nt, ns)
        ok = (sorted(order) == list(range(ns))
              and all(v is None or 0 <= v < ns for v in ch.values()))
        check(f"order is a permutation, channels in range  a={a}", ok,
              f"order={order} ch={ch}")

    # Hostile: a duplicate stem claim would otherwise drop a channel entirely.
    order, ch = plan_channels([0, 0], n_tracks=2, n_stems=2)
    check("duplicate stem claim stays a permutation",
          sorted(order) == [0, 1] and ch[1] is None, f"order={order} ch={ch}")

    # ...and the half that a permutation check cannot see.  lips_by_stem used
    # to let the LAST claimant win while plan_channels let the FIRST win, so a
    # channel was gated with one track's lips and labelled as another's.  Both
    # now route through `claims`.  This is a wrong-voice bug that raises
    # nothing, so it needs an explicit assertion.
    for a, nt, ns in (([0, 0], 2, 2), ([1, 1, 0], 3, 2), ([0, 0, 0], 3, 3)):
        tr = [T(f"t{i}") for i in range(nt)]
        lips = lips_by_stem(a, tr, ns)
        order, ch = plan_channels(a, nt, ns)
        reordered = [lips[j] for j in order]
        ok = all(reordered[c] == f"t{i}" for i, c in ch.items() if c is not None)
        check(f"duplicate claim: lips and channel agree on the owner  a={a}", ok,
              f"lips={lips} order={order} ch={ch}")

    # The arbiter itself: one stem at most once, first claimant wins, invalid
    # entries filtered.
    check("claims dedups, first wins", claims([0, 0], 2, 2) == [(0, 0)],
          str(claims([0, 0], 2, 2)))
    check("claims filters -1 and out-of-range",
          claims([-1, 5, 1], 3, 2) == [(2, 1)], str(claims([-1, 5, 1], 3, 2)))
    check("claims tolerates a short assignment", claims([0], 3, 2) == [(0, 0)])

    # The scipy-missing fallback must still be a one-to-one assignment.  The
    # old np.argmax(axis=1) was not: it collided on 72% of random square
    # matrices, and a collision here is what produced the divergence above.
    s = [[0.9, 0.8], [0.95, 0.1]]
    check("greedy fallback is one-to-one where argmax collides",
          greedy_one_to_one(s) == [(0, 1), (1, 0)], str(greedy_one_to_one(s)))

    rng = np.random.default_rng(7)
    bad = 0
    for _ in range(2000):
        n = int(rng.integers(1, 5))
        m = int(rng.integers(1, 5))
        pr = greedy_one_to_one(rng.normal(size=(n, m)).tolist())
        rows = [r for r, _ in pr]
        cols = [c for _, c in pr]
        if (len(set(rows)) != len(rows) or len(set(cols)) != len(cols)
                or len(pr) != min(n, m)
                or not all(0 <= r < n and 0 <= c < m for r, c in pr)):
            bad += 1
    check("greedy fallback is one-to-one and complete over 2000 random shapes",
          bad == 0, f"{bad} violations")

    check("greedy fallback handles a degenerate shape",
          greedy_one_to_one([]) == [] and greedy_one_to_one([[]]) == [])

    # Hostile: out-of-range assignment must not index past the WAV.
    order, ch = plan_channels([5, 0], n_tracks=2, n_stems=2)
    check("out-of-range assignment ignored",
          sorted(order) == [0, 1] and ch[0] is None and ch[1] == 0, f"ch={ch}")

    # lips_by_stem inverts the mapping -- entry j must be the track holding j.
    tracks = [T("a"), T("b"), T("c")]
    check("lips_by_stem inverts track->stem",
          lips_by_stem([1, 0], tracks, 2) == ["b", "a"])
    check("lips_by_stem: unclaimed stem is None (acoustic fallback)",
          lips_by_stem([0, -1], tracks, 2) == ["a", None])
    check("lips_by_stem: out-of-range stem ignored",
          lips_by_stem([7, 0], tracks, 2) == ["b", None])

    # The composition is what actually ships: after reordering by `order`,
    # track i's audio must be channel i's audio.
    for assignment, nt, ns in (([1, 0], 2, 2), ([0, -1, 1], 3, 2), ([2, 0], 2, 3)):
        tr = [T(f"t{i}") for i in range(nt)]
        lips = lips_by_stem(assignment, tr, ns)
        order, ch = plan_channels(assignment, nt, ns)
        reordered = [lips[j] for j in order]
        ok = all(reordered[c] == f"t{i}" for i, c in ch.items() if c is not None)
        check(f"track i's lips land on channel i  a={assignment}", ok,
              f"reordered={reordered} ch={ch}")


def check_abstention() -> None:
    print("\nabstention -- 'no face' must never mean 'not speaking'")

    v, ok = dsp.resample_hold([np.nan] * 25, FPS, 100, 100.0)
    check("all-NaN track is fully invalid", (not ok.any()) and not v.any())

    # 2 frames at 25 fps = 80 ms, inside the 200 ms hold.
    v, ok = dsp.resample_hold([1.0, np.nan, np.nan, 2.0], FPS, 8, 50.0)
    check("short dropout is held and stays valid", ok[:7].all(), f"valid={ok}")

    # 10 frames = 400 ms, past the hold.
    lip = [1.0] + [np.nan] * 10 + [2.0]
    v, ok = dsp.resample_hold(lip, FPS, 24, 50.0)
    check("long dropout is marked invalid", not ok.all(), f"valid.sum()={ok.sum()}")

    # Extrapolation past the last video frame is not observation.
    v, ok = dsp.resample_hold([0.5, 0.6, 0.7], FPS, 12, 50.0)
    check("tail past last video frame is invalid", not ok[-1],
          "a video 1 frame short of its audio must not veto the last word")

    a, ok = dsp.visual_activity([0.02] * 50, src_fps=FPS, n_frames=200, frame_ms=10.0)
    check("frozen track abstains (invalid, not silent)", not ok.any())

    a, ok = dsp.visual_activity([np.nan] * 50, src_fps=FPS, n_frames=200, frame_ms=10.0)
    check("absent face abstains", not ok.any())

    check("empty lip signal does not raise",
          dsp.visual_activity([], src_fps=FPS, n_frames=50, frame_ms=10.0)[0].size == 50)


def check_gate_dwell() -> None:
    print("\nthe leak path: dwell and dilation, not the open threshold")

    # These two defaults were changed after measuring WHICH mechanism holds the
    # gate open on leaking frames.  The answer was: never open_th.  Pin the
    # structural facts that make the new values correct, so a future retune has
    # to argue with the mechanism rather than just move the numbers.
    g = CONFIG.gate

    # 1. lookahead only has to cover the ramp, because _run_schmitt fills the
    #    onset retroactively.  If lookahead < ramp the onset gets clipped; if it
    #    is larger, the excess is pure leak with no onset benefit.
    check("lookahead_ms == ramp_ms", g.lookahead_ms == g.ramp_ms,
          f"look={g.lookahead_ms} ramp={g.ramp_ms}")

    # 2. The retroactive fill is what makes a 1-frame min_off safe -- but NOT by
    #    "refilling holes".  A frame genuinely below close_db stays closed, and
    #    should.  What the fill removes is the min_on DELAY on re-opening: the
    #    gate opens from the first frame that justified the decision, not
    #    min_on frames later.  Without it, a short min_off would cost
    #    (min_on - 1) frames of speech at EVERY re-onset -- measured at 50 ms
    #    here -- and that, not the dropout itself, is what would destroy retain.
    n = 40
    rel = np.full(n, -40.0)
    rel[20:] = 0.0                                   # onset at frame 20
    st = dsp._run_schmitt(rel, np.full(n, -20.0), np.full(n, -23.0), 6, 1)
    check("onset opens retroactively across min_on frames",
          bool(st[20]) and not bool(st[19]),
          f"first open frame = {int(np.argmax(st))}, onset at 20")

    # A brief dropout mid-speech: the gap itself closes (correct -- it really is
    # below close_db), and speech resumes on its FIRST frame, with no min_on lag.
    rel2 = np.zeros(n)
    rel2[:5] = -40.0
    rel2[22:24] = -40.0                              # 2-frame dropout mid-speech
    st2 = dsp._run_schmitt(rel2, np.full(n, -20.0), np.full(n, -23.0), 6, 1)
    check("re-onset after a dropout costs no min_on delay",
          bool(st2[24]) and not bool(st2[23]),
          "speech resumes at 24; without the fill it would resume at 29")
    # 3. The refuted floor must stay gone: a rival-relative constraint on
    #    open_th cannot close a gate that open_th never opened.
    check("no rival_floor_db knob", not hasattr(g, "rival_floor_db"))
    check("gate_mask takes no rival_db", "rival_db" not in
          inspect.signature(dsp.gate_mask).parameters)


def check_gate_equivalence() -> None:
    print("\nregression -- the gate without vision must be bit-identical to A4")

    rng = np.random.default_rng(0)
    x = (rng.normal(size=SR * 3) * 0.1).astype(np.float32)
    x[SR:2 * SR] = 0.0

    e_none = dsp.gate_mask(x, sample_rate=SR)
    e_expl = dsp.gate_mask(x, sample_rate=SR, visual=None, visual_veto_db=25.0)
    check("visual=None is bit-identical", np.array_equal(e_none, e_expl))

    e_zero = dsp.gate_mask(x, sample_rate=SR, visual=np.zeros(300),
                           visual_valid=np.zeros(300, dtype=bool),
                           visual_veto_db=25.0)
    check("all-invalid visual is bit-identical (abstains)",
          np.array_equal(e_none, e_zero))

    e_full = dsp.gate_mask(x, sample_rate=SR, visual=np.ones(300),
                           visual_valid=np.ones(300, dtype=bool),
                           visual_veto_db=25.0)
    check("fully-active visual is bit-identical (no veto)",
          np.array_equal(e_none, e_full))

    # And the guarantee the whole chain exists for.
    env = dsp.gate_mask(x, sample_rate=SR)
    check("gate still produces exact zeros", bool((env == 0.0).any()),
          f"{100 * (env == 0.0).mean():.1f}% of samples")

    # A stem shorter than its lip signal, and vice versa -- neither may raise.
    for n in (SR // 3, SR * 5):
        short = (rng.normal(size=n) * 0.1).astype(np.float32)
        out = dsp.apply_gate(np.stack([short, short]), sample_rate=SR,
                             cfg=CONFIG.gate,
                             lips=[[0.02 + 0.05 * (i % 7 == 0) for i in range(75)], None],
                             video_fps=FPS)
        check(f"length mismatch tolerated  n={n}", out.shape == (2, n))


def check_mono_and_empty() -> None:
    print("\ndegenerate inputs")

    check("empty stem set", dsp.apply_gate(np.zeros((0, 0), dtype=np.float32),
                                           sample_rate=SR, cfg=CONFIG.gate).size == 0)
    one = np.zeros((1, SR), dtype=np.float32)
    check("single stem, no lips", dsp.apply_gate(one, sample_rate=SR,
                                                 cfg=CONFIG.gate).shape == (1, SR))
    check("lips list shorter than stems",
          dsp.apply_gate(np.zeros((2, SR), dtype=np.float32), sample_rate=SR,
                         cfg=CONFIG.gate, lips=[None], video_fps=FPS).shape == (2, SR))
    check("digital silence in, digital silence out",
          not dsp.apply_gate(np.zeros((2, SR), dtype=np.float32), sample_rate=SR,
                             cfg=CONFIG.gate).any())


def check_serialization() -> None:
    """The stats of a *perfectly* muted channel must survive the wire.

    Starlette's JSONResponse renders with ``allow_nan=False``, so a single
    non-finite float 500s ``GET /api/jobs/{id}``.  Bare ``json.dumps`` instead
    emits ``-Infinity``, which the browser's ``JSON.parse`` rejects -- losing
    the terminal SSE event and hanging the progress bar on a *successful* job.
    Both paths are triggered by the project's own success condition.
    """
    print("\nserialization -- perfect silence must survive the wire")

    z = np.zeros(SR, dtype=np.float32)
    st = dsp.silence_stats(z)
    check("silence_stats(all-zero) is finite/None",
          st["nonzero_floor_db"] is None and all(
              v is None or np.isfinite(v) for v in st.values()),
          str(st))

    rng = np.random.default_rng(0)
    cont = (rng.normal(size=SR) * 0.1).astype(np.float32)     # never gates shut
    lk = dsp.measure_leakage_db(cont, z, sample_rate=SR)
    check("measure_leakage_db with no silent window is None", lk is None, repr(lk))

    lk2 = dsp.measure_leakage_db(z, z, sample_rate=SR)
    check("measure_leakage_db on silence-vs-silence is finite or None",
          lk2 is None or np.isfinite(lk2), repr(lk2))

    payload = {"silence": {"channel_0": st, "channel_1": dsp.silence_stats(cont),
                           "leakage_db": lk}}

    # The exact call Starlette makes.
    try:
        json.dumps(payload, allow_nan=False)
        check("payload survives allow_nan=False (Starlette JSONResponse)", True)
    except ValueError as exc:
        check("payload survives allow_nan=False (Starlette JSONResponse)", False,
              str(exc))

    # And what a strict parser -- including the browser's JSON.parse -- accepts.
    def _reject(c):
        raise ValueError(f"non-standard constant {c}")

    try:
        json.loads(json_dumps(payload), parse_constant=_reject)
        check("payload reparses under a strict (browser-equivalent) parser", True)
    except ValueError as exc:
        check("payload reparses under a strict (browser-equivalent) parser",
              False, str(exc))

    # json_safe must be total: anything the pipeline can put in meta.
    hostile = {"a": float("inf"), "b": float("-inf"), "c": float("nan"),
               "d": np.float32("nan"), "e": np.float64("-inf"),
               "f": [float("nan"), {"g": float("inf")}], "h": (1.0, float("nan")),
               "i": np.array([1.0, np.nan]), "j": np.float32(0.5),
               "k": np.int64(7), "l": np.True_, "m": None, "n": "-inf"}
    safe = json_safe(hostile)
    try:
        json.dumps(safe, allow_nan=False)
        ok = (safe["a"] is None and safe["b"] is None and safe["c"] is None
              and safe["n"] == "-inf" and safe["f"][0] is None
              and safe["i"] == [1.0, None])
        check("json_safe scrubs every non-finite shape", ok, str(safe))
    except (ValueError, TypeError) as exc:
        check("json_safe scrubs every non-finite shape", False, str(exc))

    # A -inf that means "no measurement" must not be confused with a real
    # -300 dB reading, which is finite and must pass through untouched.
    quiet = np.full(SR, 1e-15, dtype=np.float32)
    stq = dsp.silence_stats(quiet)
    check("a genuinely tiny floor is preserved, not nulled",
          stq["nonzero_floor_db"] is not None and stq["nonzero_floor_db"] < -200.0,
          str(stq))


def check_confidence_honesty() -> None:
    """Confidence must measure the margin of the WHOLE pairing, not one row.

    Two wrong answers were shipped here in sequence, and this pins both:

      1. The row's top-2 gap.  Hungarian optimises globally, so it can place a
         track on its second-choice stem; the top-2 gap then reports a margin
         belonging to the stem the track did NOT get.  On the real clip
         Speaker B was assigned stem 1 (0.0463) while preferring stem 0
         (0.0529) and this returned +0.0067 -- positive, above
         match.min_confidence, and arguing against its own assignment.
      2. The assigned stem's margin over the best alternative IN THAT ROW.
         Honest about direction but still a per-track measure, and the pairing
         is not decided per track: on the same clip Speaker A prefers stem 0 by
         0.35, which forces B onto stem 1 no matter how flat B's own row is.
         This reported B at 0.0 and flagged the only consistent pairing
         unreliable.

    The right measure is the total score of the winning assignment minus that
    of the best FEASIBLE alternative -- an alternative in which the stems are
    still distinct.  Verified here against a brute-force enumeration of all
    permutations, which is tractable at max_faces=4 and is an independent
    implementation of the same definition.
    """
    print("\nmatching -- confidence is the margin of the whole pairing")

    def brute(s: np.ndarray) -> float:
        """Best-minus-second-best over all one-to-one pairings, by enumeration.

        An assignment is a SET of (row, col) pairs, so rows are enumerated as
        combinations and only the columns permute.  Permuting both would emit
        the same assignment k! times, making the top two totals identical and
        every margin 0 -- which is exactly how the first version of this check
        managed to fail against a correct implementation.
        """
        import itertools
        n_r, n_c = s.shape
        k = min(n_r, n_c)
        totals = sorted(
            (sum(s[a, b] for a, b in zip(rows_, cols_))
             for rows_ in itertools.combinations(range(n_r), k)
             for cols_ in itertools.permutations(range(n_c), k)),
            reverse=True)
        return max(0.0, totals[0] - totals[1]) if len(totals) > 1 else max(0.0, totals[0])

    stem_a, stem_b = make_audio()
    stems = np.asarray([stem_a, stem_b], dtype=np.float32)
    n_lip = FPS                                   # one second of video

    # Lip signals that track each stem's own bursts, so the pairing is knowable.
    env_a = np.abs(stem_a).reshape(int(n_lip), -1).mean(1)
    env_b = np.abs(stem_b).reshape(int(n_lip), -1).mean(1)
    tracks = [T(np.cumsum(env_a)), T(np.cumsum(env_b))]

    m = matching.match_stems_to_tracks(
        stems, tracks, sample_rate=SR, video_fps=FPS, cfg=CONFIG.match)
    a, conf, score = m.assignment, m.confidence, m.score
    check("2 tracks / 2 stems: matches brute-force best-minus-second",
          abs(conf[0] - brute(score)) < 1e-9 and abs(conf[1] - brute(score)) < 1e-9,
          f"a={a} conf={[round(c, 5) for c in conf]} brute={brute(score):.5f}")
    check("a knowable pairing clears min_confidence",
          min(conf) > CONFIG.match.min_confidence,
          f"conf={[round(c, 5) for c in conf]} min={CONFIG.match.min_confidence}")

    # The extracted helper is the SAME statistic the null re-scores.  If these
    # ever diverge, the p-value calibrates something other than the number the
    # pipeline thresholds, and it would look fine while meaning nothing.
    _r, _c, margin = matching.pairing_margin(score)
    check("pairing_margin agrees with brute force", abs(margin - brute(score)) < 1e-9,
          f"margin={margin:.6f} brute={brute(score):.6f}")

    # The real clip's matrix, which is the case that motivated the change: one
    # decisive row and one nearly tied row.  The pairing is forced by the
    # decisive row, so confidence must be high for BOTH tracks.
    clip = np.array([[0.0469, -0.3051], [0.0720, 0.0099]])
    check("forced pairing: flat row still reports the pairing's margin",
          abs(brute(clip) - 0.2899) < 1e-4, f"brute={brute(clip):.4f}")
    check("pairing_margin agrees on the real clip's matrix",
          abs(matching.pairing_margin(clip)[2] - 0.2899) < 1e-4)

    # A genuine coin flip must read 0.0 -- the one case where "unreliable" is
    # the correct answer.  If this ever reports a margin, the metric has gone
    # back to measuring something other than the decision.
    check("degenerate all-equal matrix reads 0.0",
          brute(np.full((2, 2), 0.5)) == 0.0)
    check("pairing_margin on an all-equal matrix reads 0.0",
          matching.pairing_margin(np.full((2, 2), 0.5))[2] == 0.0)


def check_null_calibration() -> None:
    """The margin is a scale, not a probability -- and it inflates.

    This is the defect docs/DIAG_MATCHER.md found, and it is the reason a
    threshold on ``confidence`` cannot protect the user.  The statistic divides
    by ``n_frames`` as though smoothed envelope frames were independent samples,
    so on the real clip it read

        40 ms 0.0679 | 200 ms 0.1870 | 600 ms 0.4125 | 1600 ms 0.6654

    while the pairing was WRONG at every window.  A wider ``smooth_kernel`` would
    have shipped the same inverted answer at 13x the confidence.

    So the checks below are built on **pure noise** -- no audio-visual
    relationship of any kind.  The margin must still inflate as the smoothing
    widens (the defect, reproduced), and the p-value must not (the fix, verified
    against the same signals).

    The p-value is checked by its FALSE-POSITIVE RATE over independent noise
    draws, not on one draw.  A valid p-value is uniform under the null, so a
    single draw is a 1-in-20 coin flip and an assertion on it would be a flaky
    test that a bad seed could make pass.  What must hold is the rate.
    """
    print("\nmatching -- the margin inflates with smoothing; the p-value must not")

    def smooth(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return x
        ker = np.ones(k) / k
        return np.stack([np.convolve(r, ker, mode="same") for r in x])

    def z(x: np.ndarray) -> np.ndarray:
        return np.stack([matching._zscore(r) for r in x])

    N_DRAWS, N_SHIFTS = 12, 200
    n = 800                                  # 32 s at 40 ms, the real clip's length
    alpha = CONFIG.match.max_p_value
    stats: dict[int, list[tuple[float, float, float]]] = {}
    for k in (1, 41):
        stats[k] = []
        for seed in range(N_DRAWS):
            rng = np.random.default_rng(1000 + seed)
            L = z(smooth(rng.normal(size=(2, n)), k))
            E = z(smooth(rng.normal(size=(2, n)), k))
            margin = matching.pairing_margin((L @ E.T) / n)[2]
            p, agree = matching.pairing_significance(L, E, n_shifts=N_SHIFTS)
            stats[k].append((margin, p, agree))

    print(f"    {N_DRAWS} independent pure-noise draws, {N_SHIFTS} shifts each")
    print(f"    {'smoothing':>9} {'median margin':>14} {'median p':>9} "
          f"{'p<={:.2f}':>9} {'median null agr':>16}".format(alpha))
    med = {}
    for k, rec in stats.items():
        m = float(np.median([r[0] for r in rec]))
        p = float(np.median([r[1] for r in rec]))
        fp = sum(r[1] <= alpha for r in rec)
        ag = float(np.median([r[2] for r in rec]))
        med[k] = (m, p, fp, ag)
        print(f"    {k:>9d} {m:>14.4f} {p:>9.3f} {fp:>6d}/{N_DRAWS} {ag:>16.0%}")

    m1, _, fp1, _ = med[1]
    m2, _, fp2, _ = med[41]
    check("pure noise: the margin inflates with the smoothing window", m2 > m1,
          f"{m1:.4f} -> {m2:.4f} (this is the defect, reproduced)")
    check("pure noise: the inflated margin clears min_confidence",
          m2 > CONFIG.match.min_confidence,
          f"median margin {m2:.4f} > {CONFIG.match.min_confidence} -- a threshold "
          f"on confidence alone calls pure noise reliable")
    # A valid p-value rejects a true null at its nominal rate.  At alpha=0.05 and
    # 12 draws the expectation is 0.6; 3+ would mean the null is too easy to beat
    # and the p-value is only pretending to be calibrated.
    check("the p-value's false-positive rate stays near nominal at BOTH windows",
          fp1 <= 2 and fp2 <= 2,
          f"{fp1}/{N_DRAWS} at k=1, {fp2}/{N_DRAWS} at k=41, "
          f"expected ~{alpha * N_DRAWS:.1f}")

    # Positive control.  A null that never fires is not a test: if this fails,
    # the p-value rejects everything and every check above is vacuous.
    #
    # Irregular turn-taking, NOT a periodic alternation -- a periodic signal is
    # genuinely re-alignable by a shift of one period, so the null would be right
    # to refuse it and the control would fail for an honest reason.
    rng = np.random.default_rng(7)
    turns = np.zeros((2, n))
    who, i = 0, 0
    while i < n:
        run = int(rng.integers(4, 30))
        turns[who, i:i + run] = 1.0
        i += run + int(rng.integers(0, 8))       # a gap, sometimes zero
        who ^= 1
    L = z(smooth(turns + 0.3 * rng.normal(size=(2, n)), 5))
    E = z(smooth(turns + 0.3 * rng.normal(size=(2, n)), 5))
    p, agree = matching.pairing_significance(L, E, n_shifts=N_SHIFTS)
    rows, cols, margin = matching.pairing_margin((L @ E.T) / n)
    print(f"    positive control: margin {margin:.4f}  p {p:.4f}  "
          f"null agreement {agree:.0%}  pairing {sorted(zip(rows, cols))}")
    check("a real AV correlation beats the null", p <= alpha, f"p={p:.4f}")
    check("...and it recovers the TRUE pairing",
          sorted(zip(rows, cols)) == [(0, 0), (1, 1)],
          f"pairing {sorted(zip(rows, cols))}")

    # null_agreement is a BASELINE, not a verdict.  For a 2x2 problem a null that
    # has destroyed the alignment picks either pairing about half the time -- so
    # ~50% is what BOTH the noise draws and this correct pairing must show.  It is
    # informative only when high (a shift-invariant asymmetry decides the
    # pairing).  Pinned here so nobody later reads ~0.5 as a failure signal.
    check("null agreement sits near the coin-flip baseline for noise AND signal",
          0.2 <= med[1][3] <= 0.8 and 0.2 <= agree <= 0.8,
          f"noise {med[1][3]:.0%}, true pairing {agree:.0%} -- 0.5 is the null's "
          f"own value either way, so only a HIGH value is diagnostic")

    # Determinism: the same signals must give the same p, or two runs of one job
    # disagree about whether to trust the channel labels.
    p_again, agree_again = matching.pairing_significance(L, E, n_shifts=N_SHIFTS)
    check("significance is deterministic",
          p == p_again and agree == agree_again, f"{p} vs {p_again}")

    # Lag 0 must be excluded, or the observed arrangement is counted as evidence
    # for the null.  With a clean 1:1 signal the margin is maximal at lag 0, so
    # including it would floor p at 2/(N+1) instead of 1/(N+1).
    check("the shift null can reach its floor of 1/(n+1)",
          abs(p - 1.0 / (N_SHIFTS + 1)) < 1e-12,
          f"p={p} floor={1.0 / (N_SHIFTS + 1)}")

    # Degenerate shapes must abstain (p=1), not raise and not claim evidence.
    for L_, E_ in ((np.zeros((0, 10)), np.zeros((0, 10))),
                   (np.zeros((2, 2)), np.zeros((2, 2))),
                   (np.zeros((1, 50)), np.zeros((2, 50)))):
        try:
            pv, ag = matching.pairing_significance(L_, E_, n_shifts=8)
            ok = pv == 1.0 and 0.0 <= ag <= 1.0
        except Exception as exc:                        # noqa: BLE001
            ok, pv = False, exc
        check(f"degenerate significance input {L_.shape}/{E_.shape} abstains",
              bool(ok), str(pv))

    # And the whole matcher must survive a track with no usable lip signal at
    # all -- the p-value comes from the same block, so an all-NaN row must not
    # poison it into a NaN that compares False against every threshold and
    # silently reads as "trustworthy".
    stem_a, stem_b = make_audio()
    res = matching.match_stems_to_tracks(
        np.asarray([stem_a, stem_b], dtype=np.float32),
        [T([np.nan] * int(FPS)),
         T(np.cumsum(np.abs(stem_b).reshape(int(FPS), -1).mean(1)))],
        sample_rate=SR, video_fps=FPS, cfg=CONFIG.match)
    check("an all-NaN lip track does not produce a NaN p-value",
          np.isfinite(res.significance) and 0.0 <= res.significance <= 1.0,
          f"p={res.significance:.4f} assignment={res.assignment}")

    # A dead row does NOT drag the pairing down, and that is deliberate: with two
    # tracks and two stems, a confident row for track 1 forces track 0 by
    # elimination.  That is the whole reason confidence is a property of the
    # PAIRING rather than of a track (see pairing_margin), so the dead track
    # inherits its partner's evidence and both read the same margin.
    #
    # The residual risk this does NOT cover: if the dead track is a spurious face
    # detection rather than a silent speaker, elimination is reasoning from a
    # false premise, and neither the margin nor the p-value can see that -- it is
    # the vision stage's min-box-area and persistence guards that must.  Do not
    # add a lip-health condition here expecting it to catch that; it belongs
    # upstream, where the track is created.
    check("a dead lip row is still forced by its confident partner",
          res.assignment == [0, 1] and min(res.confidence) > CONFIG.match.min_confidence,
          f"assignment={res.assignment} conf={[round(c, 4) for c in res.confidence]}")


def check_separation_alignment() -> None:
    """Chunk-boundary permutation stitching in ``app.separation``.

    This group exists because the module had **zero** coverage while owning the
    one failure that is audibly indistinguishable from the bug everyone reaches
    for instead: a mid-clip speaker swap sounds like bleed-through, so it sends
    you tuning the gate while the stitcher is what broke.

    Three properties, in the order they matter:

      1. ``order`` is always a *permutation*.  It is applied as ``cur[order]``,
         so a repeated column duplicates one source onto both channels and drops
         the other -- one speaker on both buttons, the other gone.  This cannot
         be caught by listening to one channel.
      2. The margin reads ~0 when the two orderings are interchangeable.  The old
         ``best - trace`` was 0.0 *by construction* whenever identity won, so it
         reported 0.0 at decisive boundaries too and ``confident`` was true at
         margin 0.0 -- the warning could never fire.
      3. A boundary with no waveform evidence falls through to the accumulated
         spectral identity, and if that is also mute, HOLDS and says so.
    """
    print("\n[separation: chunk-boundary permutation]")
    from app import separation as sep

    rng = np.random.default_rng(7)
    ov = SR                                          # 1 s overlap

    def two_voices(n: int, spec=((110.0, 700.0), (220.0, 1900.0))):
        """Synthetic voices as (f0, formant) pairs.

        The FORMANT is what varies the cepstral profile, so a fixture that
        varied only f0 would test nothing -- low-order cepstral coefficients are
        deliberately blind to excitation.  That is the whole point of using them
        as an identity proxy, and it is also why the "similar voices" fixture
        below has to move the formant, not just the pitch.
        """
        t = np.arange(n) / SR
        out = []
        for k, (f, fmt) in enumerate(spec):
            x = np.zeros(n)
            for h in range(1, 25):
                if f * h >= SR / 2:
                    break
                amp = np.exp(-((f * h - fmt) ** 2) / (2 * 550.0 ** 2)) + 0.02
                x += amp * np.sin(2 * np.pi * f * h * t + k + h)
            out.append(0.2 * x / (np.abs(x).max() + 1e-12))
        return np.asarray(out)

    # -- 1. decisive boundary: the same audio on both sides of the overlap -- #
    v = two_voices(4 * SR)
    prev, cur = v[:, :2 * SR], v[:, SR:3 * SR]       # cur[:, :ov] == prev[:, -ov:]
    out, d = sep._align_permutation(cur, prev, ov, sample_rate=SR)
    check("a decisive boundary reports a LARGE margin, not 0.0",
          d["margin"] > sep.PERM_MARGIN and d["basis"] == "waveform",
          f"margin={d['margin']} basis={d['basis']}")
    check("a decisive boundary that needs no reorder is still called confident",
          d["confident"] and not d["flipped"], f"{d['confident']=} {d['flipped']=}")
    check("a decisive hold leaves the samples untouched",
          np.array_equal(out, cur))

    # -- 2. a genuine flip is detected AND applied -------------------------- #
    out, d = sep._align_permutation(cur[::-1].copy(), prev, ov, sample_rate=SR)
    check("a genuine flip is detected and applied",
          d["flipped"] and d["order"] == [1, 0] and d["confident"]
          and np.allclose(out, cur, atol=1e-12),
          f"order={d['order']} margin={d['margin']}")
    check("flip_gain is positive exactly when reordering helps",
          d["flip_gain"] > 0.0, f"flip_gain={d['flip_gain']}")

    # -- 3. the coin flip: overlap is noise on both sides ------------------- #
    # Not silence -- silence takes the RMS early-out.  This is the case the old
    # code got wrong: a real correlation matrix whose two orderings tie.
    noise = 0.02 * rng.normal(size=(2, 2 * SR))
    out, d = sep._align_permutation(noise.copy(), 0.02 * rng.normal(size=(2, 2 * SR)),
                                   ov, sample_rate=SR)
    check("an uninformative boundary reports margin ~0 and is NOT confident",
          d["margin"] < sep.PERM_MARGIN and not d["confident"]
          and d["basis"] == "hold",
          f"margin={d['margin']} confident={d['confident']} basis={d['basis']}")
    check("an uninformative boundary holds the existing order",
          d["order"] == [0, 1] and not d["flipped"] and np.array_equal(out, noise))

    # -- 4. order is always a permutation, over many random matrices -------- #
    ok = True
    for seed in range(200):
        r = np.random.default_rng(seed)
        n_src = int(r.integers(2, 5))
        _, dd = sep._align_permutation(
            0.1 * r.normal(size=(n_src, 2 * SR)),
            0.1 * r.normal(size=(n_src, 2 * SR)), ov, sample_rate=SR)
        if sorted(dd["order"]) != list(range(n_src)):
            ok = False
            break
    check("order is a permutation for every random matrix (2-4 sources)", ok)

    # -- 5. the identity anchor: what it can and cannot do ------------------ #
    a = sep._IdentityAnchor(2)
    check("an empty anchor abstains rather than returning a tie matrix",
          a.score(v[:, :SR], SR) is None)
    a.observe(v, SR)
    check("an anchor that has heard both voices is ready", a.ready())
    s = a.score(v[:, :SR], SR)
    check("the anchor scores the same voices highest on the diagonal",
          s is not None and s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0],
          None if s is None else f"diag={np.round(np.diag(s), 3)} "
                                 f"off={round(float(s[0, 1]), 3)},{round(float(s[1, 0]), 3)}")
    s_sw = a.score(v[::-1].copy(), SR)
    check("the anchor scores a swapped chunk highest OFF the diagonal",
          s_sw is not None and s_sw[0, 1] > s_sw[0, 0] and s_sw[1, 0] > s_sw[1, 1])
    check("a silent channel yields no profile (None, not a zero vector)",
          sep._cepstral_profile(np.zeros(2 * SR), SR) is None)
    # Level-invariance holds only WITHIN the measurable range: below the absolute
    # floor the profile abstains instead, and that ordering matters -- an
    # invariant-but-meaningless profile computed on the noise floor is exactly
    # what would pull the two speakers' profiles together.
    p_loud, p_quiet = (sep._cepstral_profile(v[0], SR),
                       sep._cepstral_profile(v[0] * 0.25, SR))
    check("the cepstral profile is level-invariant (c0 dropped)",
          p_loud is not None and p_quiet is not None
          and np.allclose(p_loud, p_quiet, atol=1e-6))
    check("...but below the absolute floor it abstains rather than pretending",
          sep._cepstral_profile(v[0] * 1e-4, SR) is None)

    # -- 6. the anchor rescues a shared-pause boundary ---------------------- #
    # The scenario from the docstring: the overlap lands where nobody speaks, so
    # the waveform cannot decide -- but the chunk BODY has both voices, swapped.
    # Without the anchor this is a 50/50 that propagates to the end of the clip.
    #
    # The two pauses must be INDEPENDENT.  A first draft of this fixture reused
    # one noise array on both sides of the overlap, which made the correlation
    # perfect and identity-preserving by construction -- the fixture passed the
    # waveform rung with margin 1.98 and never reached the anchor at all.  A
    # shared pause means both speakers are quiet, not that both channels contain
    # the same samples.
    pause_a = 0.01 * rng.normal(size=(2, ov))
    pause_b = 0.01 * rng.normal(size=(2, ov))
    prev_p = np.concatenate([v[:, :SR], pause_a], axis=1)
    cur_p = np.concatenate([pause_b, v[::-1, SR:3 * SR]], axis=1)
    anch = sep._IdentityAnchor(2)
    anch.observe(v, SR)                              # what we have heard so far
    out, d = sep._align_permutation(cur_p, prev_p, ov, anchor=anch, sample_rate=SR)
    check("a shared-pause boundary is undecidable from the waveform alone",
          d["margin"] < sep.PERM_MARGIN, f"margin={d['margin']}")
    check("the identity anchor breaks the tie the waveform could not",
          d["basis"] == "identity" and d["order"] == [1, 0] and d["confident"],
          f"basis={d['basis']} order={d['order']} "
          f"id_margin={d.get('id_margin')}")
    check("...and the rescued order is actually applied to the samples",
          np.allclose(out, cur_p[[1, 0]], atol=1e-12))

    # And the same boundary with NO anchor must hold and admit it -- this is the
    # baseline the anchor is measured against, not a hypothetical.
    _, d0 = sep._align_permutation(cur_p, prev_p, ov, sample_rate=SR)
    check("without the anchor the same boundary holds and reports basis=hold",
          d0["basis"] == "hold" and not d0["confident"] and not d0["flipped"],
          f"basis={d0['basis']} margin={d0['margin']}")

    # True silence in the overlap takes the RMS early-out, which must still fall
    # through to the anchor rather than returning early.  That early return was
    # the old code's behaviour and it is what made a silent boundary unrescuable.
    sil = np.zeros((2, ov))
    out_s, ds = sep._align_permutation(
        np.concatenate([sil, v[::-1, SR:3 * SR]], axis=1),
        np.concatenate([v[:, :SR], sil], axis=1), ov,
        anchor=anch, sample_rate=SR)
    check("a SILENT overlap still reaches the anchor instead of returning early",
          ds["basis"] == "identity" and ds["order"] == [1, 0],
          f"basis={ds['basis']} order={ds['order']} margin={ds['margin']}")

    # -- 7. an anchor with nothing to say must not decide anything ---------- #
    # A weak/degenerate anchor is more dangerous than no anchor: it reintroduces
    # the coin flip from a new direction while reporting itself confident.
    #
    # This check drove a design change.  With a margin threshold ALONE it failed:
    # a noise-built anchor scored id_margin 0.53 against PERM_ID_MARGIN, because
    # profiles are L2-normalised, so a profile accumulated from noise is a unit
    # vector in a random direction and its 2x2 margin is large by amplification.
    # Measured null: median 0.287, p90 0.670, max 1.383 over 400 draws -- larger
    # than the 0.573 a genuinely distinct pair of voices scores. No value of a
    # margin threshold separates them. The absolute cosine does (null max 0.428
    # over 800 draws vs 0.547-1.000 for real voices), which is why the gate now
    # requires both.
    anch_noise = sep._IdentityAnchor(2)
    anch_noise.observe(0.05 * rng.normal(size=(2, 3 * SR)), SR)
    _, dn = sep._align_permutation(noise.copy(), 0.02 * rng.normal(size=(2, 2 * SR)),
                                   ov, anchor=anch_noise, sample_rate=SR)
    check("a noise-only anchor does not manufacture a confident decision",
          dn["basis"] == "hold" and not dn["confident"],
          f"basis={dn['basis']} id_margin={dn.get('id_margin')} "
          f"id_cos={dn.get('id_cos')}")
    check("...and the reason it declines is the COSINE, not the margin",
          dn.get("id_margin", 0.0) >= sep.PERM_ID_MARGIN
          and dn.get("id_cos", 1.0) < sep.PERM_ID_COS,
          f"id_margin={dn.get('id_margin')} >= {sep.PERM_ID_MARGIN}, "
          f"id_cos={dn.get('id_cos')} < {sep.PERM_ID_COS}")

    # The false-action rate of the joint gate against the pure-noise null.  This
    # is the number that licenses the rung: 0/N, not "the threshold looks safe".
    n_draws, false_act = 120, 0
    for seed in range(n_draws):
        r = np.random.default_rng(seed)
        an = sep._IdentityAnchor(2)
        an.observe(0.05 * r.normal(size=(2, 3 * SR)), SR)
        _, dd = sep._align_permutation(
            0.02 * r.normal(size=(2, 2 * SR)), 0.02 * r.normal(size=(2, 2 * SR)),
            ov, anchor=an, sample_rate=SR)
        if dd["basis"] == "identity":
            false_act += 1
    check("the joint identity gate never fires on pure noise",
          false_act == 0, f"{false_act}/{n_draws} false actions")

    # -- 7b. acoustically SIMILAR voices must abstain, not guess ------------ #
    # The opposite failure, and the more dangerous one: two same-sex voices give
    # a cosine matrix that is high everywhere, so an absolute-cosine gate alone
    # would let it act -- confidently, and measurably sometimes wrong.  The
    # margin is what refuses here (measured median 0.00-0.02 across SNRs).
    sim = two_voices(4 * SR, ((110.0, 700.0), (125.0, 820.0)))
    anch_sim = sep._IdentityAnchor(2)
    anch_sim.observe(sim, SR)
    s_sim = anch_sim.score(sim[::-1, SR:3 * SR].copy(), SR)
    o_sim, m_sim = sep._pairing_margin(s_sim)
    cos_sim = min(float(s_sim[i, o_sim[i]]) for i in range(2))
    check("similar voices clear the cosine floor but NOT the margin",
          cos_sim >= sep.PERM_ID_COS and m_sim < sep.PERM_ID_MARGIN,
          f"cos={cos_sim:.3f} margin={m_sim:.3f}")

    # -- 8. margin agrees with brute force over all permutations ----------- #
    def brute(m: np.ndarray) -> float:
        from itertools import permutations
        tot = sorted((sum(m[i, p[i]] for i in range(m.shape[0])), p)
                     for p in permutations(range(m.shape[0])))
        return float(tot[-1][0] - tot[-2][0])

    agree = True
    for seed in range(300):
        r = np.random.default_rng(10_000 + seed)
        m = r.normal(size=(int(r.integers(2, 5)),) * 2)
        if abs(sep._pairing_margin(m)[1] - brute(m)) > 1e-9:
            agree = False
            break
    check("_pairing_margin agrees with brute force over all permutations", agree)
    check("an all-equal matrix reads margin 0.0, not a tie-break",
          sep._pairing_margin(np.full((3, 3), 0.5))[1] == 0.0)
    # The property the old best-minus-trace lacked: identity winning decisively
    # must NOT read 0.0.
    strong = np.array([[0.9, 0.1], [0.1, 0.9]])
    check("identity winning decisively reads a large margin (the old bug)",
          sep._pairing_margin(strong)[1] > 0.5,
          f"margin={sep._pairing_margin(strong)[1]:.3f}")


def main() -> None:
    print("audio-visual fusion contract checks")
    check_planning()
    check_abstention()
    check_gate_dwell()
    check_gate_equivalence()
    check_mono_and_empty()
    check_serialization()
    check_confidence_honesty()
    check_null_calibration()
    check_separation_alignment()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
