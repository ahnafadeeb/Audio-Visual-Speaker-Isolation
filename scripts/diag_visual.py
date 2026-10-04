"""Does the visual veto carry ANY separating information on this clip?

    .venv\\Scripts\\python.exe scripts\\diag_visual.py runs/<job_id>

The gate sweep (scripts/fit_gate.py) produced an uncomfortable result: on a real
640x360 clip the pure-acoustic gate at ``open_db = -18`` beat the audio-visual
gate at ``-30`` on *both* axes -- more silence AND more retained speech.  A
fusion term that loses on both axes is not fusing; it is adding noise.

But "the veto hurts" has two very different causes, and the sweep cannot tell
them apart:

  A. **The signal is uninformative.**  ``visual_activity`` is near-random with
     respect to who is actually speaking, so any weight on it is harmful and the
     only correct ``visual_veto_db`` is 0.
  B. **The signal is informative but mis-scaled.**  It separates the classes,
     but 25 dB of authority over-reacts to its noise, so it is right on average
     and catastrophic on the tail.

Cause A means abandon the veto on clips like this.  Cause B means shrink it.
The decisive measurement is not a sweep, it is the separability of the signal
itself, which is independent of any threshold:

    AUC = P(v_target > v_interferer)   for randomly drawn frames of each class

AUC 0.5 is a coin flip (cause A).  AUC 0.7+ is real information that the gate is
squandering (cause B).  Ground truth comes from the stems themselves, using the
same unambiguous-frame definition as fit_gate.score -- frames where exactly one
speaker is clearly active.  That is circular *only* if the stems are wrong about
who spoke, which cross-correlation 0.03 and the boundaries/flipped counters
already rule out.

Vision is the slow part, so the resampled lip signals are cached to
``lips.npz`` in the job dir and reused.  Delete that file after changing
anything in app/vision.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from app import dsp, matching                                   # noqa: E402
from app.config import CONFIG                                   # noqa: E402
from _job import pipeline_video                                 # noqa: E402


def _frames(v: np.ndarray, n: int) -> np.ndarray:
    m = len(v) // n * n
    return v[:m].reshape(-1, n)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random pos > random neg), ties counted as half.

    Computed via the rank-sum identity rather than by sweeping thresholds, so
    it is exact and needs no bin choice -- which matters here because ``v`` is
    clipped at 1.0 and therefore has a large atom of ties at the ceiling that a
    binned ROC would smear.
    """
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    # average ranks within tie groups
    both = np.concatenate([pos, neg])[order]
    i = 0
    while i < both.size:
        j = i
        while j + 1 < both.size and both[j + 1] == both[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def load_lips(job: Path, sr: int, n_stems: int):
    """Lip signals in STEM order, plus the source fps.  Cached."""
    cache = job / "lips.npz"
    if cache.exists():
        z = np.load(str(cache), allow_pickle=True)
        return [z[f"lip{j}"] for j in range(n_stems)], float(z["fps"])

    video = pipeline_video(job)

    from app.vision import FaceAnalyzer
    from app.pipeline import lips_by_stem

    x, _ = sf.read(str(job / "stems_raw.wav"), dtype="float32", always_2d=True)
    stems = x.T
    fa = FaceAnalyzer(CONFIG.vision)
    tracks, vm = fa.analyze(str(video))
    m = matching.match_stems_to_tracks(
        stems, tracks, sample_rate=sr, video_fps=vm["fps"], cfg=CONFIG.match)
    assignment, conf = m.assignment, m.confidence
    lips = lips_by_stem(assignment, tracks, n_stems)
    print(f"vision: {vm['backend']} {len(tracks)} tracks, assignment {assignment}, "
          f"confidence {[round(c, 3) for c in conf]}, p={m.significance:.3f}")

    out = {"fps": np.float64(vm["fps"])}
    for j, lp in enumerate(lips):
        out[f"lip{j}"] = np.asarray(lp if lp is not None else [], dtype=np.float64)
    np.savez(str(cache), **out)
    return [out[f"lip{j}"] for j in range(n_stems)], float(vm["fps"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir")
    args = ap.parse_args()

    job = Path(args.job_dir)
    x, sr = sf.read(str(job / "stems_raw.wav"), dtype="float32", always_2d=True)
    stems = x.T
    if stems.shape[0] != 2:
        print("this diagnostic assumes 2 stems")
        return 1

    g = CONFIG.gate
    lips, fps = load_lips(job, sr, 2)

    # Gate-frame grid, identical to apply_gate's.
    n = max(1, int(sr * g.frame_ms / 1000.0))
    E = [np.sqrt((_frames(s, n) ** 2).mean(1)) + 1e-12 for s in stems]
    n_frames = min(len(E[0]), len(E[1]))

    V, VALID = [], []
    for j in range(2):
        v, valid = dsp.visual_activity(
            lips[j], src_fps=fps, n_frames=n_frames, frame_ms=g.frame_ms,
            smooth_ms=g.visual_smooth_ms, hold_ms=g.visual_hold_ms,
            min_dynamic_range=g.visual_min_dynamic_range)
        V.append(v)
        VALID.append(valid)

    print("=" * 78)
    print(f"visual veto separability -- {job.name}   ({n_frames} gate frames "
          f"@ {g.frame_ms:.0f} ms)")
    print("=" * 78)

    print(f"\n{'stem':<6} {'valid':>7} {'v=0':>7} {'v=1':>7} {'mean v':>8} "
          f"{'p50':>7} {'p90':>7}")
    for j in range(2):
        v, ok = V[j], VALID[j]
        print(f"{j:<6} {ok.mean():>6.1%} {(v <= 0).mean():>6.1%} "
              f"{(v >= 1).mean():>6.1%} {v.mean():>8.3f} "
              f"{np.percentile(v, 50):>7.3f} {np.percentile(v, 90):>7.3f}")

    # Ground truth: exactly one speaker clearly active (same rule as fit_gate).
    print(f"\n{'stem':<6} {'n_target':>9} {'n_interf':>9} {'v|target':>9} "
          f"{'v|interf':>9} {'gap':>7} {'AUC':>7}   verdict")
    aucs = []
    for i in range(2):
        j = 1 - i
        ei, ej = E[i][:n_frames], E[j][:n_frames]
        ai = ei > np.percentile(ei, 95) * 10 ** (-20 / 20)
        aj = ej > np.percentile(ej, 95) * 10 ** (-20 / 20)
        tgt = ai & ~aj
        itf = aj & ~ai
        vt, vi = V[i][:n_frames][tgt], V[i][:n_frames][itf]
        a = auc(vt, vi)
        aucs.append(a)
        if not np.isfinite(a):
            verdict = "no usable frames"
        elif a < 0.55:
            verdict = "COIN FLIP -- veto carries no information"
        elif a < 0.65:
            verdict = "weak"
        elif a < 0.75:
            verdict = "usable"
        else:
            verdict = "strong"
        print(f"{i:<6} {tgt.sum():>9} {itf.sum():>9} {vt.mean():>9.3f} "
              f"{vi.mean():>9.3f} {vt.mean() - vi.mean():>7.3f} {a:>7.3f}   {verdict}")

    m = float(np.nanmean(aucs))
    print(f"\nmean AUC {m:.3f}")

    # What the veto actually costs on target frames.  effective_open rises by
    # visual_veto_db * (1 - v); on frames where the target IS speaking, every
    # dB of that is a dB of its own speech put at risk.
    print(f"\nveto pressure on the target's OWN speech "
          f"(effective_open rise = veto_db * (1-v)):")
    print(f"  {'stem':<6} {'mean rise':>11} {'p90 rise':>10}   "
          f"(at visual_veto_db = {g.visual_veto_db:.0f})")
    for i in range(2):
        j = 1 - i
        ei, ej = E[i][:n_frames], E[j][:n_frames]
        ai = ei > np.percentile(ei, 95) * 10 ** (-20 / 20)
        aj = ej > np.percentile(ej, 95) * 10 ** (-20 / 20)
        tgt = ai & ~aj
        rise = g.visual_veto_db * (1.0 - V[i][:n_frames][tgt])
        print(f"  {i:<6} {rise.mean():>10.1f} dB {np.percentile(rise, 90):>9.1f} dB")

    # ---- would a RELATIVE veto separate better? -------------------------- #
    #
    # `veto * (1 - v)` asks "is this mouth moving?", which needs the absolute
    # scale of v to be trustworthy -- and it is not: v|target is 0.69, not 1.0,
    # so two thirds of the penalty charged to the target's own speech is
    # common-mode, shared with the interferer and carrying no information.
    # Most of what corrupts v is common to both faces (camera shake, global
    # lighting, encoder noise, the detector's own box jitter), so a DIFFERENCE
    # between the two tracks cancels it.  For a 2-speaker task that difference
    # is also the question the gate actually has to answer.
    print("\n" + "-" * 78)
    print("alternative formulations (AUC on the same frames)")
    print("-" * 78)
    print(f"  {'formulation':<34} {'stem 0':>8} {'stem 1':>8} {'mean':>8}")

    def eval_signal(sig, label):
        out = []
        for i in range(2):
            j = 1 - i
            ei, ej = E[i][:n_frames], E[j][:n_frames]
            ai = ei > np.percentile(ei, 95) * 10 ** (-20 / 20)
            aj = ej > np.percentile(ej, 95) * 10 ** (-20 / 20)
            s = sig(i, j)
            out.append(auc(s[ai & ~aj], s[aj & ~ai]))
        mm = float(np.nanmean(out))
        print(f"  {label:<34} {out[0]:>8.3f} {out[1]:>8.3f} {mm:>8.3f}")
        return mm

    eval_signal(lambda i, j: V[i][:n_frames], "absolute      v_i          (current)")
    rel = eval_signal(lambda i, j: V[i][:n_frames] - V[j][:n_frames],
                      "relative      v_i - v_j")
    eval_signal(lambda i, j: V[i][:n_frames] / (V[i][:n_frames] + V[j][:n_frames] + 1e-9),
                "normalised    v_i/(v_i+v_j)")

    # ---- is the smoothing/dilation smearing the classes together? -------- #
    #
    # hold_ms dilates in BOTH directions, so 250 ms is a 500 ms window: at a
    # conversational turn rate that reaches across the boundary and paints the
    # interferer's frames with the target's motion.  It was fitted on synthetic
    # audio with long, clean turns, where that could not happen.
    print("\n" + "-" * 78)
    print("does the temporal smoothing smear the classes together?")
    print("-" * 78)
    print(f"  {'smooth_ms':>9} {'hold_ms':>8}   {'abs AUC':>8} {'rel AUC':>8}")
    for smooth_ms in (90.0, 150.0, 220.0, 300.0, 400.0):
        for hold_ms in (120.0, 250.0, 400.0):
            W = []
            for jj in range(2):
                vv, _ = dsp.visual_activity(
                    lips[jj], src_fps=fps, n_frames=n_frames, frame_ms=g.frame_ms,
                    smooth_ms=smooth_ms, hold_ms=hold_ms,
                    min_dynamic_range=g.visual_min_dynamic_range)
                W.append(vv)
            a_abs, a_rel = [], []
            for i in range(2):
                j = 1 - i
                ei, ej = E[i][:n_frames], E[j][:n_frames]
                ai = ei > np.percentile(ei, 95) * 10 ** (-20 / 20)
                aj = ej > np.percentile(ej, 95) * 10 ** (-20 / 20)
                a_abs.append(auc(W[i][ai & ~aj], W[i][aj & ~ai]))
                d = W[i] - W[j]
                a_rel.append(auc(d[ai & ~aj], d[aj & ~ai]))
            print(f"  {smooth_ms:>9.0f} {hold_ms:>8.0f}   "
                  f"{np.nanmean(a_abs):>8.3f} {np.nanmean(a_rel):>8.3f}")

    print()
    if rel > m + 0.03:
        print(f"=> The RELATIVE form separates better ({rel:.3f} vs {m:.3f}).  The")
        print("   absolute veto is spending most of its authority on common-mode")
        print("   motion that both faces share.")
    if m < 0.55:
        print("=> The lip signal does not separate speaker from interferer on this")
        print("   clip.  No value of visual_veto_db helps; the acoustic gate is the")
        print("   whole gate.  Set visual_veto_db low or off and fit open_db.")
    elif m < 0.65:
        print("=> Weak but non-zero.  The veto should be sized well below 25 dB --")
        print("   enough to break ties, not enough to override the acoustic gate.")
    else:
        print("=> The signal separates.  The 25 dB authority is not the problem;")
        print("   look at how the rise is applied on the target's own frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
