"""Why is a track's audio-visual confidence low?  Answer it with numbers.

    .venv\\Scripts\\python.exe scripts\\diag_match.py runs/<job_id>

``confidence`` is a MARGIN -- the winning stem's score minus the runner-up's --
so a low value has three completely different causes that the single number
cannot distinguish:

  1. **Dead lip signal.**  The face was found but the mouth never moved in the
     measurement: tracking dropouts (NaN), a face too small for FaceMesh, or a
     profile view.  Both scores are then near zero and so is their difference.
  2. **Degenerate scores.**  The lip signal is alive but correlates equally with
     both stems -- two people talking over each other, or a lip signal that is
     really tracking head motion, which both envelopes share.
  3. **Genuinely close call.**  Both stems really do fit, and the assignment is
     a coin flip.  This is the only case where low confidence means what it
     looks like.

Case 1 is a vision bug, case 2 a feature bug, case 3 a hard clip.  The fix is
different in each, so this script separates them before anything is tuned.

Re-runs vision on the job's own input video, so the numbers describe the clip
you actually heard rather than a re-encode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from app.config import CONFIG                                   # noqa: E402
from app import matching                                        # noqa: E402
from _job import pipeline_video                                 # noqa: E402


def _bar(v: float, lo: float, hi: float, w: int = 28) -> str:
    if not np.isfinite(v) or hi <= lo:
        return " " * w
    f = min(1.0, max(0.0, (v - lo) / (hi - lo)))
    n = int(round(f * w))
    return "#" * n + "-" * (w - n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", help="a directory under runs/")
    ap.add_argument("--stems", default="stems_raw",
                    help="which stem file to correlate against (default raw: the "
                         "gate has not yet zeroed anything, so the envelope is intact)")
    ap.add_argument("--video", default=None,
                    help="override the video file. Exists so the frame-rate "
                         "sensitivity claim in docs/DIAG_MATCHER.md is "
                         "reproducible: pass input.mp4 to analyse the 29.98 fps "
                         "upload against the same stems the 25 fps decode used, "
                         "which is the only way to compare the two decodes in ONE "
                         "index space.")
    args = ap.parse_args()

    job = Path(args.job_dir)
    if not job.is_dir():
        print(f"no such directory: {job}")
        return 1

    video = Path(args.video) if args.video else pipeline_video(job)

    stems_path = job / f"{args.stems}.wav"
    if not stems_path.exists():
        print(f"missing {stems_path}")
        return 1

    x, sr = sf.read(str(stems_path), dtype="float32", always_2d=True)
    stems = x.T                                     # (channels, samples)

    print("=" * 78)
    print(f"match diagnosis -- {job.name}")
    print(f"  video {video.name} | stems {stems_path.name} "
          f"{stems.shape[0]}ch {stems.shape[1] / sr:.2f}s @ {sr} Hz")
    print("=" * 78)

    from app.vision import FaceAnalyzer, resample_lip
    fa = FaceAnalyzer(CONFIG.vision)
    tracks, vmeta = fa.analyze(str(video))
    print(f"\nbackend {vmeta['backend']} | {vmeta['n_frames']} frames "
          f"@ {vmeta['fps']:.2f} fps | {vmeta['width']}x{vmeta['height']} "
          f"| {len(tracks)} track(s)\n")

    if not tracks:
        print("no tracks -- vision found nothing, so matching never had a chance")
        return 1

    # ---- per-track lip health ------------------------------------------- #
    # A track can be "present" (a box was found) yet carry a dead lip signal:
    # FaceMesh returning NaN inside a box the detector was happy with.  Those
    # are different failures and the presence count alone hides the second.
    print("track health")
    print(f"  {'track':<10} {'frames':>7} {'lip ok':>8} {'NaN':>6} "
          f"{'box area':>9} {'lip mean':>9} {'lip std':>9} {'dyn range':>10}")
    lip_stats = []
    for t in tracks:
        lip = np.asarray(t.lip, dtype=np.float64)
        finite = np.isfinite(lip)
        vals = lip[finite]
        boxes = [b for b in t.boxes if b is not None]
        area = float(np.mean([b[2] * b[3] for b in boxes])) if boxes else 0.0
        dyn = float(vals.max() - vals.min()) if vals.size else 0.0
        lip_stats.append({
            "label": t.label(), "n": int(finite.sum()), "total": len(lip),
            "nan_frac": float(1.0 - finite.mean()) if lip.size else 1.0,
            "box_area": area, "mean": float(vals.mean()) if vals.size else 0.0,
            "std": float(vals.std()) if vals.size else 0.0, "dyn": dyn,
        })
        s = lip_stats[-1]
        print(f"  {t.label():<10} {len(lip):>7} {s['n']:>8} {s['nan_frac']:>5.0%} "
              f"{area:>9.4f} {s['mean']:>9.5f} {s['std']:>9.5f} {dyn:>10.5f}")

    # A face occupying <2% of the frame is ~90px on a 640x360 clip; FaceMesh is
    # unreliable below roughly that, and it is the usual cause of a dead signal.
    for s in lip_stats:
        if s["box_area"] < 0.02:
            print(f"  !! {s['label']}: box is {s['box_area']:.1%} of frame -- "
                  f"likely too small for reliable FaceMesh landmarks")
        if s["nan_frac"] > 0.25:
            print(f"  !! {s['label']}: {s['nan_frac']:.0%} of frames have no "
                  f"landmarks -- tracking is dropping out")
        if s["std"] < CONFIG.gate.visual_min_dynamic_range:
            print(f"  !! {s['label']}: lip signal is essentially CONSTANT "
                  f"(std {s['std']:.2e}) -- carries no speech information")

    # ---- the score matrix ------------------------------------------------ #
    m = matching.match_stems_to_tracks(
        stems, tracks, sample_rate=sr, video_fps=vmeta["fps"], cfg=CONFIG.match)
    assignment, confidence, score = m.assignment, m.confidence, m.score
    print(f"\npairing significance p={m.significance:.4f} "
          f"({m.null_agreement:.1%} of shifted-lip nulls reproduce this pairing) "
          + ("-- beats its null" if m.trustworthy(CONFIG.match)
             else "-- NOT distinguishable from chance"))

    print(f"\nscore matrix (lip-derivative . stem-envelope, z-scored)")
    hdr = "  " + " " * 12 + "".join(f"{'stem ' + str(j):>12}" for j in range(stems.shape[0]))
    print(hdr)
    for i, t in enumerate(tracks):
        row = "".join(f"{score[i, j]:>12.4f}" for j in range(score.shape[1]))
        print(f"  {t.label():<12}{row}")

    print(f"\n  {'track':<12} {'-> stem':>8} {'confidence':>11} {'margin':>9}  verdict")
    for i, t in enumerate(tracks):
        srow = np.sort(score[i])[::-1]
        margin = float(srow[0] - srow[1]) if srow.size > 1 else float(srow[0])
        # Separate the three causes named in the docstring.
        st = lip_stats[i]
        if st["std"] < CONFIG.gate.visual_min_dynamic_range or st["nan_frac"] > 0.5:
            verdict = "DEAD LIP SIGNAL (vision)"
        elif abs(srow[0]) < 0.02 and abs(srow[1]) < 0.02:
            verdict = "DEGENERATE (lip uncorrelated with either stem)"
        elif margin < CONFIG.match.min_confidence:
            verdict = "CLOSE CALL (both stems fit)"
        else:
            verdict = "ok"
        print(f"  {t.label():<12} {assignment[i]:>8} {confidence[i]:>11.4f} "
              f"{margin:>9.4f}  {verdict}")

    # ---- is the lip signal even alive over time? ------------------------- #
    # Correlation is a single scalar over the whole clip, which hides a signal
    # that is healthy for 2 s and flat for 8.  Show it.
    envs = [matching._zscore(matching.energy_envelope(
        s, sr, CONFIG.match.env_frame_ms, CONFIG.match.smooth_kernel)) for s in stems]
    n_frames = min(len(e) for e in envs)
    env_fps = 1000.0 / CONFIG.match.env_frame_ms

    print(f"\nactivity over time ({n_frames} frames @ {env_fps:.0f} fps, "
          f"one column per second)")
    sec = int(env_fps)
    n_sec = n_frames // sec
    for j, e in enumerate(envs):
        rms = [float(np.mean(np.abs(e[k * sec:(k + 1) * sec]))) for k in range(n_sec)]
        print(f"  stem {j} env  " + "".join(
            " .:-=+*#%@"[min(9, int(v / (max(rms) + 1e-9) * 9))] for v in rms))
    for i, t in enumerate(tracks):
        r = resample_lip(t.lip, vmeta["fps"], n_frames, env_fps)
        d = np.abs(np.diff(r, prepend=r[:1]))
        rms = [float(np.mean(d[k * sec:(k + 1) * sec])) for k in range(n_sec)]
        mx = max(rms) + 1e-12
        print(f"  {t.label():<11} " + "".join(
            " .:-=+*#%@"[min(9, int(v / mx * 9))] for v in rms))

    out = job / "diag_match.json"
    out.write_text(json.dumps({
        "backend": vmeta["backend"], "fps": vmeta["fps"],
        "tracks": lip_stats, "score": score.tolist(),
        "assignment": [int(a) for a in assignment],
        "confidence": [float(c) for c in confidence],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
