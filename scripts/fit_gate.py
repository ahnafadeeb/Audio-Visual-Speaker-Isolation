"""Fit the gate thresholds against a REAL clip instead of the synthetic bench.

    .venv\\Scripts\\python.exe scripts\\fit_gate.py runs/<job_id>

The constants in ``GateConfig`` were fitted on synthetic two-tone signals where
"silence" meant a literal zero region.  Real SepFormer residual does not behave
that way: measured on a 640x360 two-person clip, both stems sat above
``open_db = -30`` for ~75% of frames, so the gate held both channels open
through 55% of the clip and the interferer stayed audible at a median of
-33.7 dB.  That is the ghost whisper, arriving through a gate that never closed.

The tempting fix -- lower ``open_db`` until the whisper disappears -- is the
move that killed the two earlier gate attempts, because the interferer's
residual and the target's own quiet speech live at *the same level*.  There is
no threshold that separates them on level alone; that is precisely why the
fusion veto exists.  So this sweep reports BOTH sides of the trade at every
setting and refuses to collapse them into one score:

    silence  -- fraction of the interferer's frames driven to bit-exact zero
                while the OTHER speaker holds the floor  (want high)
    retain   -- fraction of the target's own speech frames that survive
                (want high; this is what over-tightening destroys)

A setting is only interesting if it improves silence without moving retain.
The knee is where silence stops rising and retain starts falling.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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


def score(stems_raw: np.ndarray, gated: np.ndarray, sr: int,
          frame_ms: float = 10.0) -> tuple[float, float]:
    """(silence, retain) for a 2-stem set, averaged over both directions.

    Both are measured on the SAME frame grid, and only in regions where the
    ground truth is unambiguous:

      * silence is counted only where the other speaker is clearly active and
        this one is clearly not -- the frames where a listener would notice a
        whisper;
      * retain is counted only where THIS speaker is clearly active -- the
        frames where cutting is audible as a clipped word.

    Frames where both or neither are active are excluded from both numbers:
    they are genuinely ambiguous, and letting them into either metric lets a
    setting look good by being lucky about overlap.
    """
    n = max(1, int(sr * frame_ms / 1000.0))
    E = [np.sqrt((_frames(s, n) ** 2).mean(1)) + 1e-12 for s in stems_raw]
    G = [_frames(g, n) for g in gated]
    nz = [(np.abs(f) > 0).any(1) for f in G]            # frame passed anything
    k = min(len(E[0]), len(E[1]), len(nz[0]), len(nz[1]))

    sil, ret = [], []
    for i in (0, 1):
        j = 1 - i
        ei, ej = E[i][:k], E[j][:k]
        # "clearly active" = within 20 dB of the stem's own 95th percentile.
        ai = ei > np.percentile(ei, 95) * 10 ** (-20 / 20)
        aj = ej > np.percentile(ej, 95) * 10 ** (-20 / 20)
        interferer = aj & ~ai                            # other talks, this one not
        target = ai & ~aj                                # this one talks alone
        if interferer.any():
            sil.append(float((~nz[i][:k][interferer]).mean()))
        if target.any():
            ret.append(float(nz[i][:k][target].mean()))
    return (float(np.mean(sil)) if sil else 0.0,
            float(np.mean(ret)) if ret else 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir")
    ap.add_argument("--no-visual", action="store_true",
                    help="sweep the pure-acoustic gate (isolates what level alone can do)")
    ap.add_argument("--symmetric", action="store_true",
                    help="let the differential veto go negative (protect as well as punish)")
    args = ap.parse_args()

    job = Path(args.job_dir)
    x, sr = sf.read(str(job / "stems_raw.wav"), dtype="float32", always_2d=True)
    stems_raw = x.T
    if stems_raw.shape[0] != 2:
        print("this sweep assumes 2 stems")
        return 1

    lips = None
    fps = 25.0
    video = None if args.no_visual else pipeline_video(job)
    if video is not None:
        from app.vision import FaceAnalyzer
        from app.pipeline import lips_by_stem
        fa = FaceAnalyzer(CONFIG.vision)
        tracks, vm = fa.analyze(str(video))
        fps = vm["fps"]
        m = matching.match_stems_to_tracks(
            stems_raw, tracks, sample_rate=sr, video_fps=fps, cfg=CONFIG.match)
        assignment, conf = m.assignment, m.confidence
        lips = lips_by_stem(assignment, tracks, stems_raw.shape[0])
        print(f"vision: {vm['backend']} {len(tracks)} tracks, assignment {assignment}, "
              f"confidence {[round(c, 3) for c in conf]}, p={m.significance:.3f}")

    base = replace(CONFIG.gate, visual_symmetric=args.symmetric)
    if args.symmetric:
        print("symmetric differential veto: shift in [-veto, +veto]")
    s0, r0 = score(stems_raw, dsp.apply_gate(stems_raw, sample_rate=sr, cfg=base,
                                             lips=lips, video_fps=fps), sr)
    print("=" * 74)
    print(f"baseline  open {base.open_db:+.0f}  close {base.close_db:+.0f}  "
          f"veto {base.visual_veto_db:.0f} dB"
          f"   ->  silence {s0:6.1%}   retain {r0:6.1%}")
    print("=" * 74)

    # The hysteresis BAND is swept, not fixed at 10 dB.  Fixing it was a blind
    # spot in the first version of this sweep: diag_sir.py showed the residual
    # spans p50 -31.4 to p90 -21.5 dB relative to the stem's own p95, i.e. the
    # residual distribution and a 10 dB band are the same width.  The gate then
    # opens on the residual's loud tail and hysteresis holds it open across the
    # bulk of the residual, which is why silence stalled near 55% even with
    # open_db sitting inside the measured 5 dB headroom.  A narrower band lets
    # the gate fall shut again between residual peaks.
    print(f"\n{'open':>6} {'close':>6} {'band':>5} {'veto':>6} "
          f"{'silence':>9} {'retain':>8}   knee")
    best = None
    for open_db in (-22.0, -20.0, -18.0, -16.0):
        for band in (2.0, 3.0, 4.0, 6.0):
            for veto in ((base.visual_veto_db,) if lips is None else (0.0, 2.0, 4.0, 6.0)):
                cfg = replace(base, open_db=open_db, close_db=open_db - band,
                              visual_veto_db=veto)
                g = dsp.apply_gate(stems_raw, sample_rate=sr, cfg=cfg,
                                   lips=lips, video_fps=fps)
                s, r = score(stems_raw, g, sr)
                # Only count settings that do not sacrifice intelligibility. 2%
                # of the target's own frames is about one clipped consonant per
                # clip; beyond that the cure is worse than the whisper.
                ok = r >= r0 - 0.02
                mark = ""
                if ok and (best is None or s > best[0]):
                    best = (s, r, cfg)
                    mark = "<-- best so far"
                print(f"{open_db:>6.0f} {open_db - band:>6.0f} {band:>5.0f} "
                      f"{veto:>6.0f} {s:>9.1%} {r:>8.1%}   "
                      f"{mark if ok else 'retain loss'}")

    if best:
        s, r, cfg = best
        print("\n" + "=" * 74)
        print(f"best:  open_db {cfg.open_db:+.0f}  close_db {cfg.close_db:+.0f}  "
              f"visual_veto_db {cfg.visual_veto_db:.0f}")
        print(f"       silence {s0:.1%} -> {s:.1%}    retain {r0:.1%} -> {r:.1%}")
        print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
