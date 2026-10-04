"""The +40 dB ghost test: is the silence real, on a real run?

``redteam_silence.py`` attacks the gate with synthetic signals it designs to
break. This asks the complementary question about a clip that actually shipped:
take a run directory, find the stretches where the target demonstrably is not
speaking, amplify them by 40 dB, and listen for a ghost.

    python scripts/ghost_test.py runs/v3_e2e
    python scripts/ghost_test.py runs/v3_e2e --write      # writes the +40 dB audio

Why this exists rather than reading ``meta.silence.whisper_db``
---------------------------------------------------------------
``whisper_db`` cannot answer this, for a structural reason worth stating once.
It averages the frames inside a window defined by ``active_db = 20.0`` -- the
*same* 20 dB point as the gate's ``open_db``. So it measures exactly the frames
the gate is deliberately undecided about, and reads near -20 dB whether the
gate is working or not. A metric whose window boundary coincides with the
decision boundary it is auditing reports the boundary, not the decision.

The fix is clearance, in both axes the gate decides on.

**Level.** Zones sit at least ``CLEARANCE_DB`` away from the -20/-23 dB
hysteresis band. The band plus that margin is AMBIGUOUS and is scored as
nothing. A frame must be an order of magnitude in power clear of the gate's
opinion before this script forms one.

**Time.** The gate also decides in time -- ``min_on_ms`` holds it open through
brief dips -- so a level-only definition is not enough. Measured on the
reference clip, a level-only PAUSE reads 93.9% zero on channel 1 with a
-32.9 dBFS peak, which looks like a leak and is not: the channel is purity
1.000, and those frames are the stop closures and inter-syllable dips *inside*
words, which the dwell correctly holds open. Zeroing them would be the defect.
So a pause must also be sustained (``--min-pause``) and is scored only away
from its own edges (``--margin``), where the ramp and lookahead deliberately
live.

Everything excluded by either rule is counted and printed. The point is to
score the region where "zero" is unambiguously the right answer, not to shrink
the region until the answer is nice -- so the totals are reported alongside,
and ``--min-pause`` can be swept to confirm the verdict does not depend on it.
On the reference clip it does not: 150 / 250 / 400 ms all read 100.000%.

What "should be silent" means here
----------------------------------
Zones come from the *raw* (ungated) stem, because the gated stem cannot be
asked whether it should have been gated. Levels are relative to each stem's own
95th-percentile frame, matching ``GateConfig.ref_percentile`` -- the offsets are
what carry meaning, not the reference.

    SPEECH   rel >= open_db  + CLEARANCE   target talking; output must survive
    PAUSE    rel <= close_db - CLEARANCE, sustained, interior only -> must be 0.0

Both halves are reported. A gate that reaches perfect silence by deleting the
target has not passed, it has moved the failure somewhere quieter.

How much a pass is worth
------------------------
The silence half is partly structural: a scored frame sits 10 dB below
``close_db``, so a working gate is *expected* to have closed there. That is the
design succeeding, not the test flattering it -- but it does mean the silence
row alone is weak evidence. The load-bearing checks are the ones that can
genuinely go either way, and have: the composition invariants, the speech-damage
half (which fails on the v2 path's channel 1 at 3.68 %), and the exclusions,
which is where the first two versions of this script went wrong. Read the
excluded-region peak and the speech column before believing the headline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import dsp  # noqa: E402
from app.config import CONFIG  # noqa: E402

#: Clearance from the gate's hysteresis band, in dB. The band is 3 dB wide
#: (-20 to -23); 10 dB on each side puts every scored frame a full order of
#: magnitude in power away from any threshold this script is auditing.
CLEARANCE_DB = 10.0

#: Amplification for the listening test. 40 dB turns a -70 dBFS whisper into a
#: -30 dBFS one, which is plainly audible on headphones.
AMPLIFY_DB = 40.0

#: A nonzero remainder is still inaudible if it stays below this after
#: amplification. Exact zero is the pass condition; this is the consolation
#: bound for a channel that misses it.
INAUDIBLE_DBFS = -60.0

FINDINGS: list[str] = []


def report(name: str, ok: bool, detail: str, note: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<40} {detail}")
    if note:
        print(f"          {note}")
    if not ok:
        FINDINGS.append(f"{name}: {detail}")


def dbfs(x: np.ndarray) -> float:
    """Peak in dBFS. ``-inf`` for an all-zero signal, which is the point."""
    p = float(np.abs(x).max()) if x.size else 0.0
    return -np.inf if p == 0.0 else 20.0 * np.log10(p)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def zones(x: np.ndarray, sr: int, cfg, min_pause_s: float, margin_s: float):
    """SPEECH and scorable-PAUSE masks for one raw stem, plus what was excluded.

    Frames are built exactly as ``dsp.gate_mask`` builds them, so "rel dB" here
    and "rel dB" there are the same quantity and the clearance is real rather
    than nominal.
    """
    frame = max(1, int(round(sr * cfg.frame_ms / 1000.0)))
    n_frames = int(np.ceil(x.size / frame))
    padded = np.pad(x, (0, n_frames * frame - x.size))
    ldb = 10.0 * np.log10((padded.reshape(n_frames, frame) ** 2).mean(axis=1) + 1e-12)
    rel = ldb - float(np.percentile(ldb, cfg.ref_percentile))

    up = lambda m: np.repeat(m, frame)[:x.size]         # noqa: E731
    speech = up(rel >= cfg.open_db + CLEARANCE_DB)
    quiet = up(rel <= cfg.close_db - CLEARANCE_DB)

    M = int(round(margin_s * sr))
    pause = np.zeros(x.size, dtype=bool)
    n_kept = n_short = 0
    for s, e in runs(quiet):
        if (e - s) / sr < min_pause_s:
            n_short += 1
            continue
        pause[s + M:e - M] = True
        n_kept += 1
    return speech, pause, quiet, n_kept, n_short


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--min-pause", type=float, default=0.150,
                    help="shortest quiet run scored as a pause, seconds "
                         "(default 0.15 = 2.5x min_on_ms; shorter runs are "
                         "intra-word closures, not pauses)")
    ap.add_argument("--margin", type=float, default=None,
                    help="unscored margin at each pause edge, seconds "
                         "(default ramp_ms + lookahead_ms, the deliberate "
                         "transition region)")
    ap.add_argument("--write", action="store_true",
                    help="write ghost_chN.wav, the pause zones at +40 dB, to listen to")
    args = ap.parse_args()

    raw_p, demo_p = args.run_dir / "stems_raw.wav", args.run_dir / "stems_demo.wav"
    if not raw_p.exists() or not demo_p.exists():
        print(f"need stems_raw.wav and stems_demo.wav in {args.run_dir}")
        return 2

    raw, sr = sf.read(str(raw_p), dtype="float32", always_2d=True)
    demo, _ = sf.read(str(demo_p), dtype="float32", always_2d=True)
    raw, demo = raw.T, demo.T                      # -> (n_ch, n_samples)

    meta = {}
    mp = args.run_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))

    cfg = CONFIG.gate
    margin = args.margin if args.margin is not None else \
        (cfg.ramp_ms + cfg.lookahead_ms) / 1000.0
    # Name the non-acoustic suppressor for what this run actually used, so the
    # attribution below is not quietly wrong on one of the two paths.
    other_name = "identity" if meta.get("separator") == "avtse" else "visual veto"

    print("=" * 74)
    print(f"GHOST TEST  {args.run_dir}")
    print("=" * 74)
    print(f"separator {meta.get('separator', '?')}   {raw.shape[0]} channels   "
          f"{raw.shape[1] / sr:.1f} s @ {sr} Hz")
    print(f"gate band {cfg.close_db}/{cfg.open_db} dB rel p95   "
          f"level clearance +-{CLEARANCE_DB:.0f} dB   "
          f"pause >= {args.min_pause * 1000:.0f} ms   edge margin {margin * 1000:.0f} ms")

    # The acoustic gate alone, recomputed, so the two removals can be told
    # apart. Deterministic and cheap, and it is the only way to attribute a
    # zero in demo without guessing: whatever demo removes that this does not
    # is the identity gate's doing.
    acoustic = dsp.apply_gate(raw, sample_rate=sr, cfg=cfg)

    # -- structural invariants, before any zone reasoning ------------------- #
    print("\n[0] composition invariants")
    le = bool(np.all(np.abs(demo) <= np.abs(raw) + 1e-7))
    report("demo is raw attenuated, never amplified", le,
           "demo <= raw pointwise" if le else "demo exceeds raw somewhere")
    sup = bool(np.all(demo[acoustic == 0.0] == 0.0))
    report("every acoustic-gate zero survives into demo", sup,
           "zero sets nest" if sup else "a gated sample came back nonzero",
           "" if sup else "the identity gate must compose with the VAD, not replace it")

    for ch in range(raw.shape[0]):
        r, d = raw[ch], demo[ch]
        speech, pause, quiet, n_kept, n_short = zones(
            r, sr, cfg, args.min_pause, margin)
        excluded = (quiet.sum() - pause.sum()) / sr
        print(f"\n[{ch + 1}] channel {ch}"
              f"   speech {speech.mean() * 100:.1f}%"
              f"   quiet {quiet.sum() / sr:.2f} s"
              f" -> scored {pause.sum() / sr:.2f} s in {n_kept} pauses"
              f"   (excluded {excluded:.2f} s: {n_short} short dips + edge margins)")

        if not pause.any():
            report("has a scorable pause", False,
                   f"no quiet run reaches {args.min_pause * 1000:.0f} ms",
                   "the target never stops talking, so this clip cannot test silence")
            continue

        # -- the requirement ------------------------------------------------ #
        seg = d[pause]
        zfrac = float((seg == 0.0).mean())
        report("pause is bit-exact zero", zfrac >= 0.9999,
               f"{zfrac * 100:7.3f}% of {pause.sum() / sr:.2f} s exactly 0.0")

        amplified = dbfs(seg) + AMPLIFY_DB
        if np.abs(seg).max() == 0.0:
            report(f"+{AMPLIFY_DB:.0f} dB reveals nothing", True,
                   "digital silence -- survives any gain")
        else:
            ok = amplified < INAUDIBLE_DBFS
            report(f"+{AMPLIFY_DB:.0f} dB reveals nothing", ok,
                   f"peak {dbfs(seg):7.1f} dBFS -> {amplified:6.1f} dBFS",
                   "" if ok else "a ghost is audible here at normal listening level")

        # What the margins and short dips actually hold, so excluding them is
        # a stated bound rather than a quiet omission.
        rest = quiet & ~pause
        if rest.any():
            print(f"          excluded regions peak {dbfs(d[rest]):.1f} dBFS "
                  f"-- ramp/lookahead transitions and sub-{args.min_pause * 1000:.0f} ms "
                  f"closures, bounded by close_db by construction")

        # -- the other half: is the target still intact? --------------------- #
        sp = d[speech]
        kept = float((sp != 0.0).mean())
        # Attribute what was removed, because only one of the causes is a
        # defect. `acoustic` above is the pure-acoustic gate, so anything demo
        # removes that it does not is the run's *other* suppressor -- the
        # identity gate on the AV path, the visual veto on the audio-only one.
        # Both are deliberate: a loud stretch that is somebody else lands in
        # SPEECH by level and is supposed to vanish, and counting that as
        # damage would penalise the gate for working.
        by_acoustic = float((acoustic[ch][speech] == 0.0).mean())
        by_other = float(((acoustic[ch][speech] != 0.0) & (sp == 0.0)).mean())
        # So the pass condition is on the acoustic share alone. It is the only
        # removal nobody asked for.
        report("target's own speech survives", by_acoustic <= 0.02,
               f"kept {kept * 100:6.2f}%   removed: {by_acoustic * 100:.2f}% acoustic"
               f" + {by_other * 100:.2f}% {other_name}",
               "" if by_acoustic <= 0.02
               else "the acoustic gate is eating speech, not leakage")

        if args.write:
            g = np.zeros_like(d)
            g[pause] = np.clip(seg * 10 ** (AMPLIFY_DB / 20.0), -1.0, 1.0)
            out = args.run_dir / f"ghost_ch{ch}.wav"
            sf.write(str(out), g, sr, subtype="FLOAT")
            print(f"          wrote {out.name} -- scored pauses at +{AMPLIFY_DB:.0f} dB")

    ident = meta.get("identity")
    if ident:
        print(f"\nidentity (from meta): {ident['n_offscreen']} off-screen speaker(s), "
              f"purity {[round(p, 2) for p in ident['purity']]}, "
              f"muted {[f'{m:.1%}' for m in ident['muted_frac']]}")

    print("\n" + "=" * 74)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for f in FINDINGS:
            print(f"  - {f}")
    else:
        print("no findings -- every scored pause is bit-exact digital silence")
    print("=" * 74)
    return 1 if FINDINGS else 0


if __name__ == "__main__":
    raise SystemExit(main())
