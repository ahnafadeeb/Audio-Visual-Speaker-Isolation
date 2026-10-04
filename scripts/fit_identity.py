"""Fit the identity gate's hyper-parameters against known ground truth.

``verify_gate.py`` says whether the gate fired; this says whether it was
*right*. That needs a truth signal from outside the clustering, because a
gate scored against the clusters that produced its own references reports
AUC 1.000 no matter how badly it is wrong -- measured, at k=5, while muting
38% of a stem that holds one person start to finish.

Two independent facts about the reference clip supply that truth:

  * face 1 is one speaker for the whole clip. Established by f0 (consistently
    female, ``diag_seams.py``) and by k=3 diarization giving it 100% purity.
    So **any** muting of face 1 is a false positive.
  * face 0 is intruded on at t = 2-4 s and 12-18 s by an off-screen man,
    from joint diarization against the mixture (``verify_identity.py``).

Both are passed as flags, so the script is not welded to one clip.

    python scripts/fit_identity.py runs/862bd92a01ac/spike
    python scripts/fit_identity.py runs/862bd92a01ac/spike --clean 1 \
        --intruder 0:2-4,12-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.identity import (  # noqa: E402
    IdentityConfig, _kmeans, embed, enroll, gate_intruders,
)
from verify_gate import face_stems  # noqa: E402


def parse_spans(spec: str) -> dict[int, list[tuple[float, float]]]:
    """``"0:2-4,12-18"`` -> ``{0: [(2, 4), (12, 18)]}``."""
    out: dict[int, list[tuple[float, float]]] = {}
    for part in filter(None, spec.split(";")):
        face, _, spans = part.partition(":")
        out[int(face)] = [
            (float(a), float(b))
            for a, b in (s.split("-") for s in spans.split(",") if s)
        ]
    return out


def span_mask(spans: list[tuple[float, float]], n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in spans:
        m[int(a * sr) : int(b * sr)] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spike_dir", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clean", default="1",
                    help="comma-separated faces known to hold ONE speaker "
                         "throughout; muting any of them is a false positive")
    ap.add_argument("--intruder", default="0:2-4,12-18",
                    help="known intruder spans, 'face:a-b,c-d;face:...'")
    ap.add_argument("--win", type=float, default=None,
                    help="also sweep this window length instead of the default")
    args = ap.parse_args()

    clean = {int(c) for c in args.clean.split(",") if c.strip() != ""}
    truth = parse_spans(args.intruder)

    faces = face_stems(args.spike_dir)
    stems, sr = [], 16000
    for p in faces:
        x, sr = sf.read(str(p), dtype="float32")
        stems.append(x)
    mix = None
    mp = args.spike_dir / "mixture_ref.wav"
    if mp.exists():
        mix, _ = sf.read(str(mp), dtype="float32")

    base = IdentityConfig()
    cfg = IdentityConfig(win_s=args.win) if args.win else base

    # Embed once. The sweep only re-clusters, so the encoder runs a single
    # time instead of once per k -- the embeddings do not depend on k.
    print(f"embedding {len(stems)} stems + mixture at win={cfg.win_s:g}s "
          f"hop={cfg.hop_s:g}s ...")
    cached = [embed(s, sr, cfg, args.device) for s in stems]
    if mix is not None:
        cached.append(embed(mix, sr, cfg, args.device))
    for i, (t, e) in enumerate(cached):
        who = f"face {i}" if i < len(stems) else "mixture"
        print(f"  {who}: {len(e)} windows")

    X = np.concatenate([e for _, e in cached if len(e)])
    mix_emb = cached[-1][1] if mix is not None else None

    def min_mix_share(k: int) -> float:
        """The selector's decision variable: the share of the MIXTURE held by
        the least-attested cluster. Below min_share that cluster lives in a
        stem but not in the room, so ``_choose_k`` stops growing."""
        if mix_emb is None:
            return float("nan")
        _, C = _kmeans(X, k)
        lab = np.argmax(mix_emb @ C.T, axis=1)
        return float((np.bincount(lab, minlength=k) / len(mix_emb)).min())

    ks = [k for k in range(len(stems), len(stems) + base.max_extra + 1) if k <= len(X)]
    shares = {k: min_mix_share(k) for k in ks}
    chosen = ks[0]
    for k in ks[1:]:
        if shares[k] < base.min_share:
            break
        chosen = k

    print(f"\n{'k':>3} {'minMix':>7} {'purity':>14} {'muted %':>16} "
          f"{'FP(clean)':>10} {'recall':>8} {'prec':>7}")
    print("-" * 74)

    for k in ks:
        plan, per = enroll(stems, sr, cfg, mixture=mix, device=args.device,
                           k=k, cached=cached)

        muted, fp, tp, fn, pos = [], 0.0, 0, 0, 0
        for i, (t, E) in enumerate(per):
            s = plan.score(E, i)
            _, env = gate_intruders(stems[i], t, s, sr, cfg)
            m = env < 0.5
            muted.append(float(m.mean()))
            if i in clean:
                fp = max(fp, float(m.mean()))
            if i in truth:
                g = span_mask(truth[i], len(m), sr)
                tp += int((m & g).sum())
                fn += int((~m & g).sum())
                pos += int(m.sum())

        rec = tp / max(tp + fn, 1)
        prec = tp / max(pos, 1)
        mark = "  <== selected" if chosen == k else ""
        print(f"{k:>3} {shares[k]:>7.3f} "
              f"{' '.join(f'{p:.2f}' for p in plan.purity):>14} "
              f"{' '.join(f'{m*100:5.1f}' for m in muted):>16} "
              f"{fp*100:>9.1f}% {rec*100:>7.1f}% {prec*100:>6.1f}%{mark}")

    print(f"\n  minMix     share of the MIXTURE held by the least-attested "
          f"cluster.\n             Below min_share={base.min_share:g} that "
          f"cluster exists in a stem but\n             not in the room, so it "
          f"is a model artifact, not a person.")
    print("  FP(clean)  muting of a face known to hold one speaker. Target 0.")
    print("  recall     of the known intruder spans that got muted.")
    print("  prec       of what was muted, the share that was truly intruder.")
    print("\n  Read FP first: a gate that mutes clean speech has failed even "
          "at 100% recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
