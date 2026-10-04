"""How many distinct speakers are actually inside these stems?

The pitch gate in ``verify_avtse.py`` asks "is this stem male or female", which
assumes the stem holds exactly one person. It does not. AV-TSE is an extractor,
not a detector: when the conditioned face is not speaking there is no target to
extract, so the model returns the most speaker-like thing left in the mixture.
If someone off-screen is talking, that is who comes out -- and no pitch
threshold can tell "the same man, animated" from "a different man".

Speaker embeddings can. This pools short windows from every stem, embeds each
with ECAPA, clusters them, and prints which cluster owns which seconds. A stem
that holds one person is one cluster; a stem that changes hands partway through
is two, and the timeline shows exactly when.

    python scripts/verify_identity.py runs/862bd92a01ac/spike
    python scripts/verify_identity.py runs/862bd92a01ac/spike --k 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Windows quieter than (this stem's loud level - GATE_DB) are skipped. An
#: embedding of near-silence is an embedding of the room, not of a person, and
#: pooling those would invent a spurious "speaker" made of noise.
GATE_DB = 20.0


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt((x**2).mean()) + 1e-12))


def windows(x: np.ndarray, sr: int, win_s: float, hop_s: float, gate_db: float = GATE_DB):
    """Yield ``(t, samples)`` for windows loud enough to hold a voice."""
    n, h = int(win_s * sr), int(hop_s * sr)
    segs = [(s / sr, x[s : s + n]) for s in range(0, max(len(x) - n, 1), h)]
    if not segs:
        return []
    loud = np.percentile([rms_db(s) for _, s in segs], 90)
    return [(t, s) for t, s in segs if rms_db(s) > loud - gate_db]


def kmeans(X: np.ndarray, k: int, iters: int = 60, seed: int = 0):
    """Cosine k-means on unit vectors (so the mean is the centroid direction)."""
    rng = np.random.default_rng(seed)
    # k-means++ init: spread the seeds, otherwise a rare third speaker with few
    # windows gets swallowed by whichever cluster started nearest it.
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = 1.0 - (X @ np.stack(C).T).max(axis=1)
        C.append(X[np.argmax(d)])
    C = np.stack(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = np.argmax(X @ C.T, axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c = X[m].mean(axis=0)
                C[j] = c / (np.linalg.norm(c) + 1e-12)
    return lab, C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spike_dir", type=Path)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=None,
                    help="number of speakers to cluster into; default tries 2..4")
    ap.add_argument("--gate-db", type=float, default=GATE_DB,
                    help="skip windows this far below the signal's loud level. "
                         "Too permissive and near-silent residue forms its own "
                         "spurious 'speaker'.")
    ap.add_argument("--mixture", action="store_true",
                    help="also diarize mixture_ref.wav. This is the ground "
                         "truth for how many people are audible at all, "
                         "including any who are never on screen.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    faces = sorted(args.spike_dir.glob("face*.wav"))
    if not faces:
        print(f"no face*.wav in {args.spike_dir}")
        return 2
    names = [p.stem for p in faces]
    if args.mixture:
        mix = args.spike_dir / "mixture_ref.wav"
        if mix.exists():
            faces.append(mix)
            names.append("MIXTURE")

    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "hf"))
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path(os.environ["HF_HOME"]) / "ecapa"),
        run_opts={"device": args.device},
        local_strategy=LocalStrategy.COPY,
    )

    embs, owner, times = [], [], []
    for i, p in enumerate(faces):
        x, sr = sf.read(str(p), dtype="float32")
        ws = windows(x, sr, args.win, args.hop, args.gate_db)
        print(f"{names[i]:>8}: {len(ws)} active windows of {args.win:g}s")
        for t, s in ws:
            with torch.no_grad():
                e = enc.encode_batch(torch.from_numpy(s).unsqueeze(0)).squeeze().cpu().numpy()
            embs.append(e / (np.linalg.norm(e) + 1e-12))
            owner.append(i)
            times.append(t)
    X = np.stack(embs)
    owner = np.asarray(owner)
    times = np.asarray(times)

    ks = [args.k] if args.k else [2, 3, 4]
    for k in ks:
        lab, C = kmeans(X, k)
        sim = C @ C.T
        # Mean cosine of each point to its own centroid: how tight the clusters
        # are. Low values mean the k we chose is splitting one person in half.
        tight = float(np.mean([X[j] @ C[lab[j]] for j in range(len(X))]))
        print(f"\n===== k = {k}   (mean within-cluster cosine {tight:.3f}) =====")
        print("  centroid cosine matrix:")
        for a in range(k):
            print("   " + " ".join(f"{sim[a, b]:+.3f}" for b in range(k)))
        for i in range(len(faces)):
            m = owner == i
            print(f"\n  {names[i]} timeline (cluster id per {args.hop:g}s window):")
            line, cur = [], None
            for t, c in zip(times[m], lab[m]):
                if c != cur:
                    line.append(f"  [{t:5.1f}s] -> cluster {c}")
                    cur = c
            print("\n".join(line) if line else "    (no active windows)")
            counts = np.bincount(lab[m], minlength=k)
            print("    share: " + "  ".join(
                f"c{j} {counts[j] / max(counts.sum(), 1) * 100:.0f}%" for j in range(k)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
