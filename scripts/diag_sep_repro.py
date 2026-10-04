"""Is SepFormer's stem ORDER reproducible run to run on this machine?

    PYTHONPATH=. .venv\\Scripts\\python.exe scripts\\diag_sep_repro.py runs/pair_v2

Companion to ``diag_repro.py``, which holds the stems fixed and varies vision.
This one does the opposite: it takes vision out entirely and asks whether the
*separator* returns its two sources in a stable order.

Why it is worth a GPU minute.  The stored run (``runs/862bd92a01ac``) shipped the
pairing ``[1, 0]``, with stem 0 carrying the man's voice.  The fresh run
(``runs/pair_v2``) shipped ``[0, 1]``, with stem **1** carrying the man's voice.
Same clip, same code path, and inverted relative to each other.  There are two
explanations and they lead to opposite engineering conclusions:

  A. **The output order is not stable.**  Then "which pairing is correct" is a
     property of the RUN, not of the clip.  No amount of per-clip matcher
     calibration can fix that, a demo must never hard-code a known-good pairing,
     and the user-facing swap control is the only honest interface.

  B. **The order is deterministic** and something between the two runs moved it.
     Then the disagreement is a regression with a findable cause, and blaming
     nondeterminism would be an excuse for not looking.

So: separate the same mixture twice in one process and hash the stems.  The
mixture goes through ``media.extract_audio`` + ``media.read_audio``, the
pipeline's own two functions, rather than a hand-rolled ffmpeg call -- comparing
against a differently resampled copy would confound the answer with a decode
difference.  The decode is hashed too, and checked against the run's own
``_probe.wav`` when one is present, so "the input was identical" is measured
rather than assumed.

Note ``video.mp4`` is NOT a valid source here: ``normalize_video`` strips the
audio with ``-an`` so the browser can never play the mixture.  The samples the
separator saw came from ``input.mp4``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
import soundfile as sf                                          # noqa: E402

from app import media                                           # noqa: E402
from app.config import CACHE_DIR, CONFIG                        # noqa: E402
from app.separation import build_separator                      # noqa: E402
from _job import pipeline_audio                                 # noqa: E402


def digest(a: np.ndarray) -> str:
    """Bit-exact digest.  Rounding to print precision is how jitter hides."""
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", nargs="?", default="runs/pair_v2")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    job = Path(args.job_dir)
    sr = CONFIG.audio.sample_rate

    print("=" * 78)
    print(f"SepFormer stem-order reproducibility -- {job.name}")
    print("=" * 78)

    src = pipeline_audio(job)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mix.wav"
        media.extract_audio(src, wav, sr)
        mix = media.read_audio(wav, sr)

    print(f"  mixture  {src}  ->  {len(mix) / sr:.2f} s, {len(mix)} samples")
    print(f"           hash {digest(mix)}")
    probe = job / "_probe.wav"
    if probe.exists():
        ref = media.read_audio(probe, sr)
        same = ref.shape == mix.shape and digest(ref) == digest(mix)
        print(f"  vs _probe.wav in the run dir: "
              f"{'IDENTICAL' if same else 'DIFFERENT'} "
              f"({digest(ref)}, {len(ref)} samples)")
        if not same:
            print("           => the decode itself is not stable, so any stem")
            print("              difference below is confounded.  Fix this first.")

    device = CONFIG.runtime.resolve_device()
    # Read from CONFIG, not retyped: a drifted default would make the answer
    # describe a configuration the pipeline never runs.
    chunk_s = CONFIG.audio.resolve_chunk_s(device)
    overlap_s = CONFIG.audio.overlap_s
    print(f"  device {device}   chunk_s {chunk_s:.2f}   overlap_s {overlap_s:.2f}   "
          f"separator {CONFIG.runtime.separator}")

    passes: list[np.ndarray] = []
    for k in range(args.passes):
        # A fresh separator each pass, matching what two invocations of the
        # pipeline do.  Reusing one instance would only test the forward pass
        # and would miss any order-affecting state set up at load time.
        sep = build_separator(CONFIG.runtime.separator, device=device,
                              cache_dir=str(CACHE_DIR / "models"),
                              chunk_s=chunk_s, overlap_s=overlap_s)
        stems = np.asarray(sep.separate(mix, sr), dtype=np.float32)
        if stems.ndim != 2:
            raise SystemExit(f"unexpected stem shape {stems.shape}")
        if stems.shape[0] > stems.shape[1]:          # (samples, stems) -> (stems, samples)
            stems = stems.T
        passes.append(stems)
        print(f"\n  pass {k}: shape {stems.shape}")
        for i, s in enumerate(stems):
            rms = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
            print(f"    stem {i}  hash {digest(s)}  rms {rms:.6f}")
        del sep

    print("\n" + "=" * 78)
    ref = passes[0]
    if any(p.shape != ref.shape for p in passes):
        print("VERDICT: shapes differ across passes -- not comparable.")
        return 1

    n = ref.shape[0]
    bit_same = all(digest(p[i]) == digest(ref[i])
                   for p in passes[1:] for i in range(n))

    # Every stem of pass 0 against every stem of pass k, so a SWAP is
    # distinguishable from a numerical wobble.  A swap shows as an off-diagonal
    # near 1; jitter leaves the diagonal near 1 and the hashes different.
    def corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros((n, n))
        for i in range(n):
            u = a[i].astype(np.float64); u -= u.mean()
            for j in range(n):
                v = b[j].astype(np.float64); v -= v.mean()
                den = float(np.sqrt((u @ u) * (v @ v)))
                out[i, j] = float(u @ v / den) if den > 0 else 0.0
        return out

    order_same = True
    for k, p in enumerate(passes[1:], start=1):
        C = corr_matrix(ref, p)
        best = [int(np.argmax(np.abs(C[i]))) for i in range(n)]
        order_same &= best == list(range(n))
        print(f"  pass 0 (rows) vs pass {k} (cols) correlation, best match {best}:")
        for i in range(n):
            print("    " + "  ".join(f"{C[i, j]:+.6f}" for j in range(n)))
        print(f"    max |diff| {float(np.max(np.abs(ref - p))):.3e}")

    print()
    if bit_same:
        print("VERDICT: BIT-IDENTICAL across passes in one process.")
        print("  Stem order is deterministic on this machine, so the stored run and")
        print("  this one disagreeing is NOT run-to-run noise: either an intervening")
        print("  code change moved it, or the two runs did not feed the separator")
        print("  the same samples.  Worth naming, not worth excusing.")
        print("  Note this measures ONE process.  Order could still differ across")
        print("  processes (library init, cuDNN autotune) -- run this twice to tell.")
    elif order_same:
        print("VERDICT: same ORDER, not bit-identical.")
        print("  Numerical nondeterminism only.  The pairing is stable within a")
        print("  process; the stored-run disagreement needs another explanation.")
    else:
        print("VERDICT: the ORDER CHANGED between passes.")
        print("  The correct stem/face assignment is a property of the RUN, not the")
        print("  clip.  No per-clip matcher calibration can fix it, nothing may")
        print("  hard-code a known-good pairing, and the swap control is the only")
        print("  correct interface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
