"""Fidelity: does the extracted voice still sound like the person?

``ghost_test.py`` certifies the silence. This asks the opposite half of the
requirement -- that the audio which *does* survive is the original voice and
not a resynthesis of it.

    python scripts/fidelity.py runs/reg_v3 --mixture runs/0e3dee84dbaa/input.mp4
    python scripts/fidelity.py runs/reg_v3 --mixture ... --write   # dump excerpts

What can honestly be used as a reference
----------------------------------------
There is no ground-truth isolated stem for a real recording, so SI-SDR in its
usual sense -- against a known clean source -- is not available and any number
reported that way would be invented. What *is* available is a region argument:
where one person dominates the room, the mixture already **is** that person's
signal, near enough to serve as the reference. So the comparison is

    stems_raw[i]   vs   mixture,   restricted to face i's dominant frames

and every metric below inherits that restriction. ``stems_raw`` rather than
``stems_demo`` because gating is a separate question that ``ghost_test.py``
already answers; a gate zeroing a pause would show up here as damage.

How the dominant frames are found, and why not the obvious way
--------------------------------------------------------------
The first version of this script asked for *solo* frames: target above its own
speech threshold, every rival stem below a low absolute floor, mixture active,
all three holding across a full second. On the reference clip that found **no
regions at all**, in a way worth recording because it was not a coding error:

  * Per-frame conjunctions do not survive speech. 11.1 s of frames satisfied
    all three conditions individually and not one contiguous second did --
    natural speech dips below any fixed threshold at every stop consonant.
  * "Rival is quiet" is the wrong question to put to ``stems_raw``. AV-TSE is
    an extractor, not a detector: when the conditioned face is silent it
    returns the most speaker-like residue it can find, at a comparable level.
    Measured on the reference clip the raw channels sit within 0-6 dB of each
    other over most 1 s windows. That is the documented behaviour the identity
    gate exists to clean up, so a test that waits for a raw channel to go
    quiet waits forever.

What replaces it is a *ratio between the stems*, per frame::

    SIR(t) = 10 log10( P_target(t) / max_rival P_rival(t) )  >=  SOLO_SIR_DB

which is scale-free, needs no per-stem normalisation, and -- the reason to
prefer it -- **bounds the metric it feeds**. At 20 dB every rival contributes
at most 1% of the reference's energy, so SI-SDR cannot read above about 20 dB
however good the separation is. The ceiling is reported next to the number.

Is that circular? The frames are chosen from the stems, which are model output.
It is worth being precise about which way the circularity cuts: if the model
had failed to separate, no frame would show 20 dB of dominance and the channel
would be *skipped*, so the failure mode is finding nothing to score, not
passing something that should have failed. The mixture-active check closes the
other direction -- a stretch the extractor invented out of silence cannot
qualify, because the room was quiet there.

The one real caveat is background: where the mixture carries music or room
noise, removing it is correct behaviour that this test scores as deviation.
That makes every number here a LOWER bound on fidelity. It cannot flatter.

The four questions, and which number answers each
-------------------------------------------------
``lag``      cross-correlation peak offset in samples, measured on the **full**
             signals rather than the selection, because a time shift is a
             global property of the path and the full-length correlation is the
             most robust way to see it. Non-zero means the output is shifted
             against the input, which desynchronises it from the video
             regardless of how good it sounds.
``SI-SDR``   overall waveform agreement, scale-invariant by construction, so a
             loudness difference cannot inflate or deflate it.
``F0 ratio`` the direct test for problem 2, pitch shifting. Median voiced F0 of
             the stem over median voiced F0 of the mixture across the same
             frames. **1.000 is the pass**; 1.06 would be a semitone sharp,
             which is what a sample-rate mismatch produces and what the v1
             build was reported to do. PESQ would only note this obliquely, so
             it is measured directly rather than inferred.
``PESQ``     perceptual quality, ITU-T P.862.2 wideband. Roughly: >3.5 is hard
             to tell from the source, <2.5 is audibly processed.
``STOI``     intelligibility. Included because a voice can score acceptably on
             PESQ while being harder to understand.

The last two are *best effort* and often skipped. Both carry internal frame
alignment and neither is meaningful on spliced fragments, so they cannot use
the frame selection the other three use -- they need a continuous stretch, and
a conversational clip with overlapping speakers may simply not contain one.
Measured on two two-speaker clips, no channel offered a contiguous second of
20 dB dominance. That is a property of conversation, not a defect, and it is
reported as a skip rather than folded into the verdict.

SI-SDR and F0 are the ones that fail loudly if the signal path grows a resample
or a frame-offset bug, which is exactly the class of defect that produced the
original complaint, and both are available on any channel the separator
actually separated.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import media  # noqa: E402
from app.config import CONFIG  # noqa: E402
from scripts._job import pipeline_audio  # noqa: E402


def load_mixture(src: Path, sample_rate: int) -> np.ndarray:
    """Mono mixture from a media file, whether or not it is already a WAV.

    ``media.read_audio`` is a soundfile call and will not open an mp4, so a
    container has to go through ffmpeg first. Doing it here keeps the caller
    from having to know which of the two it was handed.
    """
    if src.suffix.lower() in {".wav", ".flac", ".ogg"}:
        return media.read_audio(src, sample_rate)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mix.wav"
        media.extract_audio(src, wav, sample_rate)
        return media.read_audio(wav, sample_rate)


#: A frame this far above the signal's own 95th percentile counts as speech.
#: Matches the gate's ``open_db`` so "active" means the same thing here as it
#: does everywhere else in this codebase.
ACTIVE_DB = -20.0

#: Frame-level dominance required before the mixture may stand in for the
#: target's clean signal. 20 dB means every rival contributes at most 1% of the
#: energy in the reference, which caps SI-SDR at about 20 dB -- a selection
#: threshold that bounds the metric it feeds has to be read together with it,
#: so the implied ceiling is printed beside the measurement.
SOLO_SIR_DB = 20.0

#: Shortest contiguous stretch a selected frame must belong to. F0 uses a 40 ms
#: analysis window, which must not straddle a splice between two selections.
MIN_RUN_MS = 100.0

#: Contiguous material PESQ and STOI need before they are attempted.
MIN_CONTIG_S = 0.5

#: Analysis frame for the level and dominance decisions.
FRAME_MS = 10.0

FINDINGS: list[str] = []
MEASURED = 0
SKIPPED = 0


def report(name: str, ok: bool | None, detail: str, note: str = "") -> None:
    global MEASURED, SKIPPED
    tag = "skip" if ok is None else ("ok  " if ok else "FAIL")
    print(f"  {tag}  {name:<30} {detail}")
    if note:
        print(f"          {note}")
    if ok is None:
        SKIPPED += 1
    else:
        MEASURED += 1
    if ok is False:
        FINDINGS.append(f"{name}: {detail}")


def frame_power(x: np.ndarray, sr: int, frame_ms: float = FRAME_MS):
    """Mean square per frame, plus the frame length in samples."""
    n = max(1, int(round(sr * frame_ms / 1000.0)))
    nf = int(np.ceil(x.size / n))
    p = np.pad(x, (0, nf * n - x.size))
    return (p.reshape(nf, n) ** 2).mean(axis=1), n


def frame_db(x: np.ndarray, sr: int, frame_ms: float = FRAME_MS) -> tuple[np.ndarray, int]:
    """Per-frame level in dB relative to the signal's own 95th percentile."""
    pw, n = frame_power(x, sr, frame_ms)
    ldb = 10.0 * np.log10(pw + 1e-12)
    return ldb - float(np.percentile(ldb, 95.0)), n


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def dominant_runs(stems: np.ndarray, mixture: np.ndarray, sr: int, ch: int,
                  sir_db: float, min_run_ms: float):
    """Frame runs where face ``ch`` dominates the room.

    Returns ``(runs, sir, frame_len)`` with ``runs`` as ``(start, end)`` sample
    ranges, each at least ``min_run_ms`` long, and ``sir`` the per-frame
    dominance in dB so the caller can report the ceiling it bought.
    """
    n = stems.shape[1]
    nch = stems.shape[0]
    pw = [frame_power(stems[i], sr)[0] for i in range(nch)]
    fr = frame_power(stems[ch], sr)[1]
    lvl = frame_db(stems[ch], sr)[0]
    mix_db = frame_db(mixture[:n], sr)[0]
    nf = min(len(mix_db), len(lvl), *(len(p) for p in pw))

    if nch > 1:
        rival = np.max([pw[i][:nf] for i in range(nch) if i != ch], axis=0)
    else:                       # single-channel export: nothing to be drowned by
        rival = np.zeros(nf)
    sir = 10.0 * np.log10((pw[ch][:nf] + 1e-20) / (rival + 1e-20))

    keep = (lvl[:nf] >= ACTIVE_DB) & (mix_db[:nf] >= ACTIVE_DB) & (sir >= sir_db)
    min_frames = max(1, int(round(min_run_ms / FRAME_MS)))
    runs = [(s * fr, min(e * fr, n)) for s, e in runs_of(keep) if e - s >= min_frames]
    return runs, sir[keep], fr


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Projects ``est`` onto ``ref`` first, so a
    pure gain difference scores as perfect rather than as distortion."""
    ref = ref - ref.mean()
    est = est - est.mean()
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    proj = a * ref
    noise = est - proj
    return 10.0 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))


def f0_track(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0,
             frame_ms: float = 40.0, hop_ms: float = 10.0):
    """Autocorrelation F0 per frame. Returns ``(f0, voiced)``.

    Deliberately plain: this is a *comparison* of two tracks measured the same
    way, so a sophisticated estimator would buy accuracy the comparison does
    not need. Any bias cancels when the ratio is taken.
    """
    n, h = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0, voiced = [], []
    for s in range(0, max(len(x) - n, 1), h):
        w = x[s:s + n].astype(np.float64)
        if w.size < n or not np.any(w):
            f0.append(0.0); voiced.append(False); continue
        w = w - w.mean()
        e = np.dot(w, w)
        if e < 1e-10:
            f0.append(0.0); voiced.append(False); continue
        ac = np.correlate(w, w, mode="full")[n - 1:]
        seg = ac[lo:hi]
        if seg.size == 0:
            f0.append(0.0); voiced.append(False); continue
        k = int(np.argmax(seg)) + lo
        f0.append(sr / k)
        voiced.append(ac[k] / e > 0.35)
    return np.array(f0), np.array(voiced)


def best_lag(ref: np.ndarray, est: np.ndarray, max_lag: int) -> int:
    """Offset in samples that best aligns ``est`` to ``ref``."""
    a = ref - ref.mean()
    b = est - est.mean()
    c = np.correlate(a, b, mode="full")
    mid = len(b) - 1
    lo, hi = max(0, mid - max_lag), min(len(c), mid + max_lag + 1)
    return int(np.argmax(c[lo:hi]) + lo - mid)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--mixture", type=Path, default=None,
                    help="media file supplying the reference mixture. Defaults "
                         "to the job's own input.mp4; needed when the job came "
                         "from the CLI, which does not copy the upload in.")
    ap.add_argument("--sir", type=float, default=SOLO_SIR_DB,
                    help="dominance a frame needs, dB. Lowering this admits "
                         "more material and lowers the SI-SDR ceiling with it")
    ap.add_argument("--write", action="store_true",
                    help="write solo_chN_{ref,est}.wav for listening")
    args = ap.parse_args()

    raw_p = args.run_dir / "stems_raw.wav"
    if not raw_p.exists():
        print(f"need stems_raw.wav in {args.run_dir}")
        return 2
    # NOT video.mp4, even though it is the file sitting in the job directory:
    # ``normalize_video`` strips the audio with ``-an``, so it carries no
    # mixture at all and ffmpeg exits with "does not contain any stream".
    # ``pipeline_audio`` knows that, and says so loudly when it has to reach
    # outside the job for the upload -- the normal case here, since every
    # CLI-produced run has only video.mp4.
    src = args.mixture or pipeline_audio(args.run_dir)
    if not src.exists():
        print(f"mixture source does not exist: {src}")
        return 2

    raw, sr = sf.read(str(raw_p), dtype="float32", always_2d=True)
    raw = raw.T
    mixture = load_mixture(src, CONFIG.audio.sample_rate)
    meta = {}
    if (args.run_dir / "meta.json").exists():
        meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
    if sr != CONFIG.audio.sample_rate:
        print(f"!! stems are {sr} Hz but the mixture was read at "
              f"{CONFIG.audio.sample_rate} Hz; the F0 ratio below would report "
              f"that mismatch rather than the separator's behaviour")

    # stems_raw was written through a shared peak gain; undo it so the stem and
    # the mixture sit at their original relative levels. SI-SDR would not care,
    # but PESQ and STOI are not scale-invariant.
    g = float(meta.get("silence", {}).get("export_gain") or 1.0)
    if g:
        raw = raw / g

    n = min(raw.shape[1], mixture.size)
    raw, mixture = raw[:, :n], mixture[:n]

    print("=" * 74)
    print(f"FIDELITY  {args.run_dir}")
    print("=" * 74)
    print(f"separator {meta.get('separator', '?')}   {raw.shape[0]} channels   "
          f"{n / sr:.1f} s @ {sr} Hz   export_gain {g:.4f}")
    print(f"mixture   {src}")
    print(f"reference = the mixture itself, on frames where this face leads the "
          f"room by {args.sir:.0f} dB")

    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        pesq_fn = None
    try:
        from pystoi import stoi as stoi_fn
    except ImportError:
        stoi_fn = None

    for ch in range(raw.shape[0]):
        runs, sirs, _ = dominant_runs(raw, mixture, sr, ch, args.sir, MIN_RUN_MS)
        total = sum(b - a for a, b in runs) / sr
        print(f"\n[{ch}] channel {ch}   {len(runs)} run(s) of >= {MIN_RUN_MS:.0f} ms, "
              f"{total:.2f} s scorable")

        if not runs:
            report("has scorable material", None,
                   f"no frame reaches {args.sir:.0f} dB dominance",
                   "either the speakers overlap throughout or this channel was "
                   "never separated; fidelity cannot be measured either way")
            continue

        ref = np.concatenate([mixture[a:b] for a, b in runs]).astype(np.float64)
        est = np.concatenate([raw[ch][a:b] for a, b in runs]).astype(np.float64)
        ceiling = float(np.mean(sirs)) if sirs.size else args.sir
        print(f"          mean dominance {ceiling:.1f} dB -- SI-SDR cannot read "
              f"above roughly that")

        # Measured on the FULL signals, not the selection: a time shift is a
        # property of the path, and splicing the reference would invent
        # correlation structure at every seam.
        lag = best_lag(mixture, raw[ch], max_lag=int(0.050 * sr))
        report("output is time-aligned", abs(lag) <= 1,
               f"lag {lag:+d} samples ({lag / sr * 1000:+.2f} ms)",
               "" if abs(lag) <= 1 else "audio will drift against the video")

        v = si_sdr(ref, est)
        report("SI-SDR vs mixture", v >= 5.0, f"{v:6.2f} dB  (ceiling ~{ceiling:.0f})",
               "" if v >= 5.0 else "the stem departs substantially from the source")

        # -- the pitch question, asked directly ----------------------------- #
        # Per run, never across a splice: the 40 ms analysis window would
        # straddle the seam and read a pitch neither side has.
        fr_all, fe_all = [], []
        for a, b in runs:
            f_ref, v_ref = f0_track(mixture[a:b].astype(np.float64), sr)
            f_est, v_est = f0_track(raw[ch][a:b].astype(np.float64), sr)
            k = min(len(f_ref), len(f_est))
            m = v_ref[:k] & v_est[:k]
            fr_all.append(f_ref[:k][m])
            fe_all.append(f_est[:k][m])
        f_ref, f_est = np.concatenate(fr_all), np.concatenate(fe_all)
        if f_ref.size >= 10:
            ratio = float(np.median(f_est) / np.median(f_ref))
            cents = 1200.0 * np.log2(ratio)
            ok = abs(cents) <= 10.0        # 10 cents is below the audible JND
            report("F0 unchanged (no pitch shift)", ok,
                   f"ratio {ratio:.4f}  ({cents:+.1f} cents, "
                   f"{f_ref.size} voiced frames)",
                   "" if ok else "the voice is being shifted -- check for a "
                                 "sample-rate mismatch in the stem path")
        else:
            report("F0 unchanged (no pitch shift)", None,
                   f"only {f_ref.size} commonly-voiced frames")

        # -- best effort: both need continuous material --------------------- #
        contig = [(a, b) for a, b in runs if (b - a) / sr >= MIN_CONTIG_S]
        if not contig:
            longest = max((b - a) / sr for a, b in runs)
            note = (f"longest continuous stretch is {longest * 1000:.0f} ms, "
                    f"under the {MIN_CONTIG_S * 1000:.0f} ms both need")
            report("PESQ (wb)", None, "no continuous stretch", note)
            report("STOI", None, "no continuous stretch")
        else:
            if pesq_fn is not None:
                scores, weights = [], []
                for a, b in contig:
                    try:
                        scores.append(pesq_fn(sr, mixture[a:b].astype(np.float32),
                                              raw[ch][a:b].astype(np.float32), "wb"))
                        weights.append(b - a)
                    except Exception:
                        continue      # degenerate segment; PESQ declines
                if scores:
                    p = float(np.average(scores, weights=weights))
                    report("PESQ (wb)", p >= 2.5,
                           f"{p:5.2f}  of 4.64  ({len(scores)} segment(s))",
                           "" if p >= 2.5 else "audibly processed")
                else:
                    report("PESQ (wb)", None, "every segment was declined")
            else:
                report("PESQ (wb)", None, "pesq not installed")

            if stoi_fn is not None:
                r = np.concatenate([mixture[a:b] for a, b in contig]).astype(np.float64)
                e = np.concatenate([raw[ch][a:b] for a, b in contig]).astype(np.float64)
                s = float(stoi_fn(r, e, sr, extended=False))
                report("STOI", s >= 0.85, f"{s:5.3f}  of 1.000",
                       "" if s >= 0.85 else "intelligibility is degraded")
            else:
                report("STOI", None, "pystoi not installed")

        if args.write:
            sf.write(str(args.run_dir / f"solo_ch{ch}_ref.wav"), ref, sr, subtype="FLOAT")
            sf.write(str(args.run_dir / f"solo_ch{ch}_est.wav"), est, sr, subtype="FLOAT")
            print(f"          wrote solo_ch{ch}_{{ref,est}}.wav ({total:.1f}s each)")

    print("\n" + "=" * 74)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for f in FINDINGS:
            print(f"  - {f}")
        rc = 1
    elif MEASURED:
        print(f"no findings -- {MEASURED} check(s) passed, {SKIPPED} not applicable")
        rc = 0
    else:
        # Distinct from a pass, and deliberately so: the first version of this
        # script printed "the surviving audio matches the source" after
        # measuring nothing at all, which is the one output a verification tool
        # must never produce.
        print("NOTHING MEASURED -- no channel offered scorable material, so this "
              "run is neither a pass nor a failure")
        rc = 3
    print("=" * 74)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
