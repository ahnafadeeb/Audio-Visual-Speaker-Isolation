"""Step-1 spike: prove AV-MossFormer2 TSE extracts the right voice.

The gate for the whole v3 architecture. Runs the vendored model on one real
clip, one call per tracked face, and writes per-face WAVs plus an ROI contact
sheet so the crop can be eyeballed.

No pipeline, no server, no gating -- deliberately. If the raw model output is
not the right speaker, nothing downstream matters. It does chunk, because the
full reference clip needs ~4.3 GB in one call and this card has 4.

    python scripts/spike_avtse.py runs/862bd92a01ac
    python scripts/spike_avtse.py runs/862bd92a01ac --seconds 8 --device cpu
    python scripts/spike_avtse.py runs/862bd92a01ac --chunk-s 1e9   # no chunks
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.avtse import SAMPLE_RATE  # noqa: E402
from app.config import Config  # noqa: E402
from app.media import read_audio  # noqa: E402
from app.roi import MIN_FACE_PX, boxes_to_array, extract_roi, n_video_frames_for  # noqa: E402
from app.separation import AVTSESeparator  # noqa: E402


def contact_sheet(rois: list[np.ndarray], path: Path, n: int = 8) -> None:
    """Write a strip of evenly-spaced ROI frames, one row per face.

    A misaligned crop is invisible in the numbers and obvious in this image,
    which is why it exists.
    """
    import cv2

    from app.roi import ROI_MEAN, ROI_STD

    rows = []
    for roi in rois:
        idx = np.linspace(0, len(roi) - 1, n).astype(int)
        frames = [np.clip(roi[i] * ROI_STD + ROI_MEAN, 0, 1) for i in idx]
        rows.append(np.concatenate(frames, axis=1))
    sheet = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    cv2.imwrite(str(path), sheet)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seconds", type=float, default=None,
                    help="truncate the clip (keeps VRAM down while iterating)")
    ap.add_argument("--chunk-s", type=float, default=None,
                    help="body length; default is AVTSEConfig.resolve_chunk_s "
                         "for this device. Pass a huge number to force one call.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run_dir
    out = args.out or run / "spike"
    out.mkdir(parents=True, exist_ok=True)

    video = run / "video.mp4"
    tracks = json.loads((run / "tracks.json").read_text())

    # ---- audio ----------------------------------------------------------
    # video.mp4 is the pipeline's fps-normalised VIDEO-ONLY stream; the audio
    # still lives in the original upload. Frame timing is shared, so the two
    # stay aligned.
    audio_src = next((p for p in (run / "input.mp4", video) if p.exists()), None)
    if audio_src is None:
        raise SystemExit(f"no audio source in {run}")

    wav = out / "mixture.wav"
    if not wav.exists():
        from app.media import extract_audio
        extract_audio(audio_src, wav, SAMPLE_RATE)
    mixture = read_audio(wav, SAMPLE_RATE)
    if args.seconds:
        mixture = mixture[: int(args.seconds * SAMPLE_RATE)]
    n_frames = n_video_frames_for(len(mixture))
    print(f"mixture : {len(mixture)} samples  {len(mixture)/SAMPLE_RATE:.2f}s"
          f"  -> {n_frames} visual frames")

    # ---- visual ---------------------------------------------------------
    size = (tracks["width"], tracks["height"])
    rois = []
    for tr in tracks["tracks"]:
        boxes = boxes_to_array(tr["boxes"])
        rt = extract_roi(str(video), boxes, n_frames, size)
        rois.append(rt)
        flag = "  << UNDERSIZED" if rt.undersized else ""
        drop = int(rt.filled.sum())
        print(f"track {tr['id']} ({tr['label']:>9}) : roi{tuple(rt.roi.shape)}"
              f"  face {rt.median_face_px:.0f}px (floor {MIN_FACE_PX})"
              f"  dropped {drop}{flag}")

    contact_sheet([r.roi for r in rois], out / "roi_contact_sheet.png")
    print(f"wrote   : {out/'roi_contact_sheet.png'}  <- EYEBALL THIS")

    # ---- extract --------------------------------------------------------
    dev = torch.device(args.device)
    chunk_s = args.chunk_s if args.chunk_s is not None else \
        Config().avtse.resolve_chunk_s(dev.type)
    sep = AVTSESeparator(str(dev), chunk_s=chunk_s)
    sep.model  # load now, so the timings below are inference, not download
    print(f"chunk   : {chunk_s:.1f}s bodies + {sep.context_s:.1f}s discarded "
          f"context each side, {sep.fade_s*1000:.0f}ms crossfade")

    stems = []
    for i, rt in enumerate(rois):
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        est = sep.separate_one(mixture, rt.roi, SAMPLE_RATE)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        stems.append(est)

        peak = f" peak {torch.cuda.max_memory_allocated()/2**20:.0f}MiB" if dev.type == "cuda" else ""
        print(f"face {i}  -> {dt:5.2f}s  RTF {dt/(len(mixture)/SAMPLE_RATE):.2f}{peak}")
        sf.write(str(out / f"face{i}.wav"), est, SAMPLE_RATE, subtype="FLOAT")

    # ---- cheap sanity numbers ------------------------------------------
    print()
    S = np.stack(stems)
    for i, s in enumerate(S):
        print(f"face {i}: rms {20*np.log10(np.sqrt((s**2).mean())+1e-12):7.2f} dB"
              f"   peak {np.abs(s).max():.4f}")
    if len(S) == 2:
        a, b = S[0] - S[0].mean(), S[1] - S[1].mean()
        r = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"\ninter-stem correlation: {r:+.4f}"
              "   (near 0 = two genuinely different voices;"
              " near 1 = the model returned the same speaker twice)")

    sf.write(str(out / "mixture_ref.wav"), mixture, SAMPLE_RATE, subtype="FLOAT")
    print(f"\nwrote {out}/face*.wav  -- LISTEN: right speaker? room preserved?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
