"""Media I/O: ffmpeg wrangling and multichannel WAV writing.

Two things here matter more than they look:

**The ``join`` filter, not ``amerge``/``amix``.**  ``join`` maps input *i* to
output channel *i*.  ``amerge`` and ``amix`` *sum* their inputs -- using either
would mix the speakers back together, silently undoing the entire pipeline.

**PCM, never a lossy codec.**  The whole point of the DSP chain is exact
time-domain zeros.  AAC/Opus add coding noise and pre-echo precisely in silent
regions, which resurrects the ghost whisper in the last mile.  Stems ship as
16-bit PCM WAV.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

_FFMPEG: str | None = None
_FFPROBE: str | None = None

#: Long-side cap for the normalised video. See :func:`normalize_video`.
MAX_VIDEO_SIDE = 1920


def ffmpeg_exe() -> str:
    """Locate ffmpeg: PATH first, then the imageio-ffmpeg bundled binary."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG = found
        return found
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Install it with `winget install Gyan.FFmpeg` "
            "(then open a NEW terminal so PATH refreshes), or "
            "`pip install imageio-ffmpeg` to use a bundled binary."
        ) from exc


def ffprobe_exe() -> str | None:
    global _FFPROBE
    if _FFPROBE:
        return _FFPROBE
    _FFPROBE = shutil.which("ffprobe")
    return _FFPROBE


def _run(cmd: list[str]) -> None:
    log.debug("run: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed ({p.returncode}):\n{tail}")


def probe_duration(path: Path) -> float:
    probe = ffprobe_exe()
    if not probe:
        return 0.0
    p = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode != 0:
        return 0.0
    try:
        return float(json.loads(p.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def extract_audio(src: Path, dst: Path, sample_rate: int) -> None:
    """Demux to mono PCM at the model's sample rate."""
    _run([ffmpeg_exe(), "-y", "-i", str(src), "-vn",
          "-ac", "1", "-ar", str(sample_rate),
          "-c:a", "pcm_s16le", str(dst)])


def normalize_video(src: Path, dst: Path, fps: float) -> None:
    """Re-encode video at a fixed fps with the audio stripped.

    Fixed fps keeps the bbox array index-addressable by
    ``round(mediaTime * fps)`` in the browser, and keeps the AV-TSE upgrade
    path viable (ClearVoice hardcodes 25 fps in three places).
    ``-an`` matters: the browser must never be able to play the original mixed
    audio by accident.

    The long side is capped at ``MAX_VIDEO_SIDE``. A phone records 4K, and
    every later stage decodes this file -- face tracking once, mouth cropping
    once per face -- so a 4K copy made a 20 s clip spend ~45 s just decoding,
    and made the browser play 4K H.264 for an overlay. The mouth ROI is
    resized to 224 px anyway: a 600 px face at 4K is a 300 px face at 1080p,
    still twice ``roi.MIN_FACE_PX``. Smaller sources pass through unscaled.
    """
    side = MAX_VIDEO_SIDE
    scale = (f"scale=w='if(gte(iw,ih),min({side},iw),-2)'"
             f":h='if(gte(iw,ih),-2,min({side},ih))'")
    _run([ffmpeg_exe(), "-y", "-i", str(src),
          "-an", "-r", f"{fps:g}", "-vf", scale,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])


def peak_gain(*stem_sets: np.ndarray) -> float:
    """The single gain that keeps every one of ``stem_sets`` under full scale.

    Call this once and pass the result to every :func:`write_multichannel_wav`
    for a given job.  Letting each file compute its own gain gives the demo and
    raw exports *different* scale factors, which breaks two things:

    * the demo and raw stems stop being directly A/B comparable, since one has
      been quietly turned up relative to the other;
    * PESQ is not scale-invariant, so a raw export normalised on its own peak
      scores differently from the same audio normalised with the demo.  (SI-SDR
      is scale-invariant and does not care.)

    Returns 1.0 when nothing clips, so the common case is bit-exact untouched.
    """
    peak = 0.0
    for stems in stem_sets:
        data = np.asarray(stems, dtype=np.float32)
        if data.size:
            peak = max(peak, float(np.abs(data).max()))
    return 0.999 / peak if peak > 0.999 else 1.0


def write_multichannel_wav(stems: np.ndarray, path: Path, sample_rate: int,
                           gain: float | None = None) -> None:
    """Write ``(n_sources, n_samples)`` as one interleaved PCM WAV.

    One file -> one ``AudioBufferSourceNode`` in the browser -> the stems
    physically cannot drift relative to each other.  Channel *i* is speaker *i*.

    ``gain`` is one scalar applied to every channel -- never a per-source
    normalisation, which would change the relative levels of the speakers and
    undo the separation's balance.  Pass the shared :func:`peak_gain` when a job
    writes more than one file; ``None`` falls back to normalising this file
    alone.

    A positive scalar cannot turn an exact zero into a nonzero one, and PCM_16
    maps 0.0 to sample 0, so the hard-mute guarantee survives the scaling.
    """
    data = np.asarray(stems, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if gain is None:
        gain = peak_gain(data)
    if gain != 1.0:
        data = data * gain
    sf.write(str(path), data.T, sample_rate, subtype="PCM_16")


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Read mono float32, resampling with soxr if needed."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        try:
            import soxr
            mono = soxr.resample(mono, sr, sample_rate)
        except ImportError:
            idx = np.linspace(0, len(mono) - 1, int(len(mono) * sample_rate / sr))
            mono = np.interp(idx, np.arange(len(mono)), mono)
    return np.ascontiguousarray(mono, dtype=np.float32)
