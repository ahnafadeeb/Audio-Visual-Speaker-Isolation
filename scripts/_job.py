"""Shared helpers for the diagnostic scripts.

Exists for one reason: **which video file a diagnostic reads changes its answer**,
and three scripts were reading the wrong one.

A job directory holds two videos:

    input.mp4   the user's upload, untouched -- 29.98 fps on the test clip
    video.mp4   what app/pipeline.py actually analyses: normalize_video() re-encodes
                the upload to cfg.vision.target_fps (25.0) before FaceAnalyzer
                ever sees it

``diag_match.py``, ``diag_visual.py`` and ``fit_gate.py`` all preferred
``input.mp4``, so they measured a decode the pipeline never used: 968 frames at
29.98 fps against the shipped 809 at 25.0.  That is not a rounding difference.
Re-running the matcher on the two decodes of the SAME clip returns *opposite*
pairings -- ``[1, 0]`` at 25 fps and ``[0, 1]`` at 29.98 -- which is also the
cleanest demonstration on record that the pairing is a coin flip: the frame rate
of the transcode should not be able to decide who is speaking, and here it does.

It further explains a discrepancy that had been left as "input-side differences":
the harness reporting confidence 0.0679 while ``meta.json`` stored 0.0376 for the
same job.  Two decodes, two independent flips.

So: anything explaining what the user *heard* must read ``video.mp4``.  Only reach
for ``input.mp4`` when the question is about the upload itself.

**Audio is a separate question with the opposite answer**, and an earlier version
of this note got it wrong by asserting "audio is not affected".  It is:
``normalize_video`` passes ``-an``, deliberately, so that the browser can never
be handed the unseparated mixture.  ``video.mp4`` therefore has *no audio stream*
and is never a valid audio source -- ``diag_f0.py`` fell back to it and ffmpeg
exited with "Output file does not contain any stream", which at least fails
loudly.  The samples the separator saw came from ``input.mp4`` via
``extract_audio``.  Use ``pipeline_audio()`` for that, and note it can legitimately
have to look outside the job directory: the CLI entry point
(``pipeline.py <in> --out runs/x``) does not copy the upload into the output dir,
so CLI-produced jobs have no ``input.mp4`` of their own.
"""

from __future__ import annotations

from pathlib import Path


def pipeline_video(job: Path, *, quiet: bool = False) -> Path:
    """The video the pipeline analysed, or the closest available stand-in.

    Raises if the job directory holds neither, because silently proceeding with
    no video is how a diagnostic ends up reporting "no tracks" as a finding.
    """
    shipped, upload = job / "video.mp4", job / "input.mp4"
    if shipped.exists():
        return shipped
    if upload.exists():
        if not quiet:
            print(f"!! no video.mp4 in {job}: falling back to input.mp4, which the "
                  f"pipeline never analysed.\n"
                  f"   Frame rate differs from the shipped decode, and the matcher "
                  f"is sensitive to that -- treat any pairing below as indicative "
                  f"only.")
        return upload
    raise SystemExit(f"no video.mp4 or input.mp4 in {job}")


def pipeline_audio(job: Path, *, quiet: bool = False) -> Path:
    """A video file whose audio is the mixture the separator was fed.

    Never ``video.mp4``: ``normalize_video`` strips the audio with ``-an``, so
    that file has no audio stream and ffmpeg fails on it.  Prefers the job's own
    ``input.mp4``; failing that, the most recently modified ``input.mp4``
    elsewhere under ``runs/``, because a job produced by the CLI has no upload of
    its own.  That fallback is only sound when the runs are the same clip, so it
    says so loudly rather than quietly returning a different recording.
    """
    own = job / "input.mp4"
    if own.exists():
        return own
    siblings = sorted((p for p in job.parent.glob("*/input.mp4")),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not siblings:
        raise SystemExit(
            f"no input.mp4 in {job} and none elsewhere in {job.parent}; "
            f"video.mp4 carries no audio (normalize_video passes -an), so there "
            f"is no mixture to read")
    if not quiet:
        print(f"!! no input.mp4 in {job} (the CLI does not copy the upload into "
              f"--out).\n   Reading the mixture from {siblings[0]} instead -- valid "
              f"ONLY if that is the same clip.")
    return siblings[0]
