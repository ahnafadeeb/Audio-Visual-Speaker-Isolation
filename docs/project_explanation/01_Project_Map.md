# Project Map

Here is the mental map of where everything lives in the codebase.

## Root Directory
* `README.md` - Quick start guide.
* `ARCHITECTURE_V2.md` - The master design document explaining the engineering rationale.
* `requirements.txt` - Python dependencies.
* `run.bat` - The startup script for Windows.

## `app/` (The Core Backend)
* `main.py` - The FastAPI web server.
* `pipeline.py` - The orchestrator. It takes a video, runs it through vision, ML, and DSP, and spits out the separated audio.
* `separation.py` - Wraps the ML models (AV-TSE and SepFormer) to do the heavy lifting in chunks.
* `dsp.py` - The custom signal processing (Wiener masking, Schmitt trigger gating).
* `vision.py` & `roi.py` - MediaPipe face tracking and lip cropping.
* `matching.py` - Used in the fallback path to match blind audio separation to lip movements.
* `adapt.py` - Logic to fine-tune the model to specific presenters.
* `jobs.py` - Manages the processing queue.

## `app/static/` (The Frontend)
* `index.html` - The UI structure.
* `app.js` - The Web Audio API player, syncs the video, and handles instant switching.
* `style.css` - The look and feel.

## `scripts/` (Diagnostics)
* Contains 40+ scripts used during development to prove our math worked (e.g., `bench_vram.py`, `diag_perm.py`).
