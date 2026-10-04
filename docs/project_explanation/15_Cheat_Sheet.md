# Cheat Sheet

## Important File Locations
* **The Web Server**: `app/main.py`
* **The Orchestrator**: `app/pipeline.py`
* **The Math/DSP**: `app/dsp.py`
* **The UI / Audio Player**: `app/static/app.js`
* **Dependency List**: `requirements.txt`

## Key Commands
* **Start Server**: `run.bat`
* **Check GPU**: `.venv\Scripts\python.exe scripts\gpu_probe.py`
* **Test Offline Pipeline**: `python -m app.pipeline input.mp4 --out runs/test`
