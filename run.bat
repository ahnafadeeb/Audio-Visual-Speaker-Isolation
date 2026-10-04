@echo off
REM ---------------------------------------------------------------------------
REM Launch the AV Speaker Isolation server.
REM
REM No --reload.  The reloader re-imports app.main in a second process, which
REM on Windows (no fork) means a second model load -- both copies resident at
REM once during the overlap window, which is how you OOM an 8 GB card while
REM "just editing a CSS file".
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

REM Keep the multi-GB HuggingFace tree inside the project, off C:.
set "HF_HOME=%~dp0.cache\hf"
REM Windows without Developer Mode cannot symlink; the loaders copy instead
REM (LocalStrategy.COPY), so the hub's per-download symlink warning is noise.
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
REM No PYTORCH_CUDA_ALLOC_CONF=expandable_segments here: the Windows CUDA
REM allocator does not implement it, and torch only answers with a warning.

REM Torch spawns one OpenMP pool per interop thread; on a laptop the default
REM oversubscribes every core and makes CPU fallback slower, not faster.
set "OMP_NUM_THREADS=4"
set "KMP_DUPLICATE_LIB_OK=TRUE"

REM Unbuffered so uvicorn's log lines appear as they happen, not in bursts.
set "PYTHONUNBUFFERED=1"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
  echo [warn] .venv not found -- using the python on PATH.
)

if "%~1"=="--preflight" (
  "%PY%" scripts\preflight.py --fetch
  goto :end
)

if "%~1"=="--demo" (
  "%PY%" scripts\make_demo_job.py || goto :fail
  echo.
)

echo Starting on http://127.0.0.1:8000  (Ctrl+C to stop)
echo.
"%PY%" -m app.main
goto :end

:fail
echo.
echo [error] demo job generation failed -- run "run.bat --preflight" to diagnose.
exit /b 1

:end
endlocal
