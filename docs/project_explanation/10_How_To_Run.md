# How To Run

## Setup
1. Ensure Python 3.12 is installed.
2. Run the virtual environment setup:
   `py -3.12 -m venv .venv`
   `.venv\Scriptsctivate`
3. Install PyTorch matching your CUDA version (e.g., cu128 for RTX 4000/5000 series):
   `pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128`
4. Install the rest:
   `pip install -r requirements.txt`

## Running the Server
Simply execute:
`run.bat`

This starts the FastAPI server on `127.0.0.1:8000`. Navigate to that URL in your browser, upload a video, and the pipeline will process it.
