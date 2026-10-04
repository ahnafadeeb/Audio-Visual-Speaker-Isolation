# Tools and Libraries

Here is the tech stack we chose and exactly *why* we chose it.

## Backend (Python)
* **PyTorch (`torch`, `torchaudio`)**: The core deep learning framework. We pinned it to `2.11.0` specifically because it's the last stable version before `torchaudio` removed certain APIs.
* **FastAPI**: The web server. Chosen because it handles asynchronous requests (like uploading large videos) beautifully and supports Server-Sent Events (SSE) for our progress bar.
* **MediaPipe (Pinned to `0.10.21`)**: Used for face tracking. We pinned it because newer versions removed `mp.solutions.face_mesh` which we rely on for the blind-separation fallback path.
* **SpeechBrain**: A toolkit for speech processing, used for our SepFormer fallback model.
* **SciPy / NumPy**: Used heavily in `dsp.py` for STFT math and array manipulations.

## Frontend (JavaScript)
* **Vanilla JavaScript / HTML5 / CSS3**: No React or Vue. We needed direct, low-level access to the DOM and Web Audio API to guarantee 25ms switching speeds. Frameworks add overhead.
* **Web Audio API**: Built into modern browsers. Used to decode the audio, split channels, and apply crossfades.
