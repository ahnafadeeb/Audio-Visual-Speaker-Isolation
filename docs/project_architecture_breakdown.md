# The Architecture: Interactive Audio-Visual Speech Separation

This document explains every single tool, why it is used, and exactly how the data flows from the moment a user uploads a video to the moment they click a face to hear isolated audio.

---

## 1. The Server & Frontend (How it starts)
* **Tools Used:** FastAPI (Python), Uvicorn, Vanilla HTML/JS/CSS.
* **How it works:** When you run the project, FastAPI starts a local web server. The user opens `localhost:8000` and sees `index.html`. When they drag and drop a video, JavaScript sends a POST request to the backend. The backend puts this into an asynchronous queue and starts the 7-stage processing pipeline.

---

## 2. Stage 1: Audio Extraction (`prepare` stage)
* **Tools Used:** FFmpeg (via Python subprocess).
* **Why:** AI models are incredibly strict. You can't just feed them an iPhone `.mp4` video.
* **How it works:** The Python code tells FFmpeg to rip the audio out of the video and convert it specifically to **16,000 Hz, mono (1 channel), .wav format**. If you don't do this, the AI will crash or output garbage.

---

## 3. Stage 2: Face & Lip Tracking (`faces` stage)
* **Tools Used:** MediaPipe FaceMesh (by Google), OpenCV (for reading video frames).
* **Why:** We need to know when people are moving their mouths so we can match them to the audio later.
* **How it works:** 
  1. OpenCV reads the video frame by frame (e.g., 30 frames per second).
  2. MediaPipe scans each frame and finds up to 2 faces. It places 468 3D coordinates on each face.
  3. **The Math:** The code specifically looks at the coordinates for the **Top Lip** and **Bottom Lip**. It calculates the "Euclidean distance" between them.
  4. **The Output:** This generates a "Lip Envelope"—a mathematical graph showing exactly how wide each person's mouth opens over time.

---

## 4. Stage 3: Audio Separation (`separate` stage)
* **Tools Used:** SpeechBrain, PyTorch, SepFormer (specifically `sepformer-whamr`).
* **Why:** This is the core AI that actually splits the overlapping voices.
* **How it works:** 
  1. The 16kHz WAV file is loaded into the GPU using PyTorch.
  2. SepFormer (a Transformer-based neural network) analyzes the frequencies. It learns the distinct pitch and tone of the two overlapping speakers.
  3. It generates "masks" and applies them to the audio, outputting two completely separate audio tracks (Track 0 and Track 1).
  4. **VRAM Chunking:** If the video is too long, it will crash the GPU memory. Your code has a smart `AudioConfig` feature that slices the audio into smaller chunks, processes them, and glues them back together.
  5. **The Problem:** SepFormer is "permutation invariant." It hands us Track 0 and Track 1, but it has no idea which face they belong to.

---

## 5. Stage 4: Audio DSP (`dsp` stage)
* **Tools Used:** SciPy, Torchaudio.
* **Why:** We have the Audio Tracks, and we have the Lip Envelopes, but they speak different languages. Audio is 16,000 samples per second; Video is 30 frames per second. We need to sync them up to compare them.
* **How it works:** The code calculates the "Energy Envelope" (the loudness) of Track 0 and Track 1. It then applies a low-pass filter (smoothing) and resamples the audio energy down to 30 Hz so it perfectly aligns with the video frames. 

---

## 6. Stage 5: Cross-Modal Matching (`match` stage)
* **Tools Used:** NumPy, SciPy (`linear_sum_assignment`).
* **Why:** To permanently pair the correct audio track to the correct face.
* **How it works:** 
  1. **Pearson Correlation:** The code takes Face A's lip graph and Audio 0's loudness graph and compares them. If the lips open exactly when the audio gets loud, the correlation is close to `1.0`. It does this for every combination (Face A/Audio 0, Face A/Audio 1, Face B/Audio 0, Face B/Audio 1).
  2. **Hungarian Algorithm:** It uses a mathematical optimization formula (the Hungarian Algorithm) to look at all those correlation scores and lock in the absolute best 1-to-1 pairings (e.g., Face A gets Audio 1, Face B gets Audio 0).

---

## 7. Stage 6: The Clean-Up Gates (`gate` stage)
* **Tools Used:** Custom DSP logic in Python.
* **Why:** SepFormer is great, but it leaves "bleed-through." When Face A is talking, you can still hear Face B faintly whispering like a robot in the background of Face A's track.
* **How it works:** 
  1. **Wiener Mask:** It compares the volume of the two tracks. If Track A is overwhelmingly louder than Track B at a specific millisecond, it mathematically suppresses Track B.
  2. **Schmitt Trigger Gate:** This is an audio engineering tool. It acts like a trapdoor. When the audio volume drops below a specific threshold (meaning the person stopped talking), the code brutally forces the audio to `0.0` (exact digital silence).
  3. **The Two Outputs:** The code saves the raw AI output as `stems_raw.wav` (so researchers can grade the AI with math), but it saves the gated, perfectly silent version as `stems_demo.wav` for humans to listen to.

---

## 8. The Frontend Playback (The UI)
* **Tools Used:** JavaScript Web Audio API.
* **Why:** Standard HTML `<audio>` tags lag when you try to switch between them. We need instant, seamless switching.
* **How it works:** 
  1. The browser downloads the video and `stems_demo.wav` (which contains both audio tracks stacked together).
  2. The Web Audio API creates two `GainNodes` (digital volume knobs).
  3. Both audio tracks are playing in the background at the exact same time, perfectly synced to the video.
  4. When you click Face A on the screen, JavaScript instantly turns Face A's GainNode to `1.0` (100% volume) and Face B's GainNode to `0.0` (0% volume). Because they are already playing, the switch takes less than 1 millisecond.
