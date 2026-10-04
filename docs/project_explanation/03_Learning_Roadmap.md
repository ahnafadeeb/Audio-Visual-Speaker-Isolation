# Learning Roadmap

To fully understand this project, you need to grasp concepts across three domains. If you were building this from scratch, here is what you would need to learn:

### 1. Digital Signal Processing (DSP)
* **STFT (Short-Time Fourier Transform)**: How to convert audio from the time domain (waveforms) to the frequency domain (spectrograms).
* **Wiener Filtering / TF Masking**: How to suppress noise by comparing the power of the target signal vs the total signal.
* **Noise Gating**: How to automatically mute a track when the volume drops below a certain threshold.

### 2. Machine Learning & Vision
* **Audio-Visual Separation**: Understanding how models like AV-MossFormer2 use visual cues (lip movements) as a guide to extract audio.
* **Face Tracking**: Using MediaPipe and S3FD to find faces and track them across frames.
* **Face Embeddings**: Using tools like SFace to recognize that a face in shot A is the same person in shot B.

### 3. Web Engineering
* **Web Audio API**: How to load binary audio data into the browser, split it into channels, and manipulate volume at the sample level using `AudioContext` and `GainNode`.
* **Video Synchronization**: Using `requestVideoFrameCallback` to ensure a muted `<video>` element plays at the exact same speed as an audio track.
