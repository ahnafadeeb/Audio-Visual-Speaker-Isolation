# Code Walkthrough

If the examiner asks you to explain the code, here is the flow.

## 1. `app/pipeline.py`
This is the conductor. Look at the `run()` function.
1. It uses `media.extract_audio()` to get the WAV.
2. It uses `FaceAnalyzer` to get face tracks.
3. It calls `_run_avtse()`, passing the audio and the face tracks.
4. Inside `_run_avtse()`, it crops the mouths, feeds them to the `AVTSESeparator`, and gets raw audio estimates back.
5. It then passes those estimates to `dsp.strict_isolation()` and `dsp.apply_gate()` to perfectly silence the pauses.
6. It exports `stems_demo.wav` and `tracks.json`.

## 2. `app/dsp.py`
Look at `gate_mask()`. This is our custom Schmitt Trigger.
It calculates the energy of the frame. It checks if the energy is above `open_db` (-30dB). If yes, it opens. It checks the visual veto: if Speaker B's lips are moving drastically more than Speaker A's, it forces the gate closed to prevent a leak.

## 3. `app/static/app.js`
Look at `engine.init()`. 
It fetches `stems_demo.wav`. It creates a `ChannelSplitterNode`. It maps each channel to a `GainNode`. 
Look at `engine.select(k)`.
It uses `g.gain.setValueCurveAtTime()` to draw the sine/cosine crossfade curves over 25ms.
