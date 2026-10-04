# Viva Questions & Defenses

**Q: Why not just use a denoiser like DeepFilterNet?**
*Defense*: Denoisers clean background noise (fans, AC). They are trained to *preserve* human speech. If we run a denoiser on a video of two people talking, it preserves *both* people. We needed Speaker Isolation, not Denoising.

**Q: Why did you write a custom DSP gate? Why not just threshold the audio?**
*Defense*: AI separation leaves a "ghost whisper" (a -30dB leak of the other person). If we just thresholded linearly, we would chop off the quiet ends of target words. By using a Schmitt trigger with hysteresis and a visual veto (checking if the other person's mouth is moving), we selectively muted the ghost whispers without clipping the target speaker's words.

**Q: How do you achieve instant switching in the browser?**
*Defense*: We do NOT buffer or seek different audio files. We load one multichannel file. Channel 0 is person A, Channel 1 is person B. We play them simultaneously but set the volume (Gain) of the unwanted channel to 0. Switching just crossfades the gains in 25ms. Because it's one file, they can never drift out of sync.

**Q: Why is your video syncing loop necessary?**
*Defense*: HTML5 `<video>` and `AudioContext` run on different hardware clocks. Over 3 minutes, they drift. We use `requestVideoFrameCallback` to read the exact screen-time of the video and dynamically nudge the video's `playbackRate` to catch up to the audio.
