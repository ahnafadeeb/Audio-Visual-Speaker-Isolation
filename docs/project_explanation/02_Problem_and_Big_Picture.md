# Problem and Big Picture

## The Problem: The Cocktail Party Effect
Imagine being at a loud party. Your brain can focus on one person's voice and tune out the rest by watching their lips and listening to their specific pitch. Computers are terrible at this. When two people talk into a single microphone, their sound waves literally add together: `x(t) = s1(t) + s2(t)`. 

The goal of this project is to take that single mixed audio track `x(t)` and a video of the speakers, and separate it back into `s1(t)` and `s2(t)`.

## Why is it hard?
1. **Blind separation is flawed**: Old AI models (like SepFormer) just listen to the audio and try to split it into two voices. But it doesn't know *who* is *who*. If the voices sound similar, it fails.
2. **AI models leak**: AI models are trained to reduce the volume of the background noise, not eliminate it. During pauses when someone stops talking, you can still hear a faint "ghost whisper" of the other person in their track.
3. **Real-time playback is hard**: If you try to process this live in the browser, it stutters. 

## Our Solution (The Big Picture)
1. **Lip-Conditioned Extraction**: Instead of blind separation, we crop the speaker's mouth from the video and feed it to an AI model alongside the audio. We tell the AI: *"Extract the voice that matches these lips."*
2. **DSP Silence Gate**: To fix the "ghost whisper", we built a custom Digital Signal Processing (DSP) chain. It mathematically detects when a person stops talking and clamps their audio track to absolute digital zero.
3. **Pre-compute and Crossfade**: We process the video on the backend first. We send a single multichannel audio file to the browser. The browser plays it and uses volume dials (`GainNodes`) to switch between speakers instantly without buffering.
