# Training, Evaluation, and Results

How do we prove our system works? We built a ground-truth simulation.

## The Ground-Truth Bench
You cannot measure how much Speaker A leaked into Speaker B if you don't have the original, clean recordings.
1. We recorded two team members reading different texts in a quiet room.
2. We artificially shifted one track by 18 seconds and mixed them together in software.
3. Because we did the mixing, we know the exact "Ground Truth".

## Results (Signal-to-Interference Ratio - SIR)
* **Raw Model Output**: ~10.4 dB SIR. (The other voice is audible).
* **After our DSP Refinement & Leak Cancellation**: **16.3 dB SIR**. 
We mathematically proved our DSP adds ~6 dB of isolation over the raw state-of-the-art AI model.

## Word Error Rate (WER)
We ran the isolated tracks through OpenAI's Whisper model.
* Mixed Audio: 110% WER (Completely unreadable).
* Our Isolated Track: **3.5% WER**. (Near perfect transcription).
