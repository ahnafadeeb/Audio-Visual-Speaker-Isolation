# Theory and Concepts

Here are the deep technical concepts you need to explain during a viva.

## 1. STFT and Overlap-Add
To edit audio, we cut it into tiny overlapping frames (e.g., 512 samples), apply a Fourier Transform to see the frequencies, alter them, and then Inverse-Fourier transform them back. Because the frames overlap, we add them together (Overlap-Add) to reconstruct the smooth audio.

## 2. Wiener Masking (The Power Ratio)
In `dsp.py`, we take the estimated voices and calculate a "Mask".
`Mask = Power(Voice A) / (Power(Voice A) + Power(Voice B))`
If Voice A is loud in a specific frequency, the mask is near 1.0 (keep it). If Voice B is louder, the mask is near 0.0 (mute it). We apply this mask to clean up the AI's output.

## 3. The Schmitt Trigger Gate
A normal noise gate opens when audio hits a threshold and closes when it drops below it. This causes "chattering" (rapidly opening and closing on borderline sounds). 
A Schmitt Trigger has *two* thresholds (hysteresis). 
* It opens at -30 dB.
* It won't close again until it drops below -40 dB.
This ensures smooth, stable muting.

## 4. Equal-Power Crossfade
When switching from Speaker A to Speaker B in the browser, if we just drop A's volume linearly and raise B's linearly, the total acoustic energy dips in the middle. We use trigonometric curves:
* Volume A = `cos(x * pi/2)`
* Volume B = `sin(x * pi/2)`
Because `sin^2 + cos^2 = 1`, the total volume stays perfectly flat during the 25ms transition.
