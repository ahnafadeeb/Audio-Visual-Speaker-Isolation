# Data Flow

How does data move through the system?

## 1. Input
* A raw `.mp4` video uploaded by the user.

## 2. Processing (Backend)
* Extracted Audio: `mixture.wav` (16 kHz, Mono).
* Extracted Vision: Bounding box coordinates for faces.
* Model Input: The AV-TSE model takes the 16 kHz audio array AND a tensor of cropped mouth images (112x112 pixels, grayscale, 25 fps).

## 3. Output (To the Browser)
* `video.mp4`: A muted version of the video.
* `stems_demo.wav`: A single, multichannel audio file. If there are 2 speakers, it has 4 channels (Channels 0 & 1 are strictly isolated, Channels 2 & 3 are natural). 
* `tracks.json`: A JSON file containing the frame-by-frame coordinates of where the faces are on the screen, so the frontend can draw the clickable boxes.
