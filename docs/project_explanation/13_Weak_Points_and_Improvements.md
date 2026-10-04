# Weak Points and Improvements

Every good engineer knows the limits of their system.

## 1. The 16 kHz Limit
The AV-MossFormer2 model is trained on 16 kHz audio. This cuts off high frequencies (above 8 kHz), making sounds like 's' and 'f' sound slightly duller than a studio recording.
*Fix*: Future work requires training the model on 48 kHz data.

## 2. Off-Screen Voices
Because our model is lip-conditioned, if a speaker turns away from the camera or walks off-screen, the model loses its guiding signal and output quality drops or mutes.
*Fix*: Implement voice-enrollment. Use the visual data when available to learn a voice print, and fall back to the voice print when the face disappears.

## 3. Processing Speed
Currently, it runs at about 0.5x real-time on an RTX 5060 laptop (a 1-minute video takes 2 minutes to process). 
*Fix*: It cannot currently be used for live Zoom calls. It is strictly a process-then-play application.
