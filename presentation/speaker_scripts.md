# Speaker scripts: Audio-Visual Speaker Isolation

EEE 312 (2026) Final Project, Group 4 (Section C2). Four speakers, about 2 minutes each.


## SPEAKER 1 – Redwan Nur Alam Shilad (2206177)

**Slide 1. Audio-Visual Speaker Isolation** (~15 s)

Good morning. We are Group 4 from Section C2 – I'm Redwan, with Ahnaf, Rifat and Shuvo. Our project is Audio-Visual Speaker Isolation: give it a video where people talk over each other, click a face, and you hear only that person.

**Slide 2. Outline** (~5 s)

We'll go from the problem, through our design and a demonstration, to evaluation – each of us covers about two minutes.

**Slide 3. 1. Summary / Abstract** (~20 s)

In one line: we watch each speaker's lips to pull their voice out of one mixed recording. On our own recording each channel reads at 3.5 percent word error, against 110 for audio-only separation. Switching takes 25 milliseconds and never clicks.

**Slide 4. 2. Introduction** (~15 s)

The problem is the cocktail party: one microphone adds the voices into a single waveform. People cope by watching lips. Here two of us read different texts at the same time; our goal is to follow just one.

**Slide 5. 3.1 Design: Problem Formulation (PO(b))** (~15 s)

In scope: any video with two or more visible speakers and one microphone, in any language, processed once and then played interactively. Out of scope: people who are never on screen, live streams, and faces too small to read lips from.

**Slide 6. 3.1 Design: Problem Formulation (PO(b))** (~20 s)

Blind separators like SepFormer split voices by sound alone; they don't know which voice is whose, and they fail on similar voices. Our first version matched voices to faces by lip-loudness correlation, and we measured it at chance. Lip-conditioned extraction gives the network the target's lips; we build on it.

**Slide 7. 3.1 Design: Problem Formulation (PO(b))** (~15 s)

Formally, the recording x is the sum of the voices plus noise. Given each face's lip video V-k, we estimate that person's voice. Three objectives: isolation, fidelity, and instant interaction – on an 8 GB laptop GPU.

**Slide 8. 3.1 Design: Problem Formulation (PO(b))** (~15 s)

Three measurements shaped the design: the matcher was at chance, blind separation interleaved both texts, and the model works best on 2-second windows – the channels become more distinct as windows shrink. Ahnaf will show the design.


## SPEAKER 2 – Ahnaf Hasan Adeeb (2206178)

**Slide 9. 3.2 Design Methods (PO(a))** (~20 s)

Eight stages. ffmpeg normalises the video; MediaPipe finds faces and SFace keeps identities across camera cuts; we crop 112-pixel mouth regions, and AV-MossFormer2 extracts one voice per face. The three red stages are ours: refinement, leak cancellation, and a gate for exact silence.

**Slide 10. 3.2 Design Methods (PO(a))** (~20 s)

Our DSP. Refinement re-extracts each voice from the mixture minus the others, after removing anything phase-coherent with our own estimate, so we never subtract ourselves. The leak canceller removes, per STFT bin, the part of a channel that is a copy of another, weighted by ownership. A Schmitt gate gives exact, click-free silence.

**Slide 11. 3.2 Design Methods (PO(a))** (~15 s)

In the browser, all voices sit in one buffer with one clock, so they cannot drift. Each channel has its own gain; clicking a face runs a 25-millisecond equal-power crossfade, and strict or natural is the same crossfade to another set of channels.

**Slide 12. 3.4 Design: Simulation** (~20 s)

To measure leakage we built a ground-truth bench from our own recording: face A's channel plus face B's channel shifted by 18 seconds, each with its own lips, so we know exactly what the output should be. Refinement and then leak cancellation raise the signal-to-interference ratio from about 10 to 16 decibels.

**Slide 13. 4 Implementation: Demonstration** (~20 s)

This is the best of our five test videos – a split-screen debate, a man and a woman talking over each other. Selecting A or B plays only that voice. The correlation map proves the pairing: each face's lip motion correlates with its own isolated channel – 0.78 and 0.32 – far more than with the other channel, and the two channels' voices have a similarity of only 0.08.

**Slide 14. 4 Implementation: Demonstration** (~10 s)

The spectrograms show it directly: channel A is silent between the man's interjections, and his loudness rises and falls with his lip motion. Rifat continues.


## SPEAKER 3 – S. M. Khairul Islam Rifat (2206179)

**Slide 15. 4.1 Implementation: Photo Gallery** (~15 s)

It works beyond one clip: our own phone recording; a Bengali talk show with camera cuts, where each person keeps their identity across shots; and the full upload, progress and play flow.

**Slide 16. 5. Design Analysis and Evaluation** (~5 s)

We evaluated the design along these seven lines.

**Slide 17. 5.1 Novelty** (~25 s)

What is new: lip-conditioned extraction replaced a chance-level matcher; refinement that subtracts the other speakers without cancelling your own voice; a leak canceller with a live strict-or-natural switch in one buffer; appearance re-identification across camera cuts; and a leak bench built from our own recordings. Every challenge we listed at the progress update is now addressed.

**Slide 18. 5.2 Design Considerations (PO(c))** (~20 s)

Design considerations: everything runs locally and nothing is uploaded; switching is click-free with one global gain, so there are no sudden loudness jumps. For the environment, we skip windows where a face is off screen, halving compute on edited video. And it is language-independent – tested in English and Bengali.

**Slide 19. 5.3 Investigations (PO(d))** (~25 s)

Our investigations: audio-only separation scored over 100 percent word error on our recording; lip-conditioned extraction brought it to 3.5. Across the test videos, strict mode cuts cross-talk where voices overlap, and the CNN debate is the cleanest at 0.08. We also tried a newer 2026 model, extra passes, sharper masks and a speaker-ID gate – all measured worse, so we rejected them.

**Slide 20. 5.4 Limitations of Tools (PO(e))** (~15 s)

Limitations of our tools: the model is 16 kHz only, so content above 8 kHz is lost; faces must be visible and at least about 150 pixels; and with two similar voices overlapping nonstop, the other voice remains 13 to 16 decibels down.

**Slide 21. 5.5 Impact Assessment (PO(f))** (~10 s)

Impact: accessibility for hearing-impaired listeners and for meetings; legally, consent and privacy of the people recorded, and the licences of the models we use.

**Slide 22. 5.6 Sustainability Evaluation (PO(g))** (~10 s)

Sustainability: about three watt-hours of GPU energy per minute of video, no new hardware, and no training – we reuse open pretrained models. Shuvo will wrap up.


## SPEAKER 4 – Md. Shahriare Arefin Shuvo (2206180)

**Slide 23. 5.7 Ethical Issues (PO(h))** (~20 s)

Ethically: our test recordings are of ourselves, with consent; broadcast clips were used only for testing. Processing stays on the machine. Because isolation could be misused for eavesdropping, it only works on faces that are visible, and we report our failures and residual leakage honestly. We also acknowledge the AI coding tools we used.

**Slide 24. 6. Reflection on Individual and Team work ** (~5 s)

Now our reflection on individual and team work.

**Slide 25. 6.1 Individual Contribution of Each Member** (~20 s)

Our work split four ways. Redwan framed the problem: the literature, the scope and our test set. Ahnaf built the architecture and the DSP – the AV-TSE integration, refinement and the leak canceller – and the backend. Rifat ran the evaluation: the leak bench, word-error scoring and the model comparisons. Shuvo set up the vision stage and the player, and led the ethics and cost analysis.

**Slide 26. 6.2 Mode of TeamWork and Diversity** (~15 s)

We worked from one shared design document and backed every decision with a measurement. Our skills complement each other – DSP theory, deep learning, web engineering and evaluation – and we tested for inclusiveness: English and Bengali, male and female voices, phone and broadcast video.

**Slide 27. 6.3 Logbook of Project** (~3 s)

(Show only.) Our logbook – from the Colab prototype to today.

**Slide 28. 7 Communication to External Stakeholders (PO(j))** (~10 s)

Our code, documentation and a demo video are shared on GitHub and YouTube.

**Slide 29. 8. Project Management and Cost Analysis (PO(k))** (~15 s)

Cost: we used hardware we already owned and open-source software, so the prototype cost nothing extra. Deployed as software, the marginal cost is about three watt-hours of GPU energy per minute of video.

**Slide 30. 9. Future Work (PO(l))** (~20 s)

Next: record solo reference takes so we can score SI-SDR, PESQ and STOI; fine-tune on similar voices overlapping nonstop; recover the band above 8 kHz; handle off-screen speakers with voice enrollment; and move toward real time.

**Slide 31. 10. References** (~5 s)

(Show only.) Our references. Thank you – we're happy to take questions and to show the live demo.
