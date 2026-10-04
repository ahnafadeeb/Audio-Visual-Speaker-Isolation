# How This Project Was Built (History)

If you are asked about the engineering process, explain the evolution.

### V1: The Colab Prototype
We started with SpeechBrain's SepFormer. It did blind separation. We tried to match the output audio to lip movements. It failed miserably because SepFormer normalized its outputs, making silent tracks extremely loud.

### V2: The DSP Era
We realized AI couldn't output digital silence. We wrote `dsp.py` from scratch, implementing Wiener Masking and Schmitt Trigger gating to force the output to 0.0 during pauses. We built the Web Audio player for instant switching.

### V3: The AV-TSE Upgrade
We realized blind separation fundamentally struggles with similar-sounding voices. We ripped out the core and replaced it with AV-MossFormer2, which uses lip video as input to guide the extraction. We kept our V2 DSP chain to clean up the AV-TSE's output, resulting in the final, highly robust system.
