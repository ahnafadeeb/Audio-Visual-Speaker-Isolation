# Models Used

We utilize several state-of-the-art AI models in our pipeline.

## 1. AV-MossFormer2 (The Core Engine)
This is an Audio-Visual Target-Speaker Extraction (AV-TSE) model. 
* **How it works**: It uses cross-attention mechanisms. It looks at the visual features of the lip movements and uses them as a "query" to search the audio mixture and extract the matching sound waves.
* **Why we chose it**: It guarantees that the voice extracted belongs to the face we provided. It solves the "who is who" problem inherently.

## 2. SepFormer (The Fallback)
* **How it works**: A blind separation model based on Transformers. It uses dual-path processing (intra-chunk and inter-chunk) to model long-term audio dependencies.
* **When we use it**: If the faces are entirely hidden, we fall back to this to separate the audio, then attempt to match it to whatever lip movement we *can* see.

## 3. SFace / S3FD
* **S3FD**: A robust face detector used to find the faces.
* **SFace**: A face recognition model. If the camera cuts from shot 1 to shot 2, SFace compares the faces to ensure we don't accidentally swap Speaker A and Speaker B's audio tracks.
