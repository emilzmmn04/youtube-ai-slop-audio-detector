# YouTube AI-Slop Audio Detector

An experimental Google Colab pipeline that combines two weak-but-complementary signals:

1. transcribe a video with word timestamps;
2. rank long transcript windows for machine-generated writing;
3. sample 4.04-second audio clips from the highest-ranked regions **and** from uniform control regions;
4. score those clips with a synthetic-speech detector; and
5. export clip-level evidence for evaluation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emilzmmn04/youtube-ai-slop-audio-detector/blob/main/youtube_ai_slop_detector_colab.ipynb)

## Run it

1. Open the Colab notebook using the badge above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run the setup and configuration cells.
4. Paste a YouTube URL, or leave it blank to upload a video/audio file.
5. Run the remaining cells in order.

The notebook saves:

- `transcript.json` — timestamped ASR output;
- `transcript_windows.csv` — transcript-window AI-writing scores;
- `clip_scores.csv` — selected/control clip synthetic-voice scores; and
- `run_summary.json` — experimental video-level aggregates.

## Models

- ASR: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
- Text routing: [`GeorgeDrayson/modernbert-ai-detection-raid-mage`](https://huggingface.co/GeorgeDrayson/modernbert-ai-detection-raid-mage)
- Synthetic voice: [`Speech-Arena-2025/DF_Arena_1B_V_1`](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1)

## Important limitations

- Transcript detection is used only to prioritize compute. AI-written text does not prove that a voice is synthetic.
- Uniform control clips are always retained so TTS reading human-written text is not systematically missed.
- Scores are not calibrated for YouTube. The notebook reports evidence; its provisional threshold is not a production blocking rule.
- Video/channel-disjoint evaluation and YouTube/AAC/Opus re-encoding tests are required before comparing accuracy.
- DF Arena is released under a **non-commercial model license**. This repository's MIT license covers only the code and notebook, not downloaded models or datasets.
- YouTube downloads from Colab can occasionally be blocked. File upload is included as a fallback.

## Local execution

This project is intentionally Colab-first. Local model execution is not required. The notebook can still be structurally validated with:

```bash
python3 tests/validate_notebook.py
```

## License

The repository code is MIT-licensed. Third-party models and datasets retain their own licenses.
