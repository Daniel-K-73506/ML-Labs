# Individual Project Proposal

**Name:** Daniel <surname>
**Student ID:** <your ID>
**Course:** Lab 6 – Training pipelines and project proposal, Vistula University

---

## Problem

Detect whether music is playing and identify which instrument is being played in short audio windows. The target is a low-power device (for example a smart practice assistant or a hearing-aid companion) that must run without cloud access, so the model has to be small and fast.

## Dataset

- **Instrument classes:** NSynth subset (5 instrument families, e.g. keyboard, guitar, string, flute, brass), cut into 1-second windows.
- **"No music" class:** clips from MUSAN and/or ESC-50 (speech, noise, rain, traffic, animal sounds).
- **Input features:** log-mel spectrograms of each 1-second window.
- **Split:** train/validation/test separated by recording, so that no recording appears in more than one split.
- **Licenses:** NSynth (CC BY 4.0), ESC-50 (CC BY-NC 4.0), MUSAN (CC BY 4.0). I will check each license and cite the datasets in the final report.

## Success Metric

Macro F1 ≥ 0.80 across 6 classes (5 instruments + "no music") on the held-out test split. A secondary target is ≥ 95% accuracy on the binary task of music vs. no music.

## Deployment Constraints

- **Latency:** ≤ 50 ms per 1-second window on CPU, including the spectrogram computation.
- **Model size:** ≤ 5 MB on disk.

## Planned Optimization

**Post-training int8 quantization.** A float32 CNN with fewer than 1 million parameters takes about 4 MB; quantizing weights to int8 reduces it to about 1 MB and speeds up CPU inference. If macro F1 drops by more than 2 points after quantization, I will switch to quantization-aware training. Distillation from a larger teacher model is a fallback if the small model does not reach the F1 target.

## Feasibility and Evaluation Plan

- Build the baseline with the YAML-driven training pipeline from this lab (seed, checkpoints, MLflow logging).
- Measure model size with `os.path.getsize` on the saved model file.
- Measure latency as the mean and 95th percentile over 1000 single-window inferences with `torch.no_grad()`, including spectrogram computation.
- Log F1, accuracy, size and latency for both the float32 and the int8 model in MLflow, so the trade-off can be compared directly.

## Risks

- Recognizing instruments in polyphonic real-world music is hard, so the instrument list is limited to 5 classes.
- Single-note NSynth audio differs from real recordings, so I may add a small set of real clips for testing if time allows.
- If the full NSynth dataset is too large for my laptop, I will use a balanced subset.
