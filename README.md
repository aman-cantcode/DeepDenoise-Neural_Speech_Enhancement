<div align="center">

# DeepDenoise

### Neural Speech Enhancement using U-Net

*A deep convolutional network that removes background noise from speech recordings — built and trained entirely from scratch*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Highlights

- **U-Net architecture** designed and implemented from scratch in TensorFlow/Keras — no pretrained backbones
- **Trained on 60 GB** of paired noisy/clean speech across three benchmark corpora
- **PESQ improved by +0.74** and **SNR by +4.81 dB** on held-out test data
- **Full signal-processing pipeline** — STFT analysis, spectrogram-domain enhancement, ISTFT synthesis
- **Systematic Engineering Workflow** — modular codebase, custom training loop, automated batch evaluation with quantitative reporting

---

## Overview

DeepDenoise treats speech enhancement as an **image-to-image translation problem**. Instead of processing raw audio, it converts speech into a time-frequency spectrogram — a 2D representation where speech and noise occupy visually distinct patterns — and applies a U-Net to reconstruct the clean version, mirroring the same architecture used in state-of-the-art medical image segmentation.

```
Noisy Audio → STFT → Magnitude Spectrogram → U-Net → Enhanced Spectrogram → ISTFT → Clean Audio
```

---

## Results

<div align="center">

| Metric | Noisy Baseline | Enhanced Output | Δ Improvement |
|:------:|:--------------:|:----------------:|:--------------:|
| **STOI** — Intelligibility | 0.8683 | **0.9210** | +0.0527 |
| **PESQ** — Perceptual Quality | 1.4961 | **2.2395** | +0.7434 |
| **SNR (dB)** — Noise Reduction | 6.93 | **11.73** | +4.81 dB |

</div>

---

## Architecture

A U-Net encoder–decoder with 4 skip connections, operating on log-magnitude spectrograms — engineered to balance global noise context with fine-grained frequency detail.

<img width="1536" height="1024" alt="UNet" src="https://github.com/user-attachments/assets/e9bc31b0-d503-4845-9cc4-8227e2adff7d" />


<div align="center">

<br>

| Component | Detail |
|---|---|
| **Parameters** | ~1.9 million |
| **Loss Function** | L1 (Mean Absolute Error) — chosen over MSE to preserve spectral sharpness |
| **Optimizer** | Adam, lr = 1e-4, gradient clipping at global norm 1.0 |
| **Regularization** | BatchNorm after every convolution |
| **Skip Connections** | 4 — preserve high-resolution frequency detail lost during pooling |

</div>

<details>
<summary><b>Why U-Net for audio?</b></summary>
<br>

Speech enhancement in the spectrogram domain is structurally identical to image segmentation: given a corrupted 2D input, reconstruct a clean version while preserving spatial detail. The encoder learns global noise/speech separation patterns; skip connections give the decoder direct access to high-resolution encoder features that pooling would otherwise permanently discard — critical for reconstructing the fine harmonic structure of speech.
</details>

---

## Dataset

<div align="center">

| Source | Contribution |
|--------|--------------|
| **VoiceBank-DEMAND** | Paired clean/noisy speech recordings |
| **LibriSpeech** | Additional clean speech for diversity |
| **MUSAN** | Real-world noise — babble, ambient, music |

**~60 GB** combined · resampled to 16 kHz mono · sliced into 4-second training segments

</div>

---

## Engineering Highlights

- **Custom `tf.data` pipeline** with parallel audio loading and prefetching — eliminates GPU idle time during training
- **Manual training loop** with `GradientTape`, wrapped in `@tf.function` for graph-mode compilation speedup
- **Gradient clipping** to prevent instability from large-magnitude gradient spikes
- **Checkpointing** every 5 epochs, enabling training resumption without loss of progress
- **Quantitative evaluation harness** — automated STOI/PESQ/SNR scoring with comparison charts across the full test set

---

## Project Structure

```
DeepDenoise/
├── model/unet.py U-Net architecture
├── audio/
│ ├── stft_utils.py STFT ↔ waveform conversion
│ └── slice_audio.py Slice long recordings into 4s clips
├── data/dataset.py tf.data training pipeline
├── evaluation/metrics.py STOI, PESQ, SNR
│
├── weights/ Trained model weights
├── dataset/ Training & test audio (not tracked in git)
│ ├── train/{clean,noisy}/
│ └── test/{clean,noisy}/
├── samples/ Demo audio for quick testing
├── outputs/ Generated results (enhanced audio, charts, reports)
│
├── train.py Train the model
├── enhance.py Enhance a single file
├── evaluate.py Batch evaluation with metrics + charts
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/aman-cantcode/DeepDenoise-Neural_Speech_Enhancement.git
cd DeepDenoise-Neural-Speech-Enhancement-using-U-Net

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Enhance a file:**
```bash
python enhance.py --input samples/noisy_demo.wav
```

**With a clean reference (adds STOI/PESQ/SNR scores):**
```bash
python enhance.py --input samples/noisy_demo.wav --clean samples/clean_demo.wav
```

**Batch evaluation:**
```bash
# place files in dataset/test/clean/ and dataset/test/noisy/
python evaluate.py
```

**Train from scratch:**
```bash
# place 4-second clips in dataset/train/{clean,noisy}/
python train.py
```

---

## Tech Stack

`TensorFlow` `Keras` `NumPy` `librosa` `SoundFile` `pystoi` `pesq` `Matplotlib`

