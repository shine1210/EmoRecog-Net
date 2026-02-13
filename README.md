# EmoRecog-Net  
A Multimodal Dual-Task Framework for Emotion Recognition from fMRI and Physiological Signals  

## 🔍 Overview

EmoRecog-Net is a multimodal deep learning framework designed for **video-induced emotion categorization and intensity recognition** from synchronized neuroimaging and physiological signals.

The model jointly learns:

- Spatial representations from fMRI  
- Temporal dependencies via recurrent modeling  
- Nonlinear cross-modal fusion of neural and autonomic signals  

This repository provides the implementation used in our study:

> EmoRecog-Net: A Multimodal Framework for Emotion Recognition from fMRI and Physiological Signals  
> (Under review)

---

## 🧠 Model Architecture

EmoRecog-Net integrates:

- **Two-stream CNN backbone** for fMRI spatial feature extraction  
- **GRU-based temporal encoder** (128 hidden units)  
- **MLP-based nonlinear multimodal fusion** for heart rate and respiration rate  
- **Dual-task output heads**:
  - Emotion categorization (positive / neutral / negative)
  - Emotion recognition (self-reported intensity regression)

The design emphasizes:
- Data efficiency under small-sample conditions  
- Robustness to missing physiological channels  
- Structured cross-modal alignment  

---

## 📊 Dataset

Experiments were conducted on the **ICBHI 2024 Multimodal Emotion Benchmark**, which includes:

- fMRI regional time-series
- Photoplethysmography (PPG)
- Respiration signals
- Video-evoked emotional stimuli

⚠️ This repository does **not** include the dataset.  
Please obtain the data from the official ICBHI 2024 Scientific Challenge website:  
https://www.icbhi2024.com/sc  

All data usage must comply with the terms and policies specified by the challenge organizers.

## 📦 Pretrained Weights

The pretrained model weights corresponding to the results reported in the manuscript are publicly available via GitHub Release.

🔗 **Download link:**  
https://github.com/shine1210/EmoRecog-Net/releases/tag/v1.0  

**Release version:** `v1.0`  
- Includes the trained **EmoRecog-Net (CNN-GRU-MLP)** model  
- Trained on the ICBHI 2024 multimodal benchmark  
- Compatible with the current repository version  

