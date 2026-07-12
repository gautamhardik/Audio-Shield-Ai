

<div align="center">

# 🛡️ AudioShield AI

### Enterprise-Grade Deepfake Audio Detection & Forensic Analysis Platform

**Detect synthetic voices with 99.11% accuracy using self-supervised speech representations, temporal sequence modeling, and attention-based forensic reasoning.**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-WavLM-yellow)](https://huggingface.co/docs/transformers/model_doc/wavlm)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://hardik-25-audioshield.hf.space/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.11%25-cyan)](https://hardik-25-audioshield.hf.space/)
[![AUC](https://img.shields.io/badge/AUC-0.9994-blue)](https://hardik-25-audioshield.hf.space/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[🚀 Live Demo](https://hardik-25-audioshield.hf.space/)** · **[📊 Performance](#-diagnostic-performance-metrics)** · **[🏗️ Architecture](#️-system-architecture)**

</div>

<br>

<div align="center">
<img src="assets/hero-landing.png" alt="AudioShield AI landing page" width="100%">
</div>

---

## 💡 Why AudioShield AI?

Voice cloning technology has crossed a threshold. Tools that once required studios and hours of training data can now clone a voice from seconds of audio — and the output is good enough to fool the human ear. This isn't a hypothetical risk anymore; it's actively being used in financial fraud, impersonation scams, and disinformation.

Most academic deepfake detectors stop at a Jupyter notebook and an accuracy score. **AudioShield AI doesn't.** It's built as a complete, deployable forensic product:

- 🎯 A production-trained model evaluated with rigorous, leak-free methodology
- 🖥️ A full-stack web application with a real inference backend, not a mock UI
- 📊 An investigation-grade dashboard that explains *why* a clip was flagged, not just *what* it was flagged as
- 📄 Exportable forensic reports suitable for documentation and review

The goal was to build something closer to a real security tool than a portfolio script — from raw waveform to actionable forensic verdict.

---

## ✨ Key Features

| | |
|---|---|
| 🔍 **Real-time Deepfake Detection** | Classifies uploaded audio as Real or Fake in under a second |
| 🧠 **WavLM Transformer Backbone** | Self-supervised speech foundation model captures prosody, speaker identity, and synthetic artifacts |
| 🔁 **BiLSTM Temporal Modeling** | Captures sequential dependencies across the embedding sequence in both directions |
| 🎯 **Attention Pooling** | Learns to weight the most forensically relevant frames instead of naive averaging |
| 📈 **Confidence & Risk Scoring** | Authenticity score, risk assessment (Low/Medium/High/Critical), and calibrated probability |
| 🗺️ **Chronological Suspicion Heatmap** | Segment-by-segment breakdown showing exactly where a clip deviates from organic speech patterns |
| 🎧 **Interactive Waveform Diagnostics** | Playback with suspicious regions highlighted directly on the oscilloscope view |
| 🕹️ **Forensic Auditory Playground** | Blind-listening challenge — test your ear against the model on real/fake samples |
| 📑 **Exportable PDF Reports** | One-click forensic investigation reports for documentation and review |
| 🗂️ **Batch Verification Logs** | Audit trail of historical classifications with filterable risk levels |
| ⚡ **FastAPI Inference Backend** | Model loads once at startup; serves predictions via a real API, not a notebook loop |
| 🐳 **Dockerized Deployment** | Single-container deployment to Hugging Face Spaces, reproducible anywhere |

---

## 🏗️ System Architecture

### End-to-End Request Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    User     │────▶│  React Frontend   │────▶│  FastAPI Backend │
│  (Browser)  │     │  (Vite + Tailwind)│     │    (Uvicorn)     │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                        │
                                                        ▼
                                          ┌──────────────────────────┐
                                          │   Audio Preprocessing    │
                                          │  16kHz · Mono · Normalize│
                                          └────────────┬─────────────┘
                                                        ▼
                                          ┌──────────────────────────┐
                                          │      deployment_model.pt  │
                                          │   (loaded once at startup)│
                                          └────────────┬─────────────┘
                                                        ▼
                                            Prediction + Confidence
                                                        │
                                                        ▼
                                          ┌──────────────────────────┐
                                          │   JSON Response → React   │
                                          │   Forensic Dashboard      │
                                          └──────────────────────────┘
```

### Model Pipeline Architecture

```
   Audio Input          WavLM Base+           BiLSTM            Attention Pooling        Classifier
  (WAV/MP3/FLAC)   ──▶  SSL Representation ──▶  Temporal  ──▶     Feature Weights   ──▶   Softmax
                        (Contextual Embeddings)  Context                                  (Real / Fake)
```

| Stage | Role |
|---|---|
| **Preprocessing** | Resamples to 16kHz, converts to mono, normalizes amplitude, handles variable-length audio via padding/truncation |
| **WavLM Base+** | Microsoft's self-supervised speech foundation model — extracts rich contextual embeddings capturing speaker traits, prosody, and synthetic artifacts |
| **BiLSTM** | Models temporal dependencies across the embedding sequence in both forward and backward directions |
| **Attention Pooling** | Learns to weight the most discriminative frames rather than pooling naively, improving sensitivity to localized synthetic artifacts |
| **Classifier** | Fully connected layer with dropout + GELU activation, outputting a calibrated Real/Fake probability |

> Pre-trained self-supervised speech representations provide strong zero-shot generalization to unseen deepfake generation methods — a key reason this architecture outperforms handcrafted-feature baselines (MFCC, Chroma, Spectrogram).

---

## 🧰 Tech Stack

<table>
<tr>
<td valign="top" width="25%">

**Machine Learning**
- Python
- PyTorch
- Hugging Face Transformers
- WavLM Base+
- NumPy · Pandas

</td>
<td valign="top" width="25%">

**Audio Processing**
- Librosa
- SoundFile

</td>
<td valign="top" width="25%">

**Frontend**
- React
- Vite
- Tailwind CSS

</td>
<td valign="top" width="25%">

**Backend & Deployment**
- FastAPI
- Uvicorn
- Docker
- Hugging Face Spaces

</td>
</tr>
</table>

---

## 📊 Dataset

Trained and evaluated on a **combined dataset** merging **Fake-or-Real (FoR)** and **ASVspoof 2019 LA** — **162,374** total samples.

| Dataset | Samples |
|---|---:|
| ASVspoof 2019 LA | 121,461 |
| Fake-or-Real (FoR) | 40,913 |
| **Combined Total** | **162,374** |

| Split | Samples |
|---:|---:|
| Training | 55,584 |
| Validation | 30,919 |
| Testing | 75,871 |

The test set (75,871 samples) combines 71,237 ASVspoof + 4,634 FoR samples, providing a rigorous held-out evaluation across diverse attack types and recording conditions.

---

## 🔬 Audio Preprocessing

Every recording is standardized before feature extraction:

1. **Loading** — via Librosa / SoundFile
2. **Resampling** — converted to 16kHz (WavLM's expected input rate)
3. **Mono conversion** — removes channel-based variance
4. **Normalization** — amplitude-normalized so volume doesn't bias predictions
5. **Padding / Truncation** — handles variable-length clips during batching (max 96,000 samples / 6 sec)

---

## 🧠 Feature Extraction — Why WavLM?

Traditional handcrafted features (MFCC, Chroma, Spectrograms) were deliberately avoided in favor of **WavLM Base+**, Microsoft's self-supervised speech foundation model.

```
Raw Audio → WavLM Processor → Transformer Encoder → Contextual Embeddings → BiLSTM → Attention Pooling → Fixed Vector
```

**Why this matters:** WavLM is pretrained on massive unlabeled speech corpora, learning representations of speaker characteristics, prosody, and speech dynamics that generalize far better than hand-engineered spectral features — particularly against unseen deepfake generation methods, which is the failure mode that sinks most academic detectors in the real world.

---

## 🏋️ Training Pipeline

- **Loss:** Focal Loss (α=0.25, γ=2.0)
- **Optimizer:** AdamW (lr=1×10⁻⁵, weight_decay=0.01)
- **Scheduler:** Cosine with linear warmup (10% warmup steps)
- **Batch Size:** 32
- **Max Audio Length:** 6 seconds (96,000 samples at 16kHz)
- **Epochs:** 7 (early stopping based on AUC improvement)
- **Mixed Precision:** FP16 via GradScaler
- **Freeze Strategy:** WavLM frozen for first 2 epochs, then unfrozen for fine-tuning
- **Output:** `deployment_model.pt` — the exact artifact served in production

The combined dataset was assembled by merging FoR and ASVspoof 2019 LA, then splitting into train/validation/test sets. Training was performed on Kaggle with GPU acceleration.

---

## 📈 Diagnostic Performance Metrics

*Evaluated on the combined held-out test set (75,871 samples: 71,237 ASVspoof + 4,634 FoR).*

| Metric | Overall | Real (Bonafide) | Spoof (Fake) |
|---|---:|---:|---:|
| **Accuracy** | **99.11%** | — | — |
| **Precision** | — | 94.14% | 99.87% |
| **Recall** | — | 99.12% | 99.10% |
| **F1 Score** | — | 96.57% | 99.49% |
| **ROC-AUC** | **0.9994** | — | — |
| **Equal Error Rate (EER)** | **0.89%** | — | — |

</div>

---

## ✅ Why These Results Are Trustworthy

High numbers invite skepticism — rightfully so. Here's why this isn't overfitting:

- **No data leakage** — splits were created before training; test data was never seen during training or model selection.
- **Large, diverse test set** — 75,871 held-out samples spanning two independent datasets with different attack types ensures robust evaluation.
- **Large training set** — 55,584 labeled samples reduce memorization risk and support generalization.
- **Transfer learning, not memorization** — performance is driven primarily by WavLM's pretrained representations, not example-level memorization.
- **Consistent validation/test alignment** — closely matched performance across splits indicates genuine generalization rather than overfitting to validation.

---

## 📊 Combined Dataset Evaluation

The model was trained on a combined dataset merging **Fake-or-Real (FoR)** and **ASVspoof 2019 LA**, then evaluated on a held-out test set from the same combined distribution.

### Evaluation Methodology

| Component | Detail |
|---|---|
| **Training Datasets** | FoR + ASVspoof 2019 LA (combined) |
| **Total Samples** | 162,374 (55,584 train / 30,919 val / 75,871 test) |
| **Model Checkpoint** | `deployment_model.pt` — best epoch by validation AUC |
| **Preprocessing** | 16kHz resample, mono, head truncation to 6s, WavLM feature extraction |
| **Evaluation Type** | In-distribution held-out test |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **Real (Bonafide)** | 94.14% | 99.12% | 96.57% | 9,619 |
| **Spoof (Fake)** | 99.87% | 99.10% | 99.49% | 66,252 |
| **Overall Accuracy** | **99.11%** | | | **75,871** |

The model achieves near-perfect spoof detection (99.87% precision, 99.10% recall) with a slight trade-off in real-class precision (94.14%), meaning it occasionally flags real audio as suspicious — a conservative bias that favors security over false negatives.

### How to Reproduce

```bash
cd cross_dataset_evaluation
python run_inference_for.py          # Run FoR inference
python compute_metrics_for.py        # Compute metrics + visualizations
```

See the [`model evaluation`](model%20evaluation/) directory for full metrics, confusion matrix, ROC/PR curves, and score distributions.

---

## 🖥️ Web Application

### Forensic Sandbox

Drag-and-drop upload with reference benchmark cases pulled from the FoR and ASVspoof 2019 LA datasets, plus live duration/sample-rate/inference-time readouts per file.

<img src="assets/forensic-sandbox.png" alt="Forensic Sandbox upload interface" width="100%">

### Investigation Dashboard — Analysis Overview

Authenticity score, detection profile, model backbone, investigation findings, confidence distribution curve, and an acoustic anomaly radar chart (Spectral Inconsistency, Vocoder Footprint, Phase Coherence, Breath Mark Gaps, Jitter/Shimmer Ratio).

<img src="assets/dashboard-overview.png" alt="Forensic Investigation Dashboard - Analysis Overview" width="100%">

### Forensic Metrics — Chronological Suspicion Heatmap

Segment-by-segment breakdown of suspicion scores across the clip timeline, paired with an interactive waveform oscilloscope that highlights suspicious regions directly on playback.

<img src="assets/forensic-metrics-heatmap.png" alt="Chronological suspicion heatmap and waveform diagnostics" width="100%">

### Investigation Report

One-click downloadable forensic PDF report — audio summary, key findings, and full timeline analysis, ready for documentation or review.

<img src="assets/investigation-report.png" alt="Forensic Investigation Report" width="100%">

### Model Pipeline & Diagnostic Metrics

<img src="assets/model-pipeline.png" alt="Model Pipeline Architecture panel" width="100%">
<img src="assets/performance-metrics.png" alt="Diagnostic Performance Metrics panel" width="100%">

---

## 📁 Repository Structure

```
Audio-Shield-AI/
│
├── cross_dataset_evaluation/       # Cross-dataset generalization evaluation
│   ├── download_asvspoof.py       # ASVspoof 2019 LA download script
│   ├── run_inference.py           # Pure inference on ASVspoof
│   ├── compute_metrics.py         # Metrics + visualizations
│   ├── error_analysis.py          # FP/FN + attack-type analysis
│   ├── generalization_report.py   # Honest generalization report
│   ├── run_all.py                 # Orchestrator
│   ├── asvspoof_data/             # Dataset (downloaded)
│   ├── results/                   # Metrics + CSVs
│   └── figures/                   # ROC, PR, confusion matrix, etc.
│
├── notebooks/
│   ├── 1_eda-data-validation.ipynb
│   ├── 2_preprocessing.ipynb
│   ├── 3_Model.ipynb
│   ├── 4_Evaluation.ipynb
│   └── 5_inference.ipynb
│
├── audioshield-app/              # React frontend source (Vite)
│   └── src/
│       ├── App.tsx
│       └── index.css
│
├── static/                       # Built frontend (served by FastAPI)
├── assets/                       # Screenshots and images
├── deployment_model.pt           # Trained model checkpoint
├── app.py                        # FastAPI server
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Reproduce Training (Kaggle)

The notebooks were developed entirely on **Kaggle** to take advantage of free GPU access and the dataset's native Kaggle hosting — avoiding a ~70,000-file local download and giving anyone reproducing this work a zero-setup environment.

1. Open `notebooks/1_eda-data-validation.ipynb` in a new Kaggle Notebook
2. Attach the Fake-or-Real Audio Dataset
3. Enable GPU acceleration (Settings → Accelerator → GPU)
4. Run notebooks sequentially: `1 → 2 → 3 → 4`

### Local Deployment

**Clone the repository**
```bash
git clone https://github.com/gautamhardik/Audio-Shield-AI.git
cd Audio-Shield-AI
```

**Backend**
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

**Frontend (development)**
```bash
cd audioshield-app
npm install
npm run dev
```

**Docker (full stack)**
```bash
docker build -t audioshield-ai .
docker run -p 7860:7860 audioshield-ai
```

### Hugging Face Spaces Deployment

The production app runs as a **Docker Space** on Hugging Face, bundling the React frontend, FastAPI backend, and `deployment_model.pt` into a single reproducible container with a public URL:

**🔗 [hardik-25-audioshield.hf.space](https://hardik-25-audioshield.hf.space/)**

---

## 🔮 Future Work

- 🌍 Multilingual deepfake detection
- 🎙️ Streaming / real-time inference for live calls
- 🔎 Explainable AI — frame-level saliency visualization
- 🛡️ Adversarial robustness testing against evasion attacks
- 📱 Mobile deployment (on-device inference)
- 📦 Batch analysis API for bulk investigations
- 🗣️ Speaker attribution / voice fingerprinting

---

## 🔬 Model Evaluation

All evaluation code, results, metrics, and visualizations are available in the [`cross_dataset_evaluation/`](cross_dataset_evaluation/) and [`model evaluation`](model%20evaluation/) directories.

---

## 🙏 Acknowledgements

- **Microsoft** — for WavLM and the broader speech-SSL research that made this possible
- **Hugging Face** — Transformers library and Spaces hosting
- **PyTorch** team
- **Kaggle** — free GPU compute and dataset hosting
- Creators of the **Fake-or-Real** and **ASVspoof** datasets

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

### 🛡️ AudioShield AI

*Enterprise Deepfake Audio Detection Platform*

**[Live Demo](https://hardik-25-audioshield.hf.space/)**

If this project was useful or interesting, consider ⭐ starring the repo.

</div>
