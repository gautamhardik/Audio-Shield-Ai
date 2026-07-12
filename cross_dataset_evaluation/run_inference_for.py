import os
import json
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from transformers import WavLMModel, AutoFeatureExtractor
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "deployment_model.pt"
FOR_DATA_DIR = ROOT / "cross_dataset_evaluation" / "for_data"
RESULTS_DIR = ROOT / "cross_dataset_evaluation" / "results_for"

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 96000
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.03487666696310043


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * x, dim=1)


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        hidden_size = self.wavlm.config.hidden_size
        self.lstm = nn.LSTM(hidden_size, 256, 1, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 256)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, input_values, attention_mask=None):
        x = self.wavlm(input_values, attention_mask=attention_mask).last_hidden_state
        x, _ = self.lstm(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def load_model():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = DeepfakeDetector()
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(DEVICE).eval()
    print("Model loaded.")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_dir = FOR_DATA_DIR / "fake"
    real_dir = FOR_DATA_DIR / "real"

    if not fake_dir.exists() or not real_dir.exists():
        print(f"ERROR: FoR data not found at {FOR_DATA_DIR}")
        return

    fake_files = sorted([f for f in fake_dir.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac")])
    real_files = sorted([f for f in real_dir.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac")])

    all_files = [(f, 1) for f in fake_files] + [(f, 0) for f in real_files]

    if args.num_samples and args.num_samples < len(all_files):
        import random
        random.seed(42)
        all_files = random.sample(all_files, args.num_samples)

    print("=" * 60)
    print("FoR Test Set Evaluation")
    print("=" * 60)
    print(f"Fake samples: {len(fake_files)}")
    print(f"Real samples: {len(real_files)}")
    print(f"Total:        {len(all_files)}")
    print(f"Device:       {DEVICE}")

    model = load_model()
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

    print(f"\nProcessing {len(all_files)} samples (batch size = {BATCH_SIZE})...")
    results = []
    audio_buffer = []
    meta_buffer = []

    results = []
    saved_count = 0

    def flush_buffer():
        nonlocal audio_buffer, meta_buffer, saved_count
        if not audio_buffer:
            return
        feats = feature_extractor(
            audio_buffer, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_AUDIO_LENGTH
        )
        iv = feats.input_values.to(DEVICE)
        am = feats.attention_mask.to(DEVICE) if hasattr(feats, "attention_mask") and feats.attention_mask is not None else None
        logits = model(iv, am)
        probs = torch.softmax(logits, dim=1)
        batch_rows = []
        for j, meta in enumerate(meta_buffer):
            p = probs[j]
            batch_rows.append({
                "filename": meta["filename"],
                "true_label": meta["true_label"],
                "prediction": int(float(p[1].item()) >= THRESHOLD),
                "fake_probability": float(p[1].item()),
                "real_probability": float(p[0].item()),
                "confidence": float(torch.max(p).item()),
            })
        results.extend(batch_rows)
        saved_count += len(batch_rows)
        if saved_count % 500 == 0:
            pd.DataFrame(results).to_csv(output_dir / "for_predictions_partial.csv", index=False)
        audio_buffer = []
        meta_buffer = []

    for filepath, label in tqdm(all_files, desc="Inference"):
        try:
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_buffer.append(audio)
            meta_buffer.append({
                "filename": filepath.name,
                "true_label": label,
            })
            if len(audio_buffer) >= BATCH_SIZE:
                flush_buffer()
        except Exception as e:
            print(f"  Error: {filepath.name}: {e}")
    flush_buffer()

    df = pd.DataFrame(results)
    csv_path = output_dir / "for_predictions.csv"
    df.to_csv(csv_path, index=False)

    correct = (df["true_label"] == df["prediction"]).sum()
    acc = correct / len(df) * 100
    print(f"\n{'=' * 60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Evaluated: {len(df)}")
    print(f"Correct:   {correct}")
    print(f"Wrong:     {len(df) - correct}")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"\nSaved: {csv_path}")

    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump({"total": len(df), "correct": int(correct), "accuracy_pct": round(acc, 2)}, f, indent=2)


if __name__ == "__main__":
    main()
