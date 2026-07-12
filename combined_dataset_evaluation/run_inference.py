import os
import json
import warnings
from pathlib import Path
from io import BytesIO
import argparse

import soundfile as sf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import WavLMModel, AutoFeatureExtractor
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "deployment_model.pt"
RESULTS_DIR = ROOT / "cross_dataset_evaluation" / "results"

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 96000
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@torch.no_grad()
def predict(audio_array, model, feature_extractor):
    if len(audio_array) > MAX_AUDIO_LENGTH:
        audio_array = audio_array[:MAX_AUDIO_LENGTH]
    feats = feature_extractor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    iv = feats.input_values.to(DEVICE)
    am = feats.attention_mask.to(DEVICE) if hasattr(feats, "attention_mask") and feats.attention_mask is not None else None
    logits = model(iv, am)
    probs = torch.softmax(logits, dim=1)
    return {
        "prediction": int(float(probs[0][1].item()) >= 0.03487666696310043),
        "real_probability": float(probs[0][0].item()),
        "fake_probability": float(probs[0][1].item()),
        "confidence": float(torch.max(probs).item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.parquet_path:
        parquet_path = Path(args.parquet_path)
    else:
        candidates = list((ROOT / "cross_dataset_evaluation" / "asvspoof_data").rglob("*.parquet"))
        if not candidates:
            print("ERROR: No parquet file found")
            return
        parquet_path = candidates[0]

    print("=" * 60)
    print("Cross-Dataset Evaluation: FoR-trained model on ASVspoof 2019 LA")
    print("=" * 60)
    print(f"Parquet: {parquet_path}")
    print(f"Device:  {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    print(f"\nTotal evaluation samples: {total}")

    model = load_model()
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

    if args.num_samples and args.num_samples < total:
        n = args.num_samples
    else:
        n = total

    print(f"Processing {n} samples (batch size = {BATCH_SIZE})...")
    results = []
    num_row_groups = pf.metadata.num_row_groups
    samples_processed = 0
    audio_buffer = []
    meta_buffer = []

    def flush_buffer():
        nonlocal audio_buffer, meta_buffer
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
        for j, meta in enumerate(meta_buffer):
            p = probs[j]
            results.append({
                "filename": meta["filename"],
                "speaker_id": meta["speaker_id"],
                "system_id": meta["system_id"],
                "key": meta["key"],
                "true_label": meta["true_label"],
                "prediction": int(float(p[1].item()) >= 0.03487666696310043),
                "fake_probability": float(p[1].item()),
                "real_probability": float(p[0].item()),
                "confidence": float(torch.max(p).item()),
            })
        audio_buffer = []
        meta_buffer = []

    for rg_idx in tqdm(range(num_row_groups), desc="Row groups"):
        if samples_processed >= n:
            break
        table = pf.read_row_group(rg_idx)
        pg_batch = table.to_pydict()
        pg_size = len(pg_batch["audio_file_name"])

        for i in range(pg_size):
            if samples_processed >= n:
                break
            try:
                flac_bytes = pg_batch["audio"][i]["bytes"]
                audio, sr = sf.read(BytesIO(flac_bytes))
                if sr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio_buffer.append(audio)
                meta_buffer.append({
                    "filename": pg_batch["audio_file_name"][i],
                    "speaker_id": pg_batch["speaker_id"][i],
                    "system_id": pg_batch["system_id"][i],
                    "key": "bonafide" if pg_batch["key"][i] == 0 else "spoof",
                    "true_label": int(pg_batch["key"][i]),
                })
                samples_processed += 1
                if len(audio_buffer) >= BATCH_SIZE:
                    flush_buffer()
            except Exception as e:
                print(f"  Error: {pg_batch['audio_file_name'][i]}: {e}")
    flush_buffer()

    df = pd.DataFrame(results)
    csv_path = output_dir / "asvspoof_predictions.csv"
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
    print("Next:  python compute_metrics.py")

    with open(output_dir / "inference_summary.json", "w") as f:
        json.dump({"total": len(df), "correct": int(correct), "accuracy_pct": round(acc, 2)}, f, indent=2)


if __name__ == "__main__":
    main()
