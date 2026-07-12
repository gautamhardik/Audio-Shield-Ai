import os
import json
import warnings
from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

EVAL_PARQUET = "data/test-00000-of-00001.parquet"
DATASET_REPO = "Bisher/ASVspoof_2019_LA"
OUTPUT_DIR = Path(__file__).parent / "asvspoof_data"


def download_eval_parquet():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dest = OUTPUT_DIR / "eval.parquet"
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        size_gb = dest.stat().st_size / 1e9
        print(f"Evaluation parquet already exists ({size_gb:.2f} GB). Skipping download.")
        return str(dest)
    print("Downloading ASVspoof 2019 LA evaluation split from Hugging Face...")
    print(f"  Repo: {DATASET_REPO}")
    print(f"  File: {EVAL_PARQUET}")
    print(f"  Size: ~4.38 GB (evaluation split only, NOT the full 15.7 GB dataset)")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=EVAL_PARQUET,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
    )
    if path and os.path.exists(path):
        size_gb = os.path.getsize(path) / 1e9
        print(f"Downloaded: {path} ({size_gb:.2f} GB)")
        return path
    return None


def extract_protocol_from_parquet(parquet_path, output_dir):
    """Extract protocol/metadata from the parquet without audio."""
    import pandas as pd
    print("Reading parquet metadata (this may take a moment)...")
    df = pd.read_parquet(parquet_path, columns=["speaker_id", "audio_file_name", "system_id", "key"])
    total = len(df)
    print(f"Evaluation samples: {total}")
    bonafide = (df["key"] == 0).sum() if df["key"].dtype in (int, float) else (df["key"] == "bonafide").sum()
    spoof = total - bonafide
    print(f"  Bonafide: {bonafide}")
    print(f"  Spoof:    {spoof}")
    protocol_path = output_dir / "eval_protocol.txt"
    with open(protocol_path, "w") as f:
        for _, row in df.iterrows():
            key_str = "bonafide" if row["key"] == 0 or row["key"] == "bonafide" else "spoof"
            f.write(f"{row['speaker_id']} {row['audio_file_name']} {row['system_id']} {key_str}\n")
    print(f"Protocol saved to {protocol_path}")
    return df, protocol_path


def main():
    parser = argparse.ArgumentParser(description="Download ASVspoof 2019 LA evaluation split")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ASVspoof 2019 LA - Evaluation Split Download")
    print("=" * 60)
    print("NOTE: We only download the evaluation split (~4.38 GB),")
    print("NOT the full dataset (~15.7 GB).")
    print()
    parquet_path = download_eval_parquet()
    if parquet_path:
        df, protocol_path = extract_protocol_from_parquet(parquet_path, output_dir)
        summary = {
            "dataset": "ASVspoof 2019 LA",
            "split": "evaluation",
            "source_repo": DATASET_REPO,
            "total_samples": len(df),
            "bonafide": int((df["key"] == 0).sum() if df["key"].dtype in (int, float) else (df["key"] == "bonafide").sum()),
            "attack_types": int(df["system_id"].nunique()),
            "parquet_path": parquet_path,
            "protocol_path": str(protocol_path),
        }
        with open(output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {output_dir / 'dataset_summary.json'}")
    else:
        print("ERROR: Failed to download evaluation parquet.")
        print("\nAlternative: Manually download from Hugging Face:")
        print(f"  https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{EVAL_PARQUET}")
        print(f"  Save to: {output_dir / 'eval.parquet'}")


if __name__ == "__main__":
    main()
