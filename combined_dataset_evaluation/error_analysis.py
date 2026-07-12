import os
import json
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Error analysis for ASVspoof evaluation")
    parser.add_argument("--predictions", type=str,
                        default=str(Path(__file__).parent / "results" / "asvspoof_predictions.csv"),
                        help="Path to predictions CSV")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "results"),
                        help="Output directory")
    parser.add_argument("--figures-dir", type=str,
                        default=str(Path(__file__).parent / "figures"),
                        help="Figures directory")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        print(f"ERROR: Predictions file not found at {pred_path}")
        return

    df = pd.read_csv(pred_path)
    print("=" * 60)
    print("ERROR ANALYSIS — ASVspoof 2019 LA Cross-Dataset Evaluation")
    print("=" * 60)

    df["is_error"] = df["true_label"] != df["prediction"]
    df["error_type"] = "correct"
    df.loc[(df["true_label"] == 0) & (df["prediction"] == 1), "error_type"] = "false_positive"
    df.loc[(df["true_label"] == 1) & (df["prediction"] == 0), "error_type"] = "false_negative"

    errors_df = df[df["is_error"]].copy()
    total_errors = len(errors_df)
    total_samples = len(df)
    fp_count = (df["error_type"] == "false_positive").sum()
    fn_count = (df["error_type"] == "false_negative").sum()

    print(f"\nTotal samples:     {total_samples}")
    print(f"Total errors:      {total_errors} ({total_errors/total_samples*100:.2f}%)")
    print(f"False Positives:   {fp_count} ({fp_count/total_errors*100:.1f}% of errors, {fp_count/total_samples*100:.2f}% of total)")
    print(f"False Negatives:   {fn_count} ({fn_count/total_errors*100:.1f}% of errors, {fn_count/total_samples*100:.2f}% of total)")

    errors_sorted = errors_df.sort_values("fake_probability", ascending=False)
    errors_csv = out_dir / "asvspoof_errors.csv"
    errors_sorted.to_csv(errors_csv, index=False)
    print(f"\nError details saved to {errors_csv}")

    print("\n--- Top 10 Most Confident False Positives ---")
    fp = errors_df[errors_df["error_type"] == "false_positive"].sort_values("confidence", ascending=False)
    if len(fp) > 0:
        for _, row in fp.head(10).iterrows():
            print(f"  {row['filename']:40s} conf={row['confidence']:.4f} fake_prob={row['fake_probability']:.4f} sys={row['system_id']}")
    else:
        print("  (none)")

    print("\n--- Top 10 Most Confident False Negatives ---")
    fn = errors_df[errors_df["error_type"] == "false_negative"].sort_values("confidence", ascending=False)
    if len(fn) > 0:
        for _, row in fn.head(10).iterrows():
            print(f"  {row['filename']:40s} conf={row['confidence']:.4f} fake_prob={row['fake_probability']:.4f} sys={row['system_id']}")
    else:
        print("  (none)")

    if len(errors_df) > 0:
        attack_error_rates = (
            errors_df.groupby("system_id")
            .agg(
                errors=("filename", "count"),
                fp=("error_type", lambda x: (x == "false_positive").sum()),
                fn=("error_type", lambda x: (x == "false_negative").sum()),
            )
        )
        total_by_attack = (
            df.groupby("system_id")
            .agg(total=("filename", "count"))
        )
        attack_stats = total_by_attack.join(attack_error_rates).fillna(0)
        attack_stats["error_rate"] = attack_stats["errors"] / attack_stats["total"]
        attack_stats = attack_stats.sort_values("error_rate", ascending=False)
        attack_stats_csv = out_dir / "attack_type_error_rates.csv"
        attack_stats.to_csv(attack_stats_csv)
        print(f"\nAttack type error rates saved to {attack_stats_csv}")
        print("\n--- Hardest Attack Types (Highest Error Rate) ---")
        for sys_id, row in attack_stats.head(15).iterrows():
            print(f"  {sys_id:20s} errors={int(row['errors']):>5}/{int(row['total']):>5} ({row['error_rate']*100:>5.1f}%) fp={int(row['fp']):>4} fn={int(row['fn']):>4}")

    if len(errors_df) > 0:
        errors_df["error_type_label"] = errors_df["error_type"].map({
            "false_positive": "False Positive (Real→Fake)",
            "false_negative": "False Negative (Fake→Real)",
        })
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {"False Positive (Real→Fake)": "#E74C3C", "False Negative (Fake→Real)": "#3498DB"}
        for error_type in errors_df["error_type_label"].unique():
            subset = errors_df[errors_df["error_type_label"] == error_type]
            bins = np.linspace(0, 1, 50)
            ax.hist(
                subset["confidence"],
                bins=bins,
                alpha=0.7,
                label=f"{error_type} (n={len(subset)})",
                color=colors.get(error_type, "#95A5A6"),
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xlabel("Confidence Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Error Confidence Distribution", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = fig_dir / "error_confidence_distribution.png"
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {hist_path}")

    error_summary = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": float(total_errors / total_samples),
        "false_positives": int(fp_count),
        "false_negatives": int(fn_count),
        "fp_rate": float(fp_count / total_samples),
        "fn_rate": float(fn_count / total_samples),
        "hardest_attack_types": attack_stats.head(10).to_dict() if len(errors_df) > 0 else {},
    }
    summary_path = out_dir / "error_summary.json"
    with open(summary_path, "w") as f:
        json.dump(error_summary, f, indent=2, default=str)
    print(f"Error summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
