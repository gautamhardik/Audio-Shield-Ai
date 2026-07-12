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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    auc,
)

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return float(eer), float(eer_threshold)


def compute_far_frr(y_true, y_pred, y_prob, threshold=0.5):
    y_pred_custom = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_custom)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return float(far), float(frr)


def generate_confusion_matrix(y_true, y_pred, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Bonafide (Real)", "Spoof (Fake)"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix — ASVspoof 2019 LA", fontsize=14, fontweight="bold", pad=20)
    for text in ax.texts:
        text.set_fontsize(14)
        text.set_fontweight("bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return cm


def generate_roc_curve(y_true, y_prob, output_path, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        fpr, tpr, color=COLORS[0], lw=2.5,
        label=f"ROC Curve (AUC = {auc_score:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", alpha=0.7, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS[0])
    eer, eer_thresh = compute_eer(y_true, y_prob)
    eer_idx = np.nanargmin(np.abs(np.array(fpr) - (1 - np.array(tpr))))
    ax.scatter(
        fpr[eer_idx], tpr[eer_idx], color=COLORS[2], s=120, zorder=5,
        label=f"EER = {eer:.4f} ({eer*100:.2f}%)", edgecolors="black", linewidth=1
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — ASVspoof 2019 LA", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return fpr, tpr


def generate_pr_curve(y_true, y_prob, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(
        recall, precision, color=COLORS[1], lw=2.5,
        label=f"PR Curve (AUC = {pr_auc:.4f})"
    )
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS[1])
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color="gray", lw=1.5, linestyle="--", alpha=0.7,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — ASVspoof 2019 LA", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return pr_auc


def generate_score_distribution(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2ECC71", "#E74C3C"]
    labels = ["Bonafide (Real)", "Spoof (Fake)"]
    for ax, label_val, color, label in zip(
        axes, [0, 1], colors, labels
    ):
        subset = df[df["true_label"] == label_val]["fake_probability"]
        ax.hist(subset, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(
            x=0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.6,
            label="Decision Threshold (0.5)"
        )
        ax.set_xlabel("Fake Probability", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Confidence Score Distribution by Class", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_attack_type_metrics(df, output_path):
    attack_metrics = []
    for system_id in df["system_id"].unique():
        subset = df[df["system_id"] == system_id]
        y_true = subset["true_label"]
        y_pred = subset["prediction"]
        if len(y_true) == 0:
            continue
        is_spoof = (y_true == 1).all()
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        attack_metrics.append({
            "system_id": system_id,
            "attack_type": "spoof" if is_spoof else "bonafide",
            "total": total,
            "correct": int(correct),
            "accuracy": float(correct / total) if total > 0 else 0.0,
            "mean_confidence": float(subset["confidence"].mean()),
            "mean_fake_prob": float(subset["fake_probability"].mean()),
        })
    attack_df = pd.DataFrame(attack_metrics)
    attack_df = attack_df.sort_values("accuracy")
    attack_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    attack_df_sorted = attack_df.sort_values("accuracy", ascending=True).head(20)
    fig, ax = plt.subplots(figsize=(12, max(6, len(attack_df_sorted) * 0.35)))
    colors_bar = ["#E74C3C" if acc < 0.8 else "#F39C12" if acc < 0.95 else "#2ECC71" for acc in attack_df_sorted["accuracy"]]
    bars = ax.barh(range(len(attack_df_sorted)), attack_df_sorted["accuracy"].values, color=colors_bar, edgecolor="white")
    ax.set_yticks(range(len(attack_df_sorted)))
    ax.set_yticklabels(attack_df_sorted["system_id"].values, fontsize=9)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Attack Type (System ID)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, acc in zip(bars, attack_df_sorted["accuracy"].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(output_path.parent / "attack_type_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.parent / 'attack_type_accuracy.png'}")
    return attack_df


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--predictions", type=str,
                        default=str(Path(__file__).parent / "results" / "asvspoof_predictions.csv"),
                        help="Path to predictions CSV")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "figures"),
                        help="Output directory for figures")
    parser.add_argument("--results-dir", type=str,
                        default=str(Path(__file__).parent / "results"),
                        help="Results directory for JSON output")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    fig_dir = Path(args.output_dir)
    res_dir = Path(args.results_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        print(f"ERROR: Predictions file not found at {pred_path}")
        print("Run run_inference.py first.")
        return

    df = pd.read_csv(pred_path)
    y_true = df["true_label"].values
    y_pred = df["prediction"].values
    y_prob = df["fake_probability"].values

    print("=" * 60)
    print("CROSS-DATASET EVALUATION METRICS")
    print("FoR-trained model evaluated on ASVspoof 2019 LA")
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_prob)
    eer, eer_thresh = compute_eer(y_true, y_prob)
    far, frr = compute_far_frr(y_true, y_pred, y_prob, threshold=0.5)

    total = len(df)
    correct = int((y_true == y_pred).sum())
    errors = total - correct

    metrics = {
        "evaluation_type": "cross-dataset_generalization",
        "training_dataset": "Fake-or-Real (FoR)",
        "evaluation_dataset": "ASVspoof 2019 LA (Evaluation Split)",
        "fine_tuning": False,
        "total_samples": total,
        "bonafide_count": int((y_true == 0).sum()),
        "spoof_count": int((y_true == 1).sum()),
        "correct_predictions": correct,
        "incorrect_predictions": errors,
        "accuracy": float(accuracy),
        "accuracy_pct": float(accuracy * 100),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc_score),
        "eer": float(eer),
        "eer_threshold": float(eer_thresh),
        "far_at_0.5": float(far),
        "frr_at_0.5": float(frr),
        "threshold_used": 0.5,
    }

    print(f"\nTotal samples evaluated: {total:>6}")
    print(f"  Bonafide (real):        {(y_true == 0).sum():>6}")
    print(f"  Spoof (fake):           {(y_true == 1).sum():>6}")
    print(f"Correct predictions:      {correct:>6}")
    print(f"Incorrect predictions:    {errors:>6}")
    print(f"\n{'Metric':<20} {'Score':<15} {'Comparison (FoR Test)':<20}")
    print("-" * 55)
    print(f"{'Accuracy':<20} {accuracy*100:>7.2f}%{'':8} {'99.89%':>10}")
    print(f"{'Precision':<20} {precision:.4f} ({precision*100:.2f}%){'':4} {'99.83%':>10}")
    print(f"{'Recall':<20} {recall:.4f} ({recall*100:.2f}%){'':4} {'99.96%':>10}")
    print(f"{'F1 Score':<20} {f1:.4f} ({f1*100:.2f}%){'':4} {'99.89%':>10}")
    print(f"{'ROC-AUC':<20} {auc_score:.4f}{'':12} {'0.9999':>10}")
    print(f"{'EER':<20} {eer*100:.2f}%{'':12} {'0.11%':>10}")
    print(f"{'FAR (thresh=0.5)':<20} {far*100:.2f}%")
    print(f"{'FRR (thresh=0.5)':<20} {frr*100:.2f}%")

    metrics_path = res_dir / "asvspoof_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print("\nGenerating visualizations...")
    cm = generate_confusion_matrix(y_true, y_pred, fig_dir / "confusion_matrix.png")
    generate_roc_curve(y_true, y_prob, fig_dir / "roc_curve.png", auc_score)
    generate_pr_curve(y_true, y_prob, fig_dir / "pr_curve.png")
    generate_score_distribution(df, fig_dir / "score_distribution.png")
    attack_df = generate_attack_type_metrics(df, res_dir / "attack_type_metrics.csv")

    class_report = classification_report(
        y_true, y_pred,
        target_names=["Bonafide (Real)", "Spoof (Fake)"],
        output_dict=True, zero_division=0
    )
    report_path = res_dir / "classification_report.csv"
    pd.DataFrame(class_report).transpose().to_csv(report_path)
    print(f"  Saved: {report_path}")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Bonafide", "Actual Spoof"],
        columns=["Predicted Bonafide", "Predicted Spoof"],
    )
    cm_df.to_csv(res_dir / "confusion_matrix.csv")
    print(f"  Saved: {res_dir / 'confusion_matrix.csv'}")

    print(f"\n{'=' * 60}")
    print("All visualizations saved to:", fig_dir)
    print("All metrics/results saved to:", res_dir)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
