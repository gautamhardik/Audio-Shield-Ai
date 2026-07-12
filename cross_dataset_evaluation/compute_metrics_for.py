import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, auc
)
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

RESULTS_DIR = Path(__file__).parent / "results_for"
FIGURES_DIR = RESULTS_DIR / "figures"

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return float(eer), float(thresholds[np.nanargmin(np.abs(fnr - fpr))])

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / "for_predictions.csv"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found"); return

    df = pd.read_csv(pred_path)
    y_true, y_pred, y_prob = df["true_label"].values, df["prediction"].values, df["fake_probability"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_prob)
    eer, eer_thresh = compute_eer(y_true, y_prob)

    total = len(df)
    correct = int((y_true == y_pred).sum())
    errors = total - correct

    metrics = {
        "evaluation_type": "in-distribution_test",
        "dataset": "Fake-or-Real (FoR) - Test Split",
        "total_samples": total,
        "real_count": int((y_true == 0).sum()),
        "fake_count": int((y_true == 1).sum()),
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
    }

    print("=" * 60)
    print("FoR TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(f"\nTotal samples:          {total}")
    print(f"  Real:                 {(y_true == 0).sum()}")
    print(f"  Fake:                 {(y_true == 1).sum()}")
    print(f"Correct predictions:    {correct}")
    print(f"Incorrect predictions:  {errors}")
    print(f"\nAccuracy:   {accuracy*100:.2f}%")
    print(f"Precision:  {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:     {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:   {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:    {auc_score:.4f}")
    print(f"EER:        {eer*100:.2f}%")

    with open(RESULTS_DIR / "for_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {RESULTS_DIR / 'for_metrics.json'}")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix — FoR Test Set", fontsize=14, fontweight="bold", pad=20)
    for text in ax.texts:
        text.set_fontsize(14); text.set_fontweight("bold")
    ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'confusion_matrix.png'}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color=COLORS[0], lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS[0])
    eer_idx = np.nanargmin(np.abs(np.array(fpr) - (1 - np.array(tpr))))
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color=COLORS[2], s=120, zorder=5,
               label=f"EER = {eer*100:.2f}%", edgecolors="black", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — FoR Test Set", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'roc_curve.png'}")

    # PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall_vals, precision_vals, color=COLORS[1], lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color=COLORS[1])
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color="gray", lw=1.5, linestyle="--", alpha=0.7, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — FoR Test Set", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'pr_curve.png'}")

    # Score Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label_val, color, label in zip(axes, [0, 1], ["#2ECC71", "#E74C3C"], ["Real", "Fake"]):
        subset = df[df["true_label"] == label_val]["fake_probability"]
        ax.hist(subset, bins=50, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.6, label="Threshold (0.5)")
        ax.set_xlabel("Fake Probability", fontsize=11); ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle("Confidence Score Distribution by Class — FoR", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'score_distribution.png'}")

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {FIGURES_DIR}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
