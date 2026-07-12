# Cross-Dataset Generalization Evaluation

Evaluates the FoR-trained model on the **ASVspoof 2019 LA** benchmark to measure **out-of-distribution generalization**. No fine-tuning — pure inference.

## Pipeline

```
Step 1: download_asvspoof.py   — Download & extract ASVspoof 2019 LA
Step 2: run_inference.py       — Run model inference (no training)
Step 3: compute_metrics.py     — Accuracy, ROC-AUC, EER, FAR, FRR, plots
Step 4: error_analysis.py      — FP/FN breakdown, hardest attacks
Step 5: generalization_report.py — Honest report analyzing transfer
```

Or run all at once:

```bash
python run_all.py
```

## Output

| Path | Contents |
|---|---|
| `results/asvspoof_predictions.csv` | Per-sample predictions with probabilities |
| `results/asvspoof_metrics.json` | All computed metrics |
| `results/asvspoof_errors.csv` | Error cases with details |
| `results/attack_type_metrics.csv` | Performance by attack type |
| `results/attack_type_error_rates.csv` | Error rates by attack type |
| `results/error_summary.json` | Error analysis summary |
| `figures/confusion_matrix.png` | Confusion matrix |
| `figures/roc_curve.png` | ROC curve with EER |
| `figures/pr_curve.png` | Precision-recall curve |
| `figures/score_distribution.png` | Confidence histogram by class |
| `figures/attack_type_accuracy.png` | Accuracy per system ID |
| `figures/error_confidence_distribution.png` | Error confidence distribution |

## Dataset Download

The script downloads ASVspoof 2019 LA (~15.7 GB) from:
https://datashare.is.ed.ac.uk/handle/10283/3336

## Important

This is a **cross-dataset generalization** test, **not** cross-validation. The model was trained exclusively on **Fake-or-Real (FoR)**. Performance on ASVspoof will likely be lower due to domain shift — this is expected and scientifically valid to report.
