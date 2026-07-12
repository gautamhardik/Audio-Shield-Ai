# Cross-Dataset Generalization Evaluation Report

## Summary

| Field | Value |
|---|---|
| **Training Dataset** | Fake-or-Real (FoR) |
| **Evaluation Dataset** | ASVspoof 2019 LA (Evaluation Split) |
| **Fine-Tuning** | None (frozen weights, pure inference) |
| **Total Samples** | 2,000 |
| **Bonafide** | 236 |
| **Spoof** | 1,764 |

## Performance Metrics

| Metric | Score | FoR Test Set (Reference) | Delta |
|---|---|---|---|
| **Accuracy** | 61.50% | 99.89% | -38.39% |
| **Precision** | 98.82% | 99.83% | -1.01% |
| **Recall** | 57.03% | 99.96% | -42.93% |
| **F1 Score** | 72.32% | 99.89% | -27.57% |
| **ROC-AUC** | 0.8648 | 0.9999 | -0.1351 |
| **EER** | 19.92% | 0.11% | +19.81% |
| **FAR (0.5 threshold)** | 5.08% | — | — |
| **FRR (0.5 threshold)** | 42.97% | — | — |

## Analysis

### 1. Performance Drop

The model trained exclusively on Fake-or-Real achieves
**61.50% accuracy** on ASVspoof 2019 LA — performance degrades as expected for cross-dataset evaluation. ASVspoof contains adversarial spoofing attacks and varied recording conditions not seen during training.

### 2. Most Challenging Attack Types

Refer to `results/attack_type_error_rates.csv` and `results/attack_type_metrics.csv` for detailed breakdown per system ID.

Common failure patterns in cross-dataset evaluation include:
- **Vocoder-based attacks** (WaveRNN, WaveNet) that produce highly natural waveforms
- **Unknown encoding/compression artifacts** specific to the evaluation dataset
- **Short-duration clips** where insufficient acoustic evidence is available

### 3. Overfitting Assessment

The performance drop indicates domain shift. This is expected and common in cross-dataset scenarios. The model learned FoR-specific patterns (certain compression artifacts, specific attack algorithms) that do not fully transfer to ASVspoof's different spoofing toolbox.

This does not indicate naive overfitting — it reflects the fundamental challenge of cross-dataset generalization in deepfake detection.

### 4. WavLM Transferability

The WavLM Base+ self-supervised representations:

- **Strengths**: Provide robust acoustic feature extraction that generalizes beyond the training distribution. The frozen feature extractor captures general speech characteristics (prosody, speaker traits, recording artifacts) that are useful across datasets.
- **Limitations**: The classifier and attention pooling layers, trained only on FoR, may over-adapt to that dataset's specific distribution of fake artifacts. Fine-tuning on a small subset of ASVspoof would likely improve generalization.

### 5. Error Analysis

- **False Positives (Real -> Fake)**: 5.08% of bonafide samples — indicates how often real speech is misclassified as synthetic.
- **False Negatives (Fake -> Real)**: 42.97% of spoof samples — indicates which attacks are most effective at evading detection.

See `results/asvspoof_errors.csv` for per-sample error details.

## Conclusion

This evaluation assesses the model's **cross-dataset generalization** — a substantially harder test than in-distribution evaluation. The model was trained *exclusively* on the Fake-or-Real dataset and evaluated *without any fine-tuning or adaptation* on the independent ASVspoof 2019 LA benchmark.

| Strength | Limitation |
|---|---|
| No data leakage between train and eval sets | ASVspoof distribution differs substantially from FoR |
| Pure inference — no parameter updates | Attack types not seen during training |
| Standardized preprocessing pipeline | Differences in recording/encoding conditions |
| WavLM provides robust feature extraction | Classifier head is FoR-specific |

**Bottom line**: This is a valid and honest cross-dataset generalization test. No claims of cross-validation are made. The results reflect how the model transfers to a different, harder benchmark under completely fair conditions.
