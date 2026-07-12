"""
Orchestrator: Run the full cross-dataset evaluation pipeline.

Usage:
    python run_all.py                    # Full pipeline
    python run_all.py --skip-download    # Skip dataset download (if already downloaded)
"""
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run full cross-dataset evaluation pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip ASVspoof dataset download")
    parser.add_argument("--asvspoof-dir", type=str, default=None,
                        help="Path to ASVspoof 2019 LA dataset")
    args = parser.parse_args()

    base = Path(__file__).parent

    if not args.skip_download:
        print("\n[Step 1/5] Downloading ASVspoof 2019 LA dataset...")
        from download_asvspoof import main as download
        download()

    print("\n[Step 2/5] Running inference...")
    from run_inference import main as inference
    inference_args = ["--output-dir", str(base / "results")]
    if args.asvspoof_dir:
        inference_args += ["--asvspoof-dir", args.asvspoof_dir]
    sys.argv = [sys.argv[0]] + inference_args
    inference()

    print("\n[Step 3/5] Computing metrics & generating visualizations...")
    from compute_metrics import main as metrics
    sys.argv = [sys.argv[0], "--predictions", str(base / "results" / "asvspoof_predictions.csv"),
                "--output-dir", str(base / "figures"),
                "--results-dir", str(base / "results")]
    metrics()

    print("\n[Step 4/5] Running error analysis...")
    from error_analysis import main as errors
    sys.argv = [sys.argv[0], "--predictions", str(base / "results" / "asvspoof_predictions.csv"),
                "--output-dir", str(base / "results"),
                "--figures-dir", str(base / "figures")]
    errors()

    print("\n[Step 5/5] Generating generalization report...")
    from generalization_report import main as report
    sys.argv = [sys.argv[0], "--results-dir", str(base / "results"),
                "--figures-dir", str(base / "figures"),
                "--output", str(base / "GENERALIZATION_REPORT.md")]
    report()

    print("\n" + "=" * 60)
    print("CROSS-DATASET EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults:    {base / 'results'}")
    print(f"Figures:    {base / 'figures'}")
    print(f"Report:     {base / 'GENERALIZATION_REPORT.md}")
    print(f"\nNext step: Add the 'Cross-Dataset Generalization Evaluation'")
    print(f"           section to your README.md")


if __name__ == "__main__":
    main()
