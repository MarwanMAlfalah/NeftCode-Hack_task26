"""CLI entrypoint for reproducible grouped-CV Hybrid Deep Sets v2 evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    CV_OUTPUTS_DIR,
    DEEP_SETS_V2_CV_RESULTS_OUTPUT_PATH,
    DEEP_SETS_V2_FOLD_METRICS_OUTPUT_PATH,
    DEEP_SETS_V2_REPORT_OUTPUT_PATH,
    REPORTS_DIR,
)
from src.models.train_deep_sets import DeepSetsConfig, load_deep_sets_data, run_deep_sets_cv


def _write_csv(frame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Hybrid Deep Sets v2 CV runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run Hybrid Deep Sets v2 grouped CV and persist the requested artifacts."""

    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared_data = load_deep_sets_data()
    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )
    summary_results, fold_metrics, report_markdown = run_deep_sets_cv(
        prepared_data=prepared_data,
        config=config,
        outer_splits=args.outer_splits,
    )

    _write_csv(summary_results, DEEP_SETS_V2_CV_RESULTS_OUTPUT_PATH)
    _write_csv(fold_metrics, DEEP_SETS_V2_FOLD_METRICS_OUTPUT_PATH)
    _write_text(report_markdown, DEEP_SETS_V2_REPORT_OUTPUT_PATH)

    print(f"deep_sets_v2_cv_results: {DEEP_SETS_V2_CV_RESULTS_OUTPUT_PATH}")
    print(f"deep_sets_v2_fold_metrics: {DEEP_SETS_V2_FOLD_METRICS_OUTPUT_PATH}")
    print(f"deep_sets_v2_report: {DEEP_SETS_V2_REPORT_OUTPUT_PATH}")
    print(
        "best_hybrid_deep_sets_v2: "
        f"{summary_results.iloc[0]['model_name']} / {summary_results.iloc[0]['target_strategy']}"
    )


if __name__ == "__main__":
    main()
