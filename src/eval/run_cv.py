"""CLI entrypoint for reproducible grouped-CV baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    BASELINE_CV_RESULTS_OUTPUT_PATH,
    BASELINE_FOLD_METRICS_OUTPUT_PATH,
    BASELINE_REPORT_OUTPUT_PATH,
    CV_OUTPUTS_DIR,
    REPORTS_DIR,
)
from src.models.train_baselines import (
    build_model_specs,
    build_target_strategies,
    load_baseline_training_data,
    load_test_feature_table,
    run_baseline_cv,
)


def _write_csv(frame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the baseline CV runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--include-mlp", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run baseline CV and persist all requested artifacts."""

    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared_data = load_baseline_training_data()
    test_features = load_test_feature_table()
    if prepared_data.X.columns.tolist() != test_features.drop(columns=["scenario_id"]).columns.tolist():
        raise ValueError("Train and test feature schemas do not match.")

    summary_results, fold_metrics, report_markdown = run_baseline_cv(
        prepared_data=prepared_data,
        model_specs=build_model_specs(include_mlp=args.include_mlp),
        target_strategies=build_target_strategies(),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )

    _write_csv(summary_results, BASELINE_CV_RESULTS_OUTPUT_PATH)
    _write_csv(fold_metrics, BASELINE_FOLD_METRICS_OUTPUT_PATH)
    _write_text(report_markdown, BASELINE_REPORT_OUTPUT_PATH)

    print(f"baseline_cv_results: {BASELINE_CV_RESULTS_OUTPUT_PATH}")
    print(f"baseline_fold_metrics: {BASELINE_FOLD_METRICS_OUTPUT_PATH}")
    print(f"baseline_report: {BASELINE_REPORT_OUTPUT_PATH}")
    print(
        "best_baseline: "
        f"{summary_results.iloc[0]['model_name']} / {summary_results.iloc[0]['target_strategy']}"
    )


if __name__ == "__main__":
    main()
