"""Stage 1.5 fixed-metric rerun for the best Stage 1 hybrid Deep Sets candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CV_OUTPUTS_DIR,
    OBJECTIVE_ALIGNMENT_FIXED_METRIC_REPORT_OUTPUT_PATH,
    OBJECTIVE_ALIGNMENT_FIXED_METRIC_RESULTS_OUTPUT_PATH,
    OBJECTIVE_ALIGNMENT_RESULTS_OUTPUT_PATH,
    REPORTS_DIR,
)
from src.eval.metrics import PLATFORM_TARGET_SCALES
from src.models.train_baselines import OXIDATION_TARGET, TARGET_COLUMNS, VISCOSITY_TARGET
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    evaluate_single_deep_sets_configuration,
    get_target_strategy_by_name,
    load_deep_sets_data,
)


BEST_EXPERIMENT_NAME = "aligned__raw__huber__fixed_platform_checkpoint"
REFERENCE_EXPERIMENT_NAME = "reference__raw__mse__combined_checkpoint"
PREVIOUS_STAGE1_RESULTS_PATH = OBJECTIVE_ALIGNMENT_RESULTS_OUTPUT_PATH


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    return parser.parse_args()


def _fixed_platform_score_from_maes(viscosity_mae: float, oxidation_mae: float) -> float:
    return 0.5 * (
        viscosity_mae / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
        + oxidation_mae / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
    )


def _summarize_rerun(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    row = {
        "experiment_name": BEST_EXPERIMENT_NAME,
        "result_source": "rerun_stage1_5",
        "model_name": str(fold_metrics["model_name"].iloc[0]),
        "target_strategy": str(fold_metrics["target_strategy"].iloc[0]),
        "loss_name": str(fold_metrics["loss_name"].iloc[0]),
        "checkpoint_metric": str(fold_metrics["checkpoint_metric"].iloc[0]),
        "combined_score__mean": float(fold_metrics["combined_score"].mean()),
        "combined_score__std": float(fold_metrics["combined_score"].std(ddof=1)),
        "platform_score__mean": float(fold_metrics["platform_score"].mean()),
        "platform_score__std": float(fold_metrics["platform_score"].std(ddof=1)),
        "platform_proxy_score__mean": float(fold_metrics["platform_proxy_score"].mean()),
        "platform_proxy_score__std": float(fold_metrics["platform_proxy_score"].std(ddof=1)),
        f"{VISCOSITY_TARGET}__rmse__mean": float(fold_metrics[f"{VISCOSITY_TARGET}__rmse"].mean()),
        f"{OXIDATION_TARGET}__rmse__mean": float(fold_metrics[f"{OXIDATION_TARGET}__rmse"].mean()),
        f"{VISCOSITY_TARGET}__mae__mean": float(fold_metrics[f"{VISCOSITY_TARGET}__mae"].mean()),
        f"{OXIDATION_TARGET}__mae__mean": float(fold_metrics[f"{OXIDATION_TARGET}__mae"].mean()),
        f"{VISCOSITY_TARGET}__platform_nmae__mean": float(fold_metrics[f"{VISCOSITY_TARGET}__platform_nmae"].mean()),
        f"{OXIDATION_TARGET}__platform_nmae__mean": float(fold_metrics[f"{OXIDATION_TARGET}__platform_nmae"].mean()),
        f"{VISCOSITY_TARGET}__platform_nmae_iqr__mean": float(
            fold_metrics[f"{VISCOSITY_TARGET}__platform_nmae_iqr"].mean()
        ),
        f"{OXIDATION_TARGET}__platform_nmae_iqr__mean": float(
            fold_metrics[f"{OXIDATION_TARGET}__platform_nmae_iqr"].mean()
        ),
        "best_epoch__mean": float(fold_metrics["best_epoch"].mean()),
        "best_inner_cv_score__mean": float(fold_metrics["best_inner_cv_score"].mean()),
    }
    return pd.DataFrame.from_records([row])


def _load_reference_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prior Stage 1 results: {path}")

    frame = pd.read_csv(path)
    reference = frame.loc[frame["experiment_name"] == REFERENCE_EXPERIMENT_NAME].copy()
    if reference.empty:
        raise ValueError(f"Could not find `{REFERENCE_EXPERIMENT_NAME}` in {path}")

    reference["result_source"] = "saved_stage1_reference_summary"
    reference["loss_name"] = "mse_mse"
    reference["checkpoint_metric"] = "combined_score"
    reference["platform_score__mean"] = reference.apply(
        lambda row: _fixed_platform_score_from_maes(
            viscosity_mae=float(row[f"{VISCOSITY_TARGET}__mae__mean"]),
            oxidation_mae=float(row[f"{OXIDATION_TARGET}__mae__mean"]),
        ),
        axis=1,
    )
    reference["platform_score__std"] = np.nan
    reference["best_inner_cv_score__mean"] = float(reference["best_inner_cv_score"].iloc[0])
    return reference[
        [
            "experiment_name",
            "result_source",
            "model_name",
            "target_strategy",
            "loss_name",
            "checkpoint_metric",
            "combined_score__mean",
            "combined_score__std",
            "platform_score__mean",
            "platform_score__std",
            "platform_proxy_score__mean",
            "platform_proxy_score__std",
            f"{VISCOSITY_TARGET}__rmse__mean",
            f"{OXIDATION_TARGET}__rmse__mean",
            f"{VISCOSITY_TARGET}__mae__mean",
            f"{OXIDATION_TARGET}__mae__mean",
            f"{VISCOSITY_TARGET}__platform_nmae_iqr__mean",
            f"{OXIDATION_TARGET}__platform_nmae_iqr__mean",
            "best_epoch__mean",
            "best_inner_cv_score__mean",
        ]
    ].reset_index(drop=True)


def build_report(results: pd.DataFrame) -> str:
    rerun = results.loc[results["result_source"] == "rerun_stage1_5"].iloc[0]
    reference = results.loc[results["result_source"] == "saved_stage1_reference_summary"].iloc[0]
    improvement = float(reference["platform_score__mean"] - rerun["platform_score__mean"])
    relative_improvement = improvement / float(reference["platform_score__mean"]) if reference["platform_score__mean"] else 0.0
    proceed_to_stage2 = improvement > 0.01 and relative_improvement > 0.01

    lines = [
        "# Objective Alignment Fixed Metric Report",
        "",
        "## Stage 1.5 Scope",
        "- Reran only the best Stage 1 candidate: `hybrid_deep_sets_v2_family_only` with raw targets.",
        "- Kept Huber-style training enabled for both targets.",
        "- Kept prior RMSE and proxy metrics for diagnostics, but selected checkpoints with the fixed platform metric.",
        (
            f"- Fixed platform score: `0.5 * (viscosity_MAE / {PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]:.2f} + "
            f"oxidation_MAE / {PLATFORM_TARGET_SCALES[OXIDATION_TARGET]:.2f})`"
        ),
        "",
        "## Comparison",
        "",
        "```text",
        results[
            [
                "experiment_name",
                "result_source",
                "target_strategy",
                "loss_name",
                "checkpoint_metric",
                "platform_score__mean",
                "combined_score__mean",
                "platform_proxy_score__mean",
                f"{VISCOSITY_TARGET}__mae__mean",
                f"{OXIDATION_TARGET}__mae__mean",
                "best_epoch__mean",
            ]
        ].sort_values("platform_score__mean").to_string(index=False),
        "```",
        "",
        "## Answers",
        f"- Does the Stage 1 improvement still hold under the fixed platform metric? `{'yes' if improvement > 0 else 'no'}`",
        (
            f"- What is the fixed-metric result for the best candidate? "
            f"`{rerun['platform_score__mean']:.6f}` mean platform score "
            f"(viscosity MAE `{rerun[f'{VISCOSITY_TARGET}__mae__mean']:.4f}`, "
            f"oxidation MAE `{rerun[f'{OXIDATION_TARGET}__mae__mean']:.4f}`)"
        ),
        f"- Fixed-metric delta vs the saved Stage 1 reference: `{improvement:+.6f}`",
        f"- Relative fixed-metric improvement vs the saved Stage 1 reference: `{relative_improvement:+.2%}`",
        f"- Is the result strong enough to proceed to GP Stage 2? `{'yes' if proceed_to_stage2 else 'no'}`",
        "",
        "## Recommendation",
        (
            "- Proceed to GP Stage 2 because the fixed-metric gain remains material."
            if proceed_to_stage2
            else "- Do not proceed to GP Stage 2 yet because the fixed-metric gain is not material."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared_data = load_deep_sets_data()
    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_metric="platform_score",
    )
    variant = HybridVariant(name="hybrid_deep_sets_v2_family_only", use_component_embedding=False, use_tabular_branch=True)
    target_strategy = get_target_strategy_by_name("raw")
    loss_config = LossConfig(
        name="huber_huber",
        use_robust_viscosity_loss=True,
        use_robust_oxidation_loss=True,
        viscosity_delta=1.0,
        oxidation_delta=0.75,
    )

    artifacts = evaluate_single_deep_sets_configuration(
        prepared_data=prepared_data,
        config=config,
        variant=variant,
        target_strategy=target_strategy,
        outer_splits=args.outer_splits,
        seed=42,
        loss_config=loss_config,
        extra_metadata={
            "experiment_name": BEST_EXPERIMENT_NAME,
            "checkpoint_metric": config.checkpoint_metric,
            "loss_name": loss_config.name,
        },
    )
    rerun_summary = _summarize_rerun(artifacts.fold_metrics)
    reference_summary = _load_reference_summary(PREVIOUS_STAGE1_RESULTS_PATH)
    results = pd.concat([rerun_summary, reference_summary], ignore_index=True).sort_values(
        "platform_score__mean"
    ).reset_index(drop=True)

    _write_csv(results, OBJECTIVE_ALIGNMENT_FIXED_METRIC_RESULTS_OUTPUT_PATH)
    _write_text(build_report(results), OBJECTIVE_ALIGNMENT_FIXED_METRIC_REPORT_OUTPUT_PATH)
    print(f"objective_alignment_fixed_metric_results: {OBJECTIVE_ALIGNMENT_FIXED_METRIC_RESULTS_OUTPUT_PATH}")
    print(f"objective_alignment_fixed_metric_report: {OBJECTIVE_ALIGNMENT_FIXED_METRIC_REPORT_OUTPUT_PATH}")
    print(f"fixed_metric_best_platform_score: {rerun_summary.iloc[0]['platform_score__mean']:.6f}")


if __name__ == "__main__":
    main()
