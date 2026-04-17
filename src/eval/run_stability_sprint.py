"""Short stabilization sprint for the frozen hybrid Deep Sets v2 family-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import (
    CV_OUTPUTS_DIR,
    REPORTS_DIR,
    STABILITY_SPRINT_REPORT_OUTPUT_PATH,
    STABILITY_SPRINT_RESULTS_OUTPUT_PATH,
)
from src.eval.metrics import compute_target_scales, evaluate_regression_predictions
from src.models.train_baselines import TARGET_COLUMNS, VISCOSITY_TARGET, OXIDATION_TARGET
from src.models.train_deep_sets import (
    DeepSetsConfig,
    build_stability_sprint_experiments,
    fit_deep_sets_model,
    get_hybrid_variant_by_name,
    get_target_strategy_by_name,
    load_deep_sets_data,
    predict_deep_sets,
)


DEFAULT_SEEDS = [42, 52, 62, 72, 82]
METRIC_MAP = {
    "combined_score": "combined_score",
    "viscosity_rmse": f"{VISCOSITY_TARGET}__rmse",
    "oxidation_rmse": f"{OXIDATION_TARGET}__rmse",
}
STABILITY_REFERENCE_PATH = Path("outputs/cv/final_model_selection_results.csv")


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the stability sprint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Repeated grouped-CV seeds.",
    )
    return parser.parse_args()


def summarize_stability_sprint(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics into seed-level and fold-level stability summaries."""

    seed_level = (
        fold_metrics.groupby(
            ["experiment_name", "model_name", "target_strategy", "loss_name", "random_seed"],
            dropna=False,
        )[list(METRIC_MAP.values())]
        .mean()
        .reset_index()
    )

    records: list[dict[str, object]] = []
    grouped = fold_metrics.groupby(
        ["experiment_name", "model_name", "target_strategy", "loss_name"],
        dropna=False,
    )
    for (experiment_name, model_name, target_strategy, loss_name), group in grouped:
        seed_group = seed_level.loc[
            (seed_level["experiment_name"] == experiment_name)
            & (seed_level["model_name"] == model_name)
            & (seed_level["target_strategy"] == target_strategy)
            & (seed_level["loss_name"] == loss_name)
        ].copy()
        record: dict[str, object] = {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "target_strategy": target_strategy,
            "loss_name": loss_name,
            "n_seeds": int(seed_group["random_seed"].nunique()),
            "n_seed_fold_runs": int(len(group)),
        }
        for display_name, column_name in METRIC_MAP.items():
            seed_values = seed_group[column_name].to_numpy(dtype=float)
            fold_values = group[column_name].to_numpy(dtype=float)
            record[f"{display_name}__seed_mean"] = float(np.mean(seed_values))
            record[f"{display_name}__seed_std"] = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
            record[f"{display_name}__seed_min"] = float(np.min(seed_values))
            record[f"{display_name}__seed_max"] = float(np.max(seed_values))
            record[f"{display_name}__fold_mean"] = float(np.mean(fold_values))
            record[f"{display_name}__fold_std"] = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
            record[f"{display_name}__fold_min"] = float(np.min(fold_values))
            record[f"{display_name}__fold_max"] = float(np.max(fold_values))
        records.append(record)

    summary = pd.DataFrame.from_records(records)
    summary = summary.sort_values(
        ["combined_score__seed_mean", "viscosity_rmse__seed_mean", "oxidation_rmse__seed_mean"]
    ).reset_index(drop=True)
    summary["rank_combined_score"] = np.arange(1, len(summary) + 1)
    return summary


def _load_previous_official_reference() -> dict[str, float] | None:
    """Load the pre-sprint final-selection reference for comparison."""

    if not STABILITY_REFERENCE_PATH.exists():
        return None
    frame = pd.read_csv(STABILITY_REFERENCE_PATH)
    match = frame.loc[
        (frame["model_name"] == "hybrid_deep_sets_v2_family_only")
        & (frame["target_strategy"] == "raw")
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "combined_score__seed_mean": float(row["combined_score__seed_mean"]),
        "combined_score__seed_std": float(row["combined_score__seed_std"]),
        "viscosity_rmse__seed_mean": float(row["viscosity_rmse__seed_mean"]),
        "viscosity_rmse__seed_std": float(row["viscosity_rmse__seed_std"]),
    }


def build_stability_sprint_report(
    summary_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    seeds: list[int],
) -> str:
    """Write a concise decision-oriented stability sprint report."""

    best_row = summary_results.iloc[0]
    previous_reference = _load_previous_official_reference()
    raw_current_row = summary_results.loc[
        (summary_results["target_strategy"] == "raw")
        & (summary_results["loss_name"] == "mse")
    ].iloc[0]

    stable_enough = (
        best_row["combined_score__seed_std"] < 0.25
        and best_row["combined_score__seed_max"] < 2.0
        and best_row["viscosity_rmse__seed_max"] < 250.0
    )

    lines = [
        "# Stability Sprint Report",
        "",
        "## Evaluation Plan",
        f"- Repeated grouped CV seeds: `{seeds}`",
        "- Grouping remained strictly by `scenario_id`.",
        "- Only the frozen `hybrid_deep_sets_v2_family_only` architecture was evaluated.",
        "",
        "## Experiment Comparison",
        "",
        "```text",
        summary_results[
            [
                "rank_combined_score",
                "experiment_name",
                "target_strategy",
                "loss_name",
                "combined_score__seed_mean",
                "combined_score__seed_std",
                "combined_score__seed_min",
                "combined_score__seed_max",
                "viscosity_rmse__seed_mean",
                "viscosity_rmse__seed_std",
                "oxidation_rmse__seed_mean",
                "oxidation_rmse__seed_std",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Decision",
        (
            f"- Best stabilized variant: `{best_row['experiment_name']}` with seed-level combined score "
            f"`{best_row['combined_score__seed_mean']:.4f}` +/- `{best_row['combined_score__seed_std']:.4f}`"
        ),
        (
            f"- Stability {'improved materially' if previous_reference and best_row['combined_score__seed_std'] < previous_reference['combined_score__seed_std'] else 'did not improve materially'} "
            "relative to the pre-sprint official-candidate validation."
        ),
        (
            f"- Combined score {'improved' if previous_reference and best_row['combined_score__seed_mean'] < previous_reference['combined_score__seed_mean'] else 'degraded or lacks a prior stable reference'} "
            "relative to the pre-sprint official-candidate validation."
        ),
        (
            f"- The model is {'now stable enough to freeze' if stable_enough else 'still not stable enough to freeze'}."
        ),
        "",
        "## Mean / Std Across Seeds",
        (
            f"- Combined score mean/std/min/max: `{best_row['combined_score__seed_mean']:.4f}` / "
            f"`{best_row['combined_score__seed_std']:.4f}` / `{best_row['combined_score__seed_min']:.4f}` / "
            f"`{best_row['combined_score__seed_max']:.4f}`"
        ),
        (
            f"- Viscosity RMSE mean/std/min/max: `{best_row['viscosity_rmse__seed_mean']:.4f}` / "
            f"`{best_row['viscosity_rmse__seed_std']:.4f}` / `{best_row['viscosity_rmse__seed_min']:.4f}` / "
            f"`{best_row['viscosity_rmse__seed_max']:.4f}`"
        ),
        (
            f"- Oxidation RMSE mean/std/min/max: `{best_row['oxidation_rmse__seed_mean']:.4f}` / "
            f"`{best_row['oxidation_rmse__seed_std']:.4f}` / `{best_row['oxidation_rmse__seed_min']:.4f}` / "
            f"`{best_row['oxidation_rmse__seed_max']:.4f}`"
        ),
        "",
        "## Failure Modes",
        "- The worst runs are still dominated by extreme viscosity failures.",
        "- Robust viscosity loss can reduce sensitivity, but catastrophic folds remain the main freeze blocker if they persist.",
        "- If stability remains weak, the safer fallback is to keep the raw-loss family-only variant as the submission path while documenting instability rather than switching families.",
    ]

    if previous_reference is not None:
        lines.extend(
            [
                "",
                "## Comparison To Pre-Sprint Official Candidate",
                (
                    f"- Previous raw/MSE seed mean/std combined score: "
                    f"`{previous_reference['combined_score__seed_mean']:.4f}` / "
                    f"`{previous_reference['combined_score__seed_std']:.4f}`"
                ),
                (
                    f"- Previous raw/MSE viscosity RMSE mean/std: "
                    f"`{previous_reference['viscosity_rmse__seed_mean']:.4f}` / "
                    f"`{previous_reference['viscosity_rmse__seed_std']:.4f}`"
                ),
                (
                    f"- Current raw/MSE seed mean/std combined score: "
                    f"`{raw_current_row['combined_score__seed_mean']:.4f}` / "
                    f"`{raw_current_row['combined_score__seed_std']:.4f}`"
                ),
                (
                    f"- Current raw/MSE viscosity RMSE mean/std: "
                    f"`{raw_current_row['viscosity_rmse__seed_mean']:.4f}` / "
                    f"`{raw_current_row['viscosity_rmse__seed_std']:.4f}`"
                ),
            ]
        )

    hardest_runs = fold_metrics.sort_values(
        ["combined_score", f"{VISCOSITY_TARGET}__rmse", f"{OXIDATION_TARGET}__rmse"],
        ascending=False,
    ).head(4)
    if not hardest_runs.empty:
        lines.extend(["", "## Hardest Seed/Fold Runs"])
        for row in hardest_runs.to_dict(orient="records"):
            lines.append(
                f"- `{row['experiment_name']}` seed `{int(row['random_seed'])}` fold `{int(row['fold_index'])}`: "
                f"combined `{row['combined_score']:.4f}`, viscosity RMSE `{row[f'{VISCOSITY_TARGET}__rmse']:.4f}`, "
                f"oxidation RMSE `{row[f'{OXIDATION_TARGET}__rmse']:.4f}`"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the short stability sprint and persist results."""

    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared_data = load_deep_sets_data()
    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )
    variant = get_hybrid_variant_by_name("hybrid_deep_sets_v2_family_only")
    experiments = build_stability_sprint_experiments()

    groups = np.asarray(prepared_data.train_data.scenario_ids, dtype=object)
    all_fold_records: list[dict[str, object]] = []

    for repeated_seed in args.seeds:
        outer_cv = GroupKFold(n_splits=args.outer_splits, shuffle=True, random_state=repeated_seed)
        for experiment in experiments:
            target_strategy = get_target_strategy_by_name(experiment.target_strategy_name)
            train_targets_raw_all = np.asarray(prepared_data.train_data.targets, dtype=np.float32)
            transformed_targets_all = target_strategy.transform(train_targets_raw_all).astype(np.float32)
            transformed_data = prepared_data.train_data.with_targets(transformed_targets_all)

            for fold_index, (train_index, valid_index) in enumerate(
                outer_cv.split(np.zeros(len(groups)), groups=groups),
                start=1,
            ):
                raw_train_fold = prepared_data.train_data.subset(train_index)
                raw_valid_fold = prepared_data.train_data.subset(valid_index)
                train_data_fold = transformed_data.subset(train_index)
                valid_data_fold = transformed_data.subset(valid_index)
                y_train_raw = np.asarray(raw_train_fold.targets, dtype=np.float32)
                y_valid_raw = np.asarray(raw_valid_fold.targets, dtype=np.float32)
                target_scales = compute_target_scales(y_train_raw, TARGET_COLUMNS)

                start_time = time.perf_counter()
                fit_artifacts = fit_deep_sets_model(
                    train_data=train_data_fold,
                    groups=groups[train_index],
                    schema=prepared_data.schema,
                    config=config,
                    variant=variant,
                    target_strategy=target_strategy,
                    raw_targets=y_train_raw,
                    loss_config=experiment.loss_config,
                    seed=repeated_seed + fold_index,
                )
                fit_time = time.perf_counter() - start_time

                y_pred_transformed = predict_deep_sets(
                    raw_data=valid_data_fold,
                    schema=prepared_data.schema,
                    config=config,
                    variant=variant,
                    fit_artifacts=fit_artifacts,
                    batch_size=config.batch_size,
                )
                y_pred_raw = target_strategy.inverse_transform(y_pred_transformed)
                metrics = evaluate_regression_predictions(
                    y_true=y_valid_raw,
                    y_pred=y_pred_raw,
                    target_names=TARGET_COLUMNS,
                    target_scales=target_scales,
                )

                record = {
                    "experiment_name": experiment.experiment_name,
                    "model_name": experiment.variant_name,
                    "target_strategy": experiment.target_strategy_name,
                    "loss_name": experiment.loss_config.name,
                    "random_seed": repeated_seed,
                    "fold_index": fold_index,
                    "n_train": len(train_index),
                    "n_valid": len(valid_index),
                    "fit_time_seconds": fit_time,
                    "best_epoch": fit_artifacts.best_epoch,
                    "best_val_loss": fit_artifacts.best_val_loss,
                    "best_val_combined_score": fit_artifacts.best_val_combined_score,
                }
                record.update(metrics)
                all_fold_records.append(record)

    fold_metrics = pd.DataFrame.from_records(all_fold_records)
    summary_results = summarize_stability_sprint(fold_metrics)
    report = build_stability_sprint_report(
        summary_results=summary_results,
        fold_metrics=fold_metrics,
        seeds=args.seeds,
    )

    _write_csv(summary_results, STABILITY_SPRINT_RESULTS_OUTPUT_PATH)
    _write_text(report, STABILITY_SPRINT_REPORT_OUTPUT_PATH)

    print(f"stability_sprint_results: {STABILITY_SPRINT_RESULTS_OUTPUT_PATH}")
    print(f"stability_sprint_report: {STABILITY_SPRINT_REPORT_OUTPUT_PATH}")
    print(f"best_stabilized_variant: {summary_results.iloc[0]['experiment_name']}")


if __name__ == "__main__":
    main()
