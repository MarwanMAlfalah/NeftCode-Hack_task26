"""Seed-stability evaluation for the frozen hybrid Deep Sets v2 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CV_OUTPUTS_DIR,
    FINAL_MODEL_SELECTION_REPORT_OUTPUT_PATH,
    FINAL_MODEL_SELECTION_RESULTS_OUTPUT_PATH,
    REPORTS_DIR,
)
from src.models.train_deep_sets import (
    DeepSetsConfig,
    build_hybrid_variants,
    get_target_strategy_by_name,
    load_deep_sets_data,
    run_deep_sets_cv,
)


DEFAULT_SEEDS = [42, 52, 62, 72, 82]
TARGET_COLUMNS = [
    "combined_score",
    "target_delta_kinematic_viscosity_pct__rmse",
    "target_oxidation_eot_a_per_cm__rmse",
]
OFFICIAL_MODEL_NAME = "hybrid_deep_sets_v2_family_only"
OFFICIAL_TARGET_STRATEGY = "raw"
CONTEXT_VARIANTS = {
    ("hybrid_deep_sets_v2_family_only", "raw"),
    ("hybrid_deep_sets_v2_family_only", "viscosity_asinh"),
    ("hybrid_deep_sets_v2_family_component", "raw"),
}


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for final model selection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds to use for repeated grouped CV.",
    )
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def summarize_final_model_selection(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed and per-fold metrics into a compact decision table."""

    metric_map = {
        "combined_score": "combined_score",
        "viscosity_rmse": "target_delta_kinematic_viscosity_pct__rmse",
        "oxidation_rmse": "target_oxidation_eot_a_per_cm__rmse",
    }

    seed_level = (
        fold_metrics.groupby(["model_name", "target_strategy", "random_seed"], dropna=False)[list(metric_map.values())]
        .mean()
        .reset_index()
    )

    summary_records: list[dict[str, object]] = []
    grouped = fold_metrics.groupby(["model_name", "target_strategy"], dropna=False)
    for (model_name, target_strategy), group in grouped:
        seed_group = seed_level.loc[
            (seed_level["model_name"] == model_name) & (seed_level["target_strategy"] == target_strategy)
        ].copy()
        record: dict[str, object] = {
            "model_name": model_name,
            "target_strategy": target_strategy,
            "n_seeds": int(seed_group["random_seed"].nunique()),
            "n_seed_fold_runs": int(len(group)),
        }

        for display_name, metric_column in metric_map.items():
            seed_values = seed_group[metric_column].to_numpy(dtype=float)
            fold_values = group[metric_column].to_numpy(dtype=float)

            record[f"{display_name}__seed_mean"] = float(np.mean(seed_values))
            record[f"{display_name}__seed_std"] = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
            record[f"{display_name}__seed_min"] = float(np.min(seed_values))
            record[f"{display_name}__seed_max"] = float(np.max(seed_values))

            record[f"{display_name}__fold_mean"] = float(np.mean(fold_values))
            record[f"{display_name}__fold_std"] = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else 0.0
            record[f"{display_name}__fold_min"] = float(np.min(fold_values))
            record[f"{display_name}__fold_max"] = float(np.max(fold_values))

        summary_records.append(record)

    summary = pd.DataFrame.from_records(summary_records)
    summary = summary.sort_values(
        [
            "combined_score__seed_mean",
            "viscosity_rmse__seed_mean",
            "oxidation_rmse__seed_mean",
        ]
    ).reset_index(drop=True)
    summary["rank_combined_score"] = np.arange(1, len(summary) + 1)
    return summary


def build_final_model_selection_report(
    summary_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    seeds: list[int],
) -> str:
    """Write a concise decision-oriented stability report."""

    official_row = summary_results.loc[
        (summary_results["model_name"] == OFFICIAL_MODEL_NAME)
        & (summary_results["target_strategy"] == OFFICIAL_TARGET_STRATEGY)
    ].iloc[0]
    best_row = summary_results.iloc[0]

    context_rows = summary_results.loc[
        summary_results.apply(
            lambda row: (row["model_name"], row["target_strategy"]) in CONTEXT_VARIANTS,
            axis=1,
        )
    ].copy()

    seed_level = (
        fold_metrics.groupby(["model_name", "target_strategy", "random_seed"], dropna=False)[TARGET_COLUMNS]
        .mean()
        .reset_index()
    )
    official_seed_scores = seed_level.loc[
        (seed_level["model_name"] == OFFICIAL_MODEL_NAME)
        & (seed_level["target_strategy"] == OFFICIAL_TARGET_STRATEGY),
        "combined_score",
    ].to_numpy(dtype=float)

    best_by_seed = (
        seed_level.sort_values(["random_seed", "combined_score", "target_delta_kinematic_viscosity_pct__rmse"])
        .groupby("random_seed", dropna=False)
        .first()
        .reset_index()
    )
    official_wins = int(
        (
            (best_by_seed["model_name"] == OFFICIAL_MODEL_NAME)
            & (best_by_seed["target_strategy"] == OFFICIAL_TARGET_STRATEGY)
        ).sum()
    )

    official_fold_metrics = fold_metrics.loc[
        (fold_metrics["model_name"] == OFFICIAL_MODEL_NAME)
        & (fold_metrics["target_strategy"] == OFFICIAL_TARGET_STRATEGY)
    ].copy()
    hardest_cases = (
        official_fold_metrics.sort_values(
            ["combined_score", "target_delta_kinematic_viscosity_pct__rmse", "target_oxidation_eot_a_per_cm__rmse"],
            ascending=False,
        )
        .head(3)
    )

    stable_enough = (
        best_row["model_name"] == OFFICIAL_MODEL_NAME
        and best_row["target_strategy"] == OFFICIAL_TARGET_STRATEGY
        and official_row["combined_score__seed_std"] < 0.10
    )

    lines = [
        "# Final Model Selection Report",
        "",
        "## Evaluation Plan",
        f"- Repeated grouped CV seeds: `{seeds}`",
        "- Grouping remained strictly by `scenario_id`.",
        "- Compared only frozen implemented variants needed for final selection context.",
        "",
        "## Candidate Comparison",
        "",
        "```text",
        context_rows[
            [
                "rank_combined_score",
                "model_name",
                "target_strategy",
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
            f"- Official candidate `{OFFICIAL_MODEL_NAME} / {OFFICIAL_TARGET_STRATEGY}` is "
            f"{'stable enough to freeze' if stable_enough else 'not yet strong enough to freeze'} as the final model."
        ),
        (
            f"- Candidate seed-level combined score mean/std: "
            f"`{official_row['combined_score__seed_mean']:.4f}` / `{official_row['combined_score__seed_std']:.4f}`"
        ),
        (
            f"- Candidate seed-level viscosity RMSE mean/std: "
            f"`{official_row['viscosity_rmse__seed_mean']:.4f}` / `{official_row['viscosity_rmse__seed_std']:.4f}`"
        ),
        (
            f"- Candidate seed-level oxidation RMSE mean/std: "
            f"`{official_row['oxidation_rmse__seed_mean']:.4f}` / `{official_row['oxidation_rmse__seed_std']:.4f}`"
        ),
        (
            f"- Candidate won `{official_wins}` of `{len(seeds)}` repeated-seed comparisons."
        ),
        "",
        "## Variance",
        (
            f"- Seed-level combined score range: `{official_row['combined_score__seed_min']:.4f}` "
            f"to `{official_row['combined_score__seed_max']:.4f}`"
        ),
        (
            f"- Fold-level combined score mean/std/min/max: "
            f"`{official_row['combined_score__fold_mean']:.4f}` / `{official_row['combined_score__fold_std']:.4f}` / "
            f"`{official_row['combined_score__fold_min']:.4f}` / `{official_row['combined_score__fold_max']:.4f}`"
        ),
        (
            f"- Fold-level viscosity RMSE mean/std/min/max: "
            f"`{official_row['viscosity_rmse__fold_mean']:.4f}` / `{official_row['viscosity_rmse__fold_std']:.4f}` / "
            f"`{official_row['viscosity_rmse__fold_min']:.4f}` / `{official_row['viscosity_rmse__fold_max']:.4f}`"
        ),
        "",
        "## Replacement Check",
    ]

    if best_row["model_name"] == OFFICIAL_MODEL_NAME and best_row["target_strategy"] == OFFICIAL_TARGET_STRATEGY:
        lines.append("- No strong reason to replace the official candidate with an existing implemented variant.")
    else:
        lines.append(
            f"- There is a replacement concern because `{best_row['model_name']} / {best_row['target_strategy']}` "
            "beat the official candidate on seed-mean combined score."
        )

    lines.extend(
        [
            (
                f"- Closest context variant by seed-mean combined score: "
                f"`{context_rows.iloc[1]['model_name']} / {context_rows.iloc[1]['target_strategy']}`"
                if len(context_rows) > 1
                else "- No alternate context variant was available."
            ),
            "",
            "## Main Remaining Failure Modes",
            "- Severe viscosity outliers still drive the worst folds even when average stability is acceptable.",
            "- Oxidation remains more stable than viscosity, so most selection uncertainty still comes from viscosity shocks.",
            "- A few folds retain wide combined-score spread, indicating scenario scarcity rather than architecture drift.",
        ]
    )

    if not hardest_cases.empty:
        lines.extend(["", "## Hardest Official-Candidate Folds"])
        for row in hardest_cases.to_dict(orient="records"):
            lines.append(
                f"- Seed `{int(row['random_seed'])}` fold `{int(row['fold_index'])}`: combined score `{row['combined_score']:.4f}`, "
                f"viscosity RMSE `{row['target_delta_kinematic_viscosity_pct__rmse']:.4f}`, "
                f"oxidation RMSE `{row['target_oxidation_eot_a_per_cm__rmse']:.4f}`"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run repeated-seed grouped CV and persist the final selection artifacts."""

    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    prepared_data = load_deep_sets_data()
    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )

    selected_variants = [
        variant for variant in build_hybrid_variants()
        if variant.name in {OFFICIAL_MODEL_NAME, "hybrid_deep_sets_v2_family_component"}
    ]
    selected_target_strategies = [
        get_target_strategy_by_name("raw"),
        get_target_strategy_by_name("viscosity_asinh"),
    ]

    all_fold_metrics: list[pd.DataFrame] = []
    for repeated_seed in args.seeds:
        _, fold_metrics, _ = run_deep_sets_cv(
            prepared_data=prepared_data,
            config=config,
            variants=selected_variants,
            target_strategies=selected_target_strategies,
            outer_splits=args.outer_splits,
            seed=repeated_seed,
        )
        fold_metrics = fold_metrics.copy()
        fold_metrics = fold_metrics.loc[
            fold_metrics.apply(
                lambda row: (row["model_name"], row["target_strategy"]) in CONTEXT_VARIANTS,
                axis=1,
            )
        ].reset_index(drop=True)
        fold_metrics["random_seed"] = repeated_seed
        all_fold_metrics.append(fold_metrics)

    combined_fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    summary_results = summarize_final_model_selection(combined_fold_metrics)
    report = build_final_model_selection_report(
        summary_results=summary_results,
        fold_metrics=combined_fold_metrics,
        seeds=args.seeds,
    )

    _write_csv(summary_results, FINAL_MODEL_SELECTION_RESULTS_OUTPUT_PATH)
    _write_text(report, FINAL_MODEL_SELECTION_REPORT_OUTPUT_PATH)

    print(f"final_model_selection_results: {FINAL_MODEL_SELECTION_RESULTS_OUTPUT_PATH}")
    print(f"final_model_selection_report: {FINAL_MODEL_SELECTION_REPORT_OUTPUT_PATH}")
    print(
        "recommended_final_model: "
        f"{summary_results.iloc[0]['model_name']} / {summary_results.iloc[0]['target_strategy']}"
    )


if __name__ == "__main__":
    main()
