"""Target-specialist routing and Ridge stacking over existing non-tree models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_OUTPUTS_DIR,
    REPORTS_DIR,
    TARGET_SPECIALIST_REPORT_OUTPUT_PATH,
    TARGET_SPECIALIST_RESULTS_OUTPUT_PATH,
)
from src.eval.metrics import evaluate_regression_predictions
from src.models.train_baselines import (
    PreparedBaselineData,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    OXIDATION_TARGET,
    evaluate_single_baseline_configuration,
    get_model_spec_by_name,
    get_target_strategy_by_name as get_baseline_target_strategy_by_name,
    load_baseline_training_data,
    select_baseline_feature_columns,
)
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    evaluate_single_deep_sets_configuration,
    load_deep_sets_data,
)


BEST_BASELINE_FEATURE_SETTING = "conditions_structure_family"
BEST_BASELINE_MODEL_NAME = "pls_regression"
TARGET_STRATEGY_NAME = "raw"
RANDOM_SEED = 42
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


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
    parser.add_argument(
        "--skip-robust-viscosity",
        action="store_true",
        help="Skip the optional robust-viscosity hybrid branch.",
    )
    return parser.parse_args()


def _select_baseline_columns(prepared_data: PreparedBaselineData, feature_setting: str) -> list[str]:
    return select_baseline_feature_columns(prepared_data=prepared_data, feature_setting=feature_setting)


def collect_oof_predictions(
    outer_splits: int,
    config: DeepSetsConfig,
    include_robust_viscosity: bool,
) -> pd.DataFrame:
    """Collect aligned OOF predictions for all base models used in the experiment."""

    baseline_prepared = load_baseline_training_data()
    baseline_columns = _select_baseline_columns(baseline_prepared, BEST_BASELINE_FEATURE_SETTING)
    baseline_subset = PreparedBaselineData(
        scenario_ids=baseline_prepared.scenario_ids.copy(),
        X=baseline_prepared.X.loc[:, baseline_columns].copy(),
        y=baseline_prepared.y.copy(),
        feature_manifest=baseline_prepared.feature_manifest,
    )
    baseline_artifacts = evaluate_single_baseline_configuration(
        prepared_data=baseline_subset,
        model_spec=get_model_spec_by_name(BEST_BASELINE_MODEL_NAME),
        target_strategy=get_baseline_target_strategy_by_name(TARGET_STRATEGY_NAME),
        outer_splits=outer_splits,
        inner_splits=3,
        seed=RANDOM_SEED,
        extra_metadata={"candidate_group": "base_model"},
    )
    base_frame = baseline_artifacts.oof_predictions.rename(
        columns={
            f"{VISCOSITY_TARGET}__pred": "pls_viscosity_pred",
            f"{OXIDATION_TARGET}__pred": "pls_oxidation_pred",
            f"{VISCOSITY_TARGET}__true": f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true": f"{OXIDATION_TARGET}__true",
        }
    ).loc[
        :,
        [
            "scenario_id",
            "fold_index",
            f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true",
            "viscosity_scale",
            "oxidation_scale",
            "pls_viscosity_pred",
            "pls_oxidation_pred",
        ],
    ]

    deep_sets_prepared = load_deep_sets_data()
    deep_sets_v1_variant = HybridVariant(
        name="deep_sets_v1",
        use_component_embedding=True,
        use_tabular_branch=False,
    )
    deep_sets_v1_artifacts = evaluate_single_deep_sets_configuration(
        prepared_data=deep_sets_prepared,
        config=config,
        variant=deep_sets_v1_variant,
        target_strategy=get_baseline_target_strategy_by_name(TARGET_STRATEGY_NAME),
        outer_splits=outer_splits,
        seed=RANDOM_SEED,
        extra_metadata={"candidate_group": "base_model"},
    )
    deep_sets_v1_frame = deep_sets_v1_artifacts.oof_predictions.rename(
        columns={
            f"{VISCOSITY_TARGET}__pred": "deep_sets_v1_viscosity_pred",
            f"{OXIDATION_TARGET}__pred": "deep_sets_v1_oxidation_pred",
        }
    ).loc[
        :,
        [
            "scenario_id",
            "fold_index",
            "deep_sets_v1_viscosity_pred",
            "deep_sets_v1_oxidation_pred",
        ],
    ]

    hybrid_v2_variant = HybridVariant(
        name="hybrid_deep_sets_v2_family_only",
        use_component_embedding=False,
        use_tabular_branch=True,
    )
    hybrid_v2_artifacts = evaluate_single_deep_sets_configuration(
        prepared_data=deep_sets_prepared,
        config=config,
        variant=hybrid_v2_variant,
        target_strategy=get_baseline_target_strategy_by_name(TARGET_STRATEGY_NAME),
        outer_splits=outer_splits,
        seed=RANDOM_SEED,
        extra_metadata={"candidate_group": "base_model"},
    )
    hybrid_v2_frame = hybrid_v2_artifacts.oof_predictions.rename(
        columns={
            f"{VISCOSITY_TARGET}__pred": "hybrid_v2_viscosity_pred",
            f"{OXIDATION_TARGET}__pred": "hybrid_v2_oxidation_pred",
        }
    ).loc[
        :,
        [
            "scenario_id",
            "fold_index",
            "hybrid_v2_viscosity_pred",
            "hybrid_v2_oxidation_pred",
        ],
    ]

    merged = base_frame.merge(
        deep_sets_v1_frame,
        on=["scenario_id", "fold_index"],
        how="inner",
        validate="one_to_one",
    ).merge(
        hybrid_v2_frame,
        on=["scenario_id", "fold_index"],
        how="inner",
        validate="one_to_one",
    )

    if include_robust_viscosity:
        robust_frame = evaluate_single_deep_sets_configuration(
            prepared_data=deep_sets_prepared,
            config=config,
            variant=hybrid_v2_variant,
            target_strategy=get_baseline_target_strategy_by_name(TARGET_STRATEGY_NAME),
            outer_splits=outer_splits,
            seed=RANDOM_SEED,
            loss_config=LossConfig(name="robust_viscosity", use_robust_viscosity_loss=True, viscosity_delta=1.0),
            extra_metadata={"candidate_group": "base_model"},
        ).oof_predictions.rename(
            columns={
                f"{VISCOSITY_TARGET}__pred": "hybrid_v2_robust_viscosity_pred",
            }
        ).loc[
            :,
            [
                "scenario_id",
                "fold_index",
                "hybrid_v2_robust_viscosity_pred",
            ],
        ]
        merged = merged.merge(
            robust_frame,
            on=["scenario_id", "fold_index"],
            how="inner",
            validate="one_to_one",
        )

    return merged.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _fit_target_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
) -> Pipeline:
    """Fit one target-wise Ridge blender with grouped inner CV."""

    inner_splits = max(2, min(4, len(np.unique(groups))))
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
        scoring=make_scorer(
            lambda y_true, y_pred: -float(np.sqrt(mean_squared_error(y_true, y_pred))),
            greater_is_better=True,
        ),
        cv=GroupKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=None,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_


def _evaluate_candidate_predictions(
    frame: pd.DataFrame,
    candidate_name: str,
    viscosity_pred: np.ndarray,
    oxidation_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one candidate from scenario-level held-out predictions."""

    prediction_frame = frame.loc[:, ["scenario_id", "fold_index", "viscosity_scale", "oxidation_scale"]].copy()
    prediction_frame["candidate_name"] = candidate_name
    prediction_frame[f"{VISCOSITY_TARGET}__true"] = frame[f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float)
    prediction_frame[f"{OXIDATION_TARGET}__true"] = frame[f"{OXIDATION_TARGET}__true"].to_numpy(dtype=float)
    prediction_frame[f"{VISCOSITY_TARGET}__pred"] = np.asarray(viscosity_pred, dtype=float)
    prediction_frame[f"{OXIDATION_TARGET}__pred"] = np.asarray(oxidation_pred, dtype=float)

    fold_records: list[dict[str, object]] = []
    for fold_index, fold_frame in prediction_frame.groupby("fold_index", dropna=False):
        metrics = evaluate_regression_predictions(
            y_true=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__pred", f"{OXIDATION_TARGET}__pred"]].to_numpy(dtype=float),
            target_names=TARGET_COLUMNS,
            target_scales={
                VISCOSITY_TARGET: float(fold_frame["viscosity_scale"].iloc[0]),
                OXIDATION_TARGET: float(fold_frame["oxidation_scale"].iloc[0]),
            },
        )
        fold_records.append(
            {
                "candidate_name": candidate_name,
                "fold_index": int(fold_index),
                **metrics,
            }
        )

    fold_metrics = pd.DataFrame.from_records(fold_records)
    summary = fold_metrics[
        [
            "candidate_name",
            "combined_score",
            f"{VISCOSITY_TARGET}__rmse",
            f"{OXIDATION_TARGET}__rmse",
        ]
    ].groupby("candidate_name", dropna=False).agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return fold_metrics, summary


def build_direct_routing_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Route viscosity to the hybrid and oxidation to Deep Sets v1."""

    return (
        frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float),
        frame["deep_sets_v1_oxidation_pred"].to_numpy(dtype=float),
    )


def build_target_wise_stacking_predictions(
    frame: pd.DataFrame,
    include_robust_viscosity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit target-wise Ridge blenders on OOF base predictions only."""

    viscosity_features = [
        "pls_viscosity_pred",
        "deep_sets_v1_viscosity_pred",
        "hybrid_v2_viscosity_pred",
    ]
    if include_robust_viscosity and "hybrid_v2_robust_viscosity_pred" in frame.columns:
        viscosity_features.append("hybrid_v2_robust_viscosity_pred")
    oxidation_features = [
        "pls_oxidation_pred",
        "deep_sets_v1_oxidation_pred",
        "hybrid_v2_oxidation_pred",
    ]

    viscosity_predictions = np.zeros(len(frame), dtype=float)
    oxidation_predictions = np.zeros(len(frame), dtype=float)

    for fold_index in sorted(frame["fold_index"].unique()):
        train_mask = frame["fold_index"] != fold_index
        valid_mask = frame["fold_index"] == fold_index

        viscosity_model = _fit_target_ridge(
            X_train=frame.loc[train_mask, viscosity_features],
            y_train=frame.loc[train_mask, f"{VISCOSITY_TARGET}__true"],
            groups=frame.loc[train_mask, "scenario_id"].to_numpy(),
        )
        oxidation_model = _fit_target_ridge(
            X_train=frame.loc[train_mask, oxidation_features],
            y_train=frame.loc[train_mask, f"{OXIDATION_TARGET}__true"],
            groups=frame.loc[train_mask, "scenario_id"].to_numpy(),
        )

        viscosity_predictions[valid_mask] = viscosity_model.predict(frame.loc[valid_mask, viscosity_features])
        oxidation_predictions[valid_mask] = oxidation_model.predict(frame.loc[valid_mask, oxidation_features])

    return viscosity_predictions, oxidation_predictions


def build_results_table(candidate_summaries: list[pd.DataFrame]) -> pd.DataFrame:
    results = pd.concat(candidate_summaries, ignore_index=True)
    results = results.sort_values(
        [
            "combined_score__mean",
            f"{VISCOSITY_TARGET}__rmse__mean",
            f"{OXIDATION_TARGET}__rmse__mean",
        ]
    ).reset_index(drop=True)
    results["rank_combined_score"] = np.arange(1, len(results) + 1)
    return results


def build_report(results: pd.DataFrame, include_robust_viscosity: bool) -> str:
    best_row = results.iloc[0]
    shipping_row = results.loc[results["candidate_name"] == "single_model__hybrid_deep_sets_v2_family_only__raw"].iloc[0]
    routing_row = results.loc[results["candidate_name"] == "direct_route__hybrid_v2_viscosity__deep_sets_v1_oxidation"].iloc[0]
    plain_stacking_row = results.loc[results["candidate_name"] == "ridge_stack__target_wise"].iloc[0]
    robust_stacking_row = None
    if include_robust_viscosity and "ridge_stack__target_wise__plus_robust_viscosity" in results["candidate_name"].tolist():
        robust_stacking_row = results.loc[
            results["candidate_name"] == "ridge_stack__target_wise__plus_robust_viscosity"
        ].iloc[0]

    lines = [
        "# Target Specialist Report",
        "",
        "## Experiment Design",
        "- Base models were reconstructed with grouped CV by `scenario_id` only.",
        "- Base OOF predictions were collected for `pls_regression / raw / conditions_structure_family`, `deep_sets_v1 / raw`, and `hybrid_deep_sets_v2_family_only / raw`.",
        (
            "- An optional `hybrid_deep_sets_v2_family_only / raw / robust_viscosity` branch was included "
            "for viscosity stacking only."
            if include_robust_viscosity
            else "- No robust-viscosity auxiliary branch was included."
        ),
        "- Direct routing uses hybrid v2 for viscosity and Deep Sets v1 for oxidation.",
        "- Ridge stacking trains separate linear blenders for viscosity and oxidation on OOF predictions only.",
        "",
        "## Candidate Comparison",
        "",
        "```text",
        results[
            [
                "rank_combined_score",
                "candidate_name",
                "combined_score__mean",
                "combined_score__std",
                f"{VISCOSITY_TARGET}__rmse__mean",
                f"{OXIDATION_TARGET}__rmse__mean",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Decision",
        (
            f"- Best candidate: `{best_row['candidate_name']}` with combined score "
            f"`{best_row['combined_score__mean']:.4f}`"
        ),
        (
            f"- Direct specialist routing {'beats' if routing_row['combined_score__mean'] < shipping_row['combined_score__mean'] else 'does not beat'} "
            "the current shipping model."
        ),
        (
            f"- Target-wise Ridge stacking {'beats' if plain_stacking_row['combined_score__mean'] < routing_row['combined_score__mean'] else 'does not beat'} "
            "direct routing."
        ),
        (
            f"- Platform-submission candidate recommendation: "
            f"`{best_row['candidate_name']}`"
        ),
        "",
        "## Key Deltas",
        (
            f"- Shipping reference combined score: `{shipping_row['combined_score__mean']:.4f}`"
        ),
        (
            f"- Direct routing combined score delta vs shipping: "
            f"`{routing_row['combined_score__mean'] - shipping_row['combined_score__mean']:+.4f}`"
        ),
        (
            f"- Plain target-wise stacking combined score delta vs direct routing: "
            f"`{plain_stacking_row['combined_score__mean'] - routing_row['combined_score__mean']:+.4f}`"
        ),
        (
            f"- Best-candidate viscosity RMSE: `{best_row[f'{VISCOSITY_TARGET}__rmse__mean']:.4f}`"
        ),
        (
            f"- Best-candidate oxidation RMSE: `{best_row[f'{OXIDATION_TARGET}__rmse__mean']:.4f}`"
        ),
    ]
    if robust_stacking_row is not None:
        lines.append(
            f"- Robust-viscosity stacking delta vs plain stacking: "
            f"`{robust_stacking_row['combined_score__mean'] - plain_stacking_row['combined_score__mean']:+.4f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )
    include_robust_viscosity = not args.skip_robust_viscosity
    oof_frame = collect_oof_predictions(
        outer_splits=args.outer_splits,
        config=config,
        include_robust_viscosity=include_robust_viscosity,
    )

    candidate_summaries: list[pd.DataFrame] = []
    for candidate_name, viscosity_col, oxidation_col in [
        (
            "single_model__pls_regression__raw__conditions_structure_family",
            "pls_viscosity_pred",
            "pls_oxidation_pred",
        ),
        (
            "single_model__deep_sets_v1__raw",
            "deep_sets_v1_viscosity_pred",
            "deep_sets_v1_oxidation_pred",
        ),
        (
            "single_model__hybrid_deep_sets_v2_family_only__raw",
            "hybrid_v2_viscosity_pred",
            "hybrid_v2_oxidation_pred",
        ),
    ]:
        _, summary = _evaluate_candidate_predictions(
            frame=oof_frame,
            candidate_name=candidate_name,
            viscosity_pred=oof_frame[viscosity_col].to_numpy(dtype=float),
            oxidation_pred=oof_frame[oxidation_col].to_numpy(dtype=float),
        )
        candidate_summaries.append(summary)

    routing_viscosity, routing_oxidation = build_direct_routing_predictions(oof_frame)
    _, routing_summary = _evaluate_candidate_predictions(
        frame=oof_frame,
        candidate_name="direct_route__hybrid_v2_viscosity__deep_sets_v1_oxidation",
        viscosity_pred=routing_viscosity,
        oxidation_pred=routing_oxidation,
    )
    candidate_summaries.append(routing_summary)

    stacked_viscosity, stacked_oxidation = build_target_wise_stacking_predictions(
        frame=oof_frame,
        include_robust_viscosity=False,
    )
    _, stacking_summary = _evaluate_candidate_predictions(
        frame=oof_frame,
        candidate_name="ridge_stack__target_wise",
        viscosity_pred=stacked_viscosity,
        oxidation_pred=stacked_oxidation,
    )
    candidate_summaries.append(stacking_summary)

    if include_robust_viscosity and "hybrid_v2_robust_viscosity_pred" in oof_frame.columns:
        robust_stacked_viscosity, robust_stacked_oxidation = build_target_wise_stacking_predictions(
            frame=oof_frame,
            include_robust_viscosity=True,
        )
        _, robust_stacking_summary = _evaluate_candidate_predictions(
            frame=oof_frame,
            candidate_name="ridge_stack__target_wise__plus_robust_viscosity",
            viscosity_pred=robust_stacked_viscosity,
            oxidation_pred=robust_stacked_oxidation,
        )
        candidate_summaries.append(robust_stacking_summary)

    results = build_results_table(candidate_summaries)
    report = build_report(results=results, include_robust_viscosity=include_robust_viscosity)

    _write_csv(results, TARGET_SPECIALIST_RESULTS_OUTPUT_PATH)
    _write_text(report, TARGET_SPECIALIST_REPORT_OUTPUT_PATH)

    print(f"target_specialist_results: {TARGET_SPECIALIST_RESULTS_OUTPUT_PATH}")
    print(f"target_specialist_report: {TARGET_SPECIALIST_REPORT_OUTPUT_PATH}")
    print(f"best_candidate: {results.iloc[0]['candidate_name']}")


if __name__ == "__main__":
    main()
