"""Chemistry-guided target-specialist convex blending with an optional viscosity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (
    CHEMISTRY_ENSEMBLE_REPORT_OUTPUT_PATH,
    CHEMISTRY_ENSEMBLE_RESULTS_OUTPUT_PATH,
    CV_OUTPUTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)
from src.eval.metrics import evaluate_regression_predictions
from src.eval.run_target_specialist import collect_oof_predictions
from src.models.train_baselines import OXIDATION_TARGET, TARGET_COLUMNS, VISCOSITY_TARGET
from src.models.train_deep_sets import DeepSetsConfig


BASELINE_VISCOSITY_COLUMNS = [
    "pls_viscosity_pred",
    "deep_sets_v1_viscosity_pred",
    "hybrid_v2_viscosity_pred",
]
ROBUST_VISCOSITY_COLUMN = "hybrid_v2_robust_viscosity_pred"
OXIDATION_COLUMNS = [
    "pls_oxidation_pred",
    "deep_sets_v1_oxidation_pred",
    "hybrid_v2_oxidation_pred",
]
SEVERITY_CORE_COLUMNS = [
    "test_temperature_c",
    "test_duration_h",
    "biofuel_mass_fraction_pct",
    "family__antioksidant__mass_sum",
    "family__antioksidant__mass_share",
    "family__zagustitel__mass_sum",
    "family__zagustitel__mass_share",
]
SEVERITY_MISSINGNESS_COLUMNS = [
    "missing_all_props__mass_share",
    "property_join_source__missing__row_ratio",
    "property_cell_nonmissing_density",
]


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
        help="Skip the optional robust-viscosity viscosity input and gate expert.",
    )
    return parser.parse_args()


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = np.asarray(logits, dtype=float) - float(np.max(logits))
    exponents = np.exp(centered)
    return exponents / np.sum(exponents)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_convex_weights(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit non-negative weights that sum to one through a simplex parameterization."""

    predictions = np.asarray(predictions, dtype=float)
    target = np.asarray(target, dtype=float)
    n_models = predictions.shape[1]

    def _objective(logits: np.ndarray) -> float:
        weights = _softmax(logits)
        blended = predictions @ weights
        return float(np.mean((target - blended) ** 2))

    result = minimize(
        _objective,
        x0=np.zeros(n_models, dtype=float),
        method="L-BFGS-B",
    )
    return _softmax(result.x)


def _predict_convex(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(predictions, dtype=float) @ np.asarray(weights, dtype=float)


def _load_severity_features(path: Path = TRAIN_SCENARIO_FEATURES_OUTPUT_PATH) -> pd.DataFrame:
    """Load chemistry-guided viscosity severity features at scenario level."""

    frame = pd.read_csv(path)
    required = [
        "scenario_id",
        "catalyst_dosage_category",
        *SEVERITY_CORE_COLUMNS,
        *SEVERITY_MISSINGNESS_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing severity feature columns: {missing}")

    selected = frame.loc[:, required].copy()
    selected["missingness_burden"] = (
        selected["missing_all_props__mass_share"].astype(float)
        + selected["property_join_source__missing__row_ratio"].astype(float)
        + (1.0 - selected["property_cell_nonmissing_density"].astype(float))
    )
    catalyst_dummies = pd.get_dummies(
        selected["catalyst_dosage_category"].astype(str),
        prefix="catalyst_category",
        dtype=float,
    )
    output = pd.concat(
        [
            selected.loc[:, ["scenario_id", *SEVERITY_CORE_COLUMNS, "missingness_burden"]].copy(),
            catalyst_dummies,
        ],
        axis=1,
    )
    return output


def build_experiment_frame(
    outer_splits: int,
    config: DeepSetsConfig,
    include_robust_viscosity: bool,
) -> pd.DataFrame:
    """Collect aligned OOF predictions and merge the severity features."""

    oof_frame = collect_oof_predictions(
        outer_splits=outer_splits,
        config=config,
        include_robust_viscosity=include_robust_viscosity,
    )
    severity_features = _load_severity_features()
    merged = oof_frame.merge(
        severity_features,
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(oof_frame):
        raise ValueError("Severity feature merge changed the OOF scenario count.")
    return merged.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _fit_gated_viscosity_model(
    low_predictions: np.ndarray,
    high_predictions: np.ndarray,
    severity_features: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit a smooth logistic gate between low- and high-severity convex experts."""

    low_predictions = np.asarray(low_predictions, dtype=float)
    high_predictions = np.asarray(high_predictions, dtype=float)
    target = np.asarray(target, dtype=float)

    feature_mean = np.nanmean(severity_features, axis=0)
    feature_std = np.nanstd(severity_features, axis=0)
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    feature_std = np.where(np.isfinite(feature_std) & (feature_std > 0), feature_std, 1.0)
    standardized = (severity_features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    low_count = low_predictions.shape[1]
    high_count = high_predictions.shape[1]
    feature_count = standardized.shape[1]

    def _objective(theta: np.ndarray) -> float:
        low_logits = theta[:low_count]
        high_logits = theta[low_count : low_count + high_count]
        gate_beta = theta[low_count + high_count : low_count + high_count + feature_count]
        gate_bias = theta[-1]

        low_weights = _softmax(low_logits)
        high_weights = _softmax(high_logits)
        low_blend = low_predictions @ low_weights
        high_blend = high_predictions @ high_weights
        gate = _sigmoid(standardized @ gate_beta + gate_bias)
        blended = (1.0 - gate) * low_blend + gate * high_blend
        regularization = 1e-3 * float(np.mean(gate_beta**2))
        return float(np.mean((target - blended) ** 2) + regularization)

    initial = np.zeros(low_count + high_count + feature_count + 1, dtype=float)
    result = minimize(_objective, x0=initial, method="L-BFGS-B")
    optimum = result.x
    return {
        "low_weights": _softmax(optimum[:low_count]),
        "high_weights": _softmax(optimum[low_count : low_count + high_count]),
        "gate_beta": optimum[low_count + high_count : low_count + high_count + feature_count],
        "gate_bias": np.asarray([optimum[-1]], dtype=float),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def _predict_gated_viscosity(
    low_predictions: np.ndarray,
    high_predictions: np.ndarray,
    severity_features: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    standardized = (severity_features - model["feature_mean"].reshape(1, -1)) / model["feature_std"].reshape(1, -1)
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    low_blend = np.asarray(low_predictions, dtype=float) @ model["low_weights"]
    high_blend = np.asarray(high_predictions, dtype=float) @ model["high_weights"]
    gate = _sigmoid(standardized @ model["gate_beta"] + float(model["gate_bias"][0]))
    return (1.0 - gate) * low_blend + gate * high_blend


def _evaluate_candidate(
    frame: pd.DataFrame,
    candidate_name: str,
    viscosity_predictions: np.ndarray,
    oxidation_predictions: np.ndarray,
    extra_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold_index, fold_frame in frame.groupby("fold_index", dropna=False):
        valid_mask = frame["fold_index"].to_numpy() == fold_index
        metrics = evaluate_regression_predictions(
            y_true=frame.loc[valid_mask, [VISCOSITY_TARGET + "__true", OXIDATION_TARGET + "__true"]].to_numpy(dtype=float),
            y_pred=np.column_stack(
                [
                    np.asarray(viscosity_predictions, dtype=float)[valid_mask],
                    np.asarray(oxidation_predictions, dtype=float)[valid_mask],
                ]
            ),
            target_names=TARGET_COLUMNS,
            target_scales={
                VISCOSITY_TARGET: float(fold_frame["viscosity_scale"].iloc[0]),
                OXIDATION_TARGET: float(fold_frame["oxidation_scale"].iloc[0]),
            },
        )
        record = {
            "candidate_name": candidate_name,
            "fold_index": int(fold_index),
            **metrics,
        }
        if extra_metadata:
            record.update(extra_metadata)
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_candidates(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "combined_score",
        f"{VISCOSITY_TARGET}__rmse",
        f"{OXIDATION_TARGET}__rmse",
    ]
    summary = fold_metrics.groupby("candidate_name", dropna=False)[metric_columns].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(
        [
            "combined_score__mean",
            f"{VISCOSITY_TARGET}__rmse__mean",
            f"{OXIDATION_TARGET}__rmse__mean",
        ]
    ).reset_index(drop=True)
    summary["rank_combined_score"] = np.arange(1, len(summary) + 1)
    return summary


def build_report(
    results: pd.DataFrame,
    convex_viscosity_weights: np.ndarray,
    convex_oxidation_weights: np.ndarray,
    gated_details: dict[str, np.ndarray] | None,
) -> str:
    shipping_row = results.loc[results["candidate_name"] == "shipping_reference__hybrid_v2_raw"].iloc[0]
    convex_row = results.loc[results["candidate_name"] == "convex_blend__target_wise"].iloc[0]
    gate_row = None
    if "convex_blend__target_wise__plus_viscosity_gate" in results["candidate_name"].tolist():
        gate_row = results.loc[
            results["candidate_name"] == "convex_blend__target_wise__plus_viscosity_gate"
        ].iloc[0]
    best_row = results.iloc[0]

    platform_ready = (
        best_row["combined_score__mean"] < shipping_row["combined_score__mean"]
        and best_row["combined_score__std"] <= shipping_row["combined_score__std"] + 0.10
    )
    oxidation_preserved = convex_row[f"{OXIDATION_TARGET}__rmse__mean"] <= shipping_row[f"{OXIDATION_TARGET}__rmse__mean"] + 1.0
    viscosity_improved = convex_row[f"{VISCOSITY_TARGET}__rmse__mean"] < shipping_row[f"{VISCOSITY_TARGET}__rmse__mean"]
    gate_added_value = gate_row is not None and gate_row["combined_score__mean"] < convex_row["combined_score__mean"]

    lines = [
        "# Chemistry Ensemble Report",
        "",
        "## Experiment Design",
        "- Base models remained unchanged and non-tree-based.",
        "- OOF base predictions were reconstructed for PLS, Deep Sets v1, hybrid Deep Sets v2 family-only raw, and optional hybrid raw robust-viscosity.",
        "- Target-wise convex blenders used non-negative weights constrained to sum to one.",
        "- The optional viscosity gate used a smooth logistic function over chemistry-guided severity features only.",
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
        "## Learned Convex Weights",
        f"- Viscosity blend weights over {BASELINE_VISCOSITY_COLUMNS + [ROBUST_VISCOSITY_COLUMN] if len(convex_viscosity_weights) == 4 else BASELINE_VISCOSITY_COLUMNS}: `{np.round(convex_viscosity_weights, 4).tolist()}`",
        f"- Oxidation blend weights over {OXIDATION_COLUMNS}: `{np.round(convex_oxidation_weights, 4).tolist()}`",
        "",
        "## Decision",
        f"- Best candidate: `{best_row['candidate_name']}` with combined score `{best_row['combined_score__mean']:.4f}`",
        f"- Improvement stable enough for a platform attempt: `{'yes' if platform_ready else 'no'}`",
        f"- Oxidation preserved while viscosity improved under simple convex blending: `{'yes' if oxidation_preserved and viscosity_improved else 'no'}`",
        f"- Severity gate added value over simple convex blending: `{'yes' if gate_added_value else 'no'}`",
        "",
        "## Key Deltas Vs Shipping",
        f"- Shipping combined score: `{shipping_row['combined_score__mean']:.4f}`",
        f"- Convex blend combined score delta: `{convex_row['combined_score__mean'] - shipping_row['combined_score__mean']:+.4f}`",
        f"- Convex blend viscosity RMSE delta: `{convex_row[f'{VISCOSITY_TARGET}__rmse__mean'] - shipping_row[f'{VISCOSITY_TARGET}__rmse__mean']:+.4f}`",
        f"- Convex blend oxidation RMSE delta: `{convex_row[f'{OXIDATION_TARGET}__rmse__mean'] - shipping_row[f'{OXIDATION_TARGET}__rmse__mean']:+.4f}`",
    ]
    if gate_row is not None:
        lines.extend(
            [
                f"- Gated blend combined score delta vs convex blend: `{gate_row['combined_score__mean'] - convex_row['combined_score__mean']:+.4f}`",
                f"- Gated blend viscosity RMSE delta vs convex blend: `{gate_row[f'{VISCOSITY_TARGET}__rmse__mean'] - convex_row[f'{VISCOSITY_TARGET}__rmse__mean']:+.4f}`",
            ]
        )
    if gated_details is not None:
        lines.extend(
            [
                "",
                "## Severity Gate Details",
                f"- Low-severity expert weights: `{np.round(gated_details['low_weights'], 4).tolist()}`",
                f"- High-severity expert weights: `{np.round(gated_details['high_weights'], 4).tolist()}`",
                f"- Gate feature coefficients: `{json.dumps({name: round(float(value), 4) for name, value in zip(gated_details['feature_names'], gated_details['gate_beta'], strict=False)}, sort_keys=True)}`",
            ]
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
    frame = build_experiment_frame(
        outer_splits=args.outer_splits,
        config=config,
        include_robust_viscosity=include_robust_viscosity,
    )

    oxidation_feature_columns = OXIDATION_COLUMNS
    viscosity_feature_columns = BASELINE_VISCOSITY_COLUMNS.copy()
    if include_robust_viscosity and ROBUST_VISCOSITY_COLUMN in frame.columns:
        viscosity_feature_columns.append(ROBUST_VISCOSITY_COLUMN)

    severity_feature_columns = [
        column
        for column in frame.columns
        if column in SEVERITY_CORE_COLUMNS or column == "missingness_burden" or column.startswith("catalyst_category_")
    ]

    shipping_viscosity = frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float)
    shipping_oxidation = frame["hybrid_v2_oxidation_pred"].to_numpy(dtype=float)

    convex_viscosity_predictions = np.zeros(len(frame), dtype=float)
    convex_oxidation_predictions = np.zeros(len(frame), dtype=float)
    gated_viscosity_predictions = np.zeros(len(frame), dtype=float) if include_robust_viscosity and ROBUST_VISCOSITY_COLUMN in frame.columns else None
    last_convex_viscosity_weights = np.zeros(len(viscosity_feature_columns), dtype=float)
    last_convex_oxidation_weights = np.zeros(len(oxidation_feature_columns), dtype=float)
    last_gate_details: dict[str, np.ndarray] | None = None

    for fold_index in sorted(frame["fold_index"].unique()):
        train_mask = frame["fold_index"] != fold_index
        valid_mask = frame["fold_index"] == fold_index

        viscosity_weights = _fit_convex_weights(
            predictions=frame.loc[train_mask, viscosity_feature_columns].to_numpy(dtype=float),
            target=frame.loc[train_mask, f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float),
        )
        oxidation_weights = _fit_convex_weights(
            predictions=frame.loc[train_mask, oxidation_feature_columns].to_numpy(dtype=float),
            target=frame.loc[train_mask, f"{OXIDATION_TARGET}__true"].to_numpy(dtype=float),
        )
        last_convex_viscosity_weights = viscosity_weights
        last_convex_oxidation_weights = oxidation_weights

        convex_viscosity_predictions[valid_mask] = _predict_convex(
            predictions=frame.loc[valid_mask, viscosity_feature_columns].to_numpy(dtype=float),
            weights=viscosity_weights,
        )
        convex_oxidation_predictions[valid_mask] = _predict_convex(
            predictions=frame.loc[valid_mask, oxidation_feature_columns].to_numpy(dtype=float),
            weights=oxidation_weights,
        )

        if gated_viscosity_predictions is not None:
            low_columns = BASELINE_VISCOSITY_COLUMNS
            high_columns = viscosity_feature_columns
            gate_model = _fit_gated_viscosity_model(
                low_predictions=frame.loc[train_mask, low_columns].to_numpy(dtype=float),
                high_predictions=frame.loc[train_mask, high_columns].to_numpy(dtype=float),
                severity_features=frame.loc[train_mask, severity_feature_columns].to_numpy(dtype=float),
                target=frame.loc[train_mask, f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float),
            )
            gated_viscosity_predictions[valid_mask] = _predict_gated_viscosity(
                low_predictions=frame.loc[valid_mask, low_columns].to_numpy(dtype=float),
                high_predictions=frame.loc[valid_mask, high_columns].to_numpy(dtype=float),
                severity_features=frame.loc[valid_mask, severity_feature_columns].to_numpy(dtype=float),
                model=gate_model,
            )
            last_gate_details = {
                **gate_model,
                "feature_names": np.asarray(severity_feature_columns, dtype=object),
            }

    fold_metrics = pd.concat(
        [
            _evaluate_candidate(
                frame=frame,
                candidate_name="shipping_reference__hybrid_v2_raw",
                viscosity_predictions=shipping_viscosity,
                oxidation_predictions=shipping_oxidation,
            ),
            _evaluate_candidate(
                frame=frame,
                candidate_name="convex_blend__target_wise",
                viscosity_predictions=convex_viscosity_predictions,
                oxidation_predictions=convex_oxidation_predictions,
            ),
        ],
        ignore_index=True,
    )
    if gated_viscosity_predictions is not None:
        fold_metrics = pd.concat(
            [
                fold_metrics,
                _evaluate_candidate(
                    frame=frame,
                    candidate_name="convex_blend__target_wise__plus_viscosity_gate",
                    viscosity_predictions=gated_viscosity_predictions,
                    oxidation_predictions=convex_oxidation_predictions,
                ),
            ],
            ignore_index=True,
        )

    results = summarize_candidates(fold_metrics)
    report = build_report(
        results=results,
        convex_viscosity_weights=last_convex_viscosity_weights,
        convex_oxidation_weights=last_convex_oxidation_weights,
        gated_details=last_gate_details,
    )

    _write_csv(results, CHEMISTRY_ENSEMBLE_RESULTS_OUTPUT_PATH)
    _write_text(report, CHEMISTRY_ENSEMBLE_REPORT_OUTPUT_PATH)
    print(f"chemistry_ensemble_results: {CHEMISTRY_ENSEMBLE_RESULTS_OUTPUT_PATH}")
    print(f"chemistry_ensemble_report: {CHEMISTRY_ENSEMBLE_REPORT_OUTPUT_PATH}")
    print(f"best_candidate: {results.iloc[0]['candidate_name']}")


if __name__ == "__main__":
    main()
