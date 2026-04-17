"""Late-stage local recalibration on top of the frozen shipping predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import build_bundle_from_predictions, validate_predictions_csv
from src.config import (
    LOCAL_RECALIBRATION_REPORT_OUTPUT_PATH,
    LOCAL_RECALIBRATION_RESULTS_OUTPUT_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEST_SCENARIO_FEATURES_OUTPUT_PATH,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)
from src.eval.metrics import evaluate_regression_predictions
from src.eval.run_target_specialist import collect_oof_predictions
from src.models.train_baselines import TARGET_COLUMNS, VISCOSITY_TARGET, OXIDATION_TARGET
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    PREDICTION_COLUMN_MAP,
    load_deep_sets_data,
    train_full_deep_sets_variant_ensemble_and_predict,
)


SHIPPING_PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
CANDIDATE_STEM = "neftekod_dot_submission_local_recalibration_v1"
CANDIDATE_ZIP_NAME = f"{CANDIDATE_STEM}.zip"
SHRINKAGES = [0.10, 0.15, 0.20]
RIDGE_ALPHA_GRID = [10.0, 25.0, 50.0, 100.0, 250.0]
SEVERE_QUANTILE = 0.75


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _to_internal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]]: TARGET_COLUMNS[0],
            PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]]: TARGET_COLUMNS[1],
        }
    )
    return renamed.sort_values("scenario_id").reset_index(drop=True)


def _to_submission_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            TARGET_COLUMNS[0]: PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]],
            TARGET_COLUMNS[1]: PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]],
        }
    )
    return renamed.loc[:, ["scenario_id", PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]], PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    return parser.parse_args()


def _load_recalibration_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "scenario_id",
        "test_temperature_c",
        "test_duration_h",
        "biofuel_mass_fraction_pct",
        "catalyst_dosage_category",
        "missing_all_props__mass_share",
        "property_join_source__missing__row_ratio",
        "property_cell_nonmissing_density",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing recalibration feature columns: {missing}")
    selected = frame.loc[:, required].copy()
    selected["missingness_burden"] = (
        selected["missing_all_props__mass_share"].astype(float)
        + selected["property_join_source__missing__row_ratio"].astype(float)
        + (1.0 - selected["property_cell_nonmissing_density"].astype(float))
    )
    return selected.loc[
        :,
        [
            "scenario_id",
            "test_temperature_c",
            "test_duration_h",
            "biofuel_mass_fraction_pct",
            "catalyst_dosage_category",
            "missingness_burden",
        ],
    ]


def _fit_severity_params(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    columns = [
        "test_temperature_c",
        "test_duration_h",
        "biofuel_mass_fraction_pct",
        "catalyst_dosage_category",
        "missingness_burden",
    ]
    values = frame.loc[:, columns].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return {
        "columns": np.asarray(columns, dtype=object),
        "mean": mean,
        "std": std,
    }


def _apply_severity_score(frame: pd.DataFrame, params: dict[str, np.ndarray]) -> np.ndarray:
    columns = params["columns"].tolist()
    values = frame.loc[:, columns].to_numpy(dtype=float)
    standardized = (values - params["mean"].reshape(1, -1)) / params["std"].reshape(1, -1)
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(standardized, axis=1)


def _fit_small_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> Pipeline:
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    unique_group_count = len(np.unique(groups))
    if unique_group_count < 2 or len(X_train) < 4:
        estimator.set_params(ridge__alpha=50.0)
        estimator.fit(X_train, y_train)
        return estimator

    inner_splits = max(2, min(4, unique_group_count))
    search = GridSearchCV(
        estimator=estimator,
        param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=None,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_


def _load_oof_frame(config: DeepSetsConfig, outer_splits: int) -> pd.DataFrame:
    base = collect_oof_predictions(
        outer_splits=outer_splits,
        config=config,
        include_robust_viscosity=False,
    )
    features = _load_recalibration_features(TRAIN_SCENARIO_FEATURES_OUTPUT_PATH)
    merged = base.merge(features, on="scenario_id", how="inner", validate="one_to_one")
    return merged.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _candidate_predictions_from_fold(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    severity_params = _fit_severity_params(train_frame)
    train_severity = _apply_severity_score(train_frame, severity_params)
    valid_severity = _apply_severity_score(valid_frame, severity_params)

    oxidation_train_features = np.column_stack(
        [
            train_frame["hybrid_v2_oxidation_pred"].to_numpy(dtype=float),
            train_frame["deep_sets_v1_oxidation_pred"].to_numpy(dtype=float),
            train_severity,
        ]
    )
    oxidation_valid_features = np.column_stack(
        [
            valid_frame["hybrid_v2_oxidation_pred"].to_numpy(dtype=float),
            valid_frame["deep_sets_v1_oxidation_pred"].to_numpy(dtype=float),
            valid_severity,
        ]
    )
    oxidation_model = _fit_small_ridge(
        X_train=oxidation_train_features,
        y_train=train_frame[f"{OXIDATION_TARGET}__true"].to_numpy(dtype=float),
        groups=train_frame["scenario_id"].to_numpy(),
    )
    calibrated_oxidation_train = oxidation_model.predict(oxidation_train_features)
    calibrated_oxidation_valid = oxidation_model.predict(oxidation_valid_features)

    severity_threshold = float(np.quantile(train_severity, SEVERE_QUANTILE))
    oxidation_threshold = float(np.quantile(calibrated_oxidation_train, SEVERE_QUANTILE))

    severe_train_mask = (train_severity >= severity_threshold) | (calibrated_oxidation_train >= oxidation_threshold)
    severe_valid_mask = (valid_severity >= severity_threshold) | (calibrated_oxidation_valid >= oxidation_threshold)

    base_viscosity_train = train_frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float)
    base_viscosity_valid = valid_frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float)

    correction_train_features = np.column_stack(
        [
            base_viscosity_train,
            calibrated_oxidation_train,
            train_severity,
            calibrated_oxidation_train * train_severity,
        ]
    )
    correction_valid_features = np.column_stack(
        [
            base_viscosity_valid,
            calibrated_oxidation_valid,
            valid_severity,
            calibrated_oxidation_valid * valid_severity,
        ]
    )
    residual_targets = train_frame[f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float) - base_viscosity_train

    corrected_viscosity = base_viscosity_valid.copy()
    if int(np.sum(severe_train_mask)) >= 8:
        correction_model = _fit_small_ridge(
            X_train=correction_train_features[severe_train_mask],
            y_train=residual_targets[severe_train_mask],
            groups=train_frame.loc[severe_train_mask, "scenario_id"].to_numpy(),
        )
        predicted_residual = correction_model.predict(correction_valid_features)
        corrected_viscosity[severe_valid_mask] += shrinkage * predicted_residual[severe_valid_mask]

    return corrected_viscosity, calibrated_oxidation_valid, severe_valid_mask.astype(bool)


def _evaluate_candidate(
    frame: pd.DataFrame,
    candidate_name: str,
    shrinkage: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_records: list[dict[str, object]] = []
    severe_records: list[dict[str, object]] = []

    for fold_index in sorted(frame["fold_index"].unique()):
        train_frame = frame.loc[frame["fold_index"] != fold_index].reset_index(drop=True)
        valid_frame = frame.loc[frame["fold_index"] == fold_index].reset_index(drop=True)

        if shrinkage is None:
            viscosity_predictions = valid_frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float)
            oxidation_predictions = valid_frame["hybrid_v2_oxidation_pred"].to_numpy(dtype=float)
            severe_mask = np.zeros(len(valid_frame), dtype=bool)
        else:
            viscosity_predictions, oxidation_predictions, severe_mask = _candidate_predictions_from_fold(
                train_frame=train_frame,
                valid_frame=valid_frame,
                shrinkage=shrinkage,
            )

        metrics = evaluate_regression_predictions(
            y_true=valid_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=np.column_stack([viscosity_predictions, oxidation_predictions]),
            target_names=TARGET_COLUMNS,
            target_scales={
                VISCOSITY_TARGET: float(valid_frame["viscosity_scale"].iloc[0]),
                OXIDATION_TARGET: float(valid_frame["oxidation_scale"].iloc[0]),
            },
        )
        fold_records.append(
            {
                "candidate_name": candidate_name,
                "shrinkage": 0.0 if shrinkage is None else shrinkage,
                "fold_index": int(fold_index),
                "severe_case_share": float(np.mean(severe_mask)) if len(severe_mask) else 0.0,
                **metrics,
            }
        )

        severe_true = valid_frame.loc[severe_mask, f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float)
        severe_pred = np.asarray(viscosity_predictions, dtype=float)[severe_mask]
        severe_shipping_pred = valid_frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float)[severe_mask]
        severe_rmse = float(np.sqrt(np.mean((severe_true - severe_pred) ** 2))) if len(severe_true) else np.nan
        severe_shipping_rmse = (
            float(np.sqrt(np.mean((severe_true - severe_shipping_pred) ** 2))) if len(severe_true) else np.nan
        )
        severe_records.append(
            {
                "candidate_name": candidate_name,
                "shrinkage": 0.0 if shrinkage is None else shrinkage,
                "fold_index": int(fold_index),
                "severe_case_count": int(np.sum(severe_mask)),
                "severe_viscosity_rmse": severe_rmse,
                "severe_shipping_viscosity_rmse": severe_shipping_rmse,
            }
        )

    return pd.DataFrame.from_records(fold_records), pd.DataFrame.from_records(severe_records)


def _summarize_results(fold_metrics: pd.DataFrame, severe_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "combined_score",
        f"{VISCOSITY_TARGET}__rmse",
        f"{OXIDATION_TARGET}__rmse",
        "severe_case_share",
    ]
    summary = (
        fold_metrics.groupby(["candidate_name", "shrinkage"], dropna=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    severe_summary = (
        severe_metrics.groupby(
            ["candidate_name", "shrinkage"], dropna=False
        )[["severe_case_count", "severe_viscosity_rmse", "severe_shipping_viscosity_rmse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    severe_summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in severe_summary.columns
    ]
    merged = summary.merge(severe_summary, on=["candidate_name", "shrinkage"], how="left", validate="one_to_one")
    merged = merged.sort_values(
        ["combined_score__mean", f"{VISCOSITY_TARGET}__rmse__mean", f"{OXIDATION_TARGET}__rmse__mean"]
    ).reset_index(drop=True)
    merged["rank_combined_score"] = np.arange(1, len(merged) + 1)
    return merged


def _fit_final_candidate(
    frame: pd.DataFrame,
    shipping_test: pd.DataFrame,
    deep_sets_v1_test: pd.DataFrame,
    test_features: pd.DataFrame,
    shrinkage: float,
) -> pd.DataFrame:
    severity_params = _fit_severity_params(frame)
    train_severity = _apply_severity_score(frame, severity_params)
    test_severity = _apply_severity_score(test_features, severity_params)

    oxidation_train_features = np.column_stack(
        [
            frame["hybrid_v2_oxidation_pred"].to_numpy(dtype=float),
            frame["deep_sets_v1_oxidation_pred"].to_numpy(dtype=float),
            train_severity,
        ]
    )
    oxidation_model = _fit_small_ridge(
        X_train=oxidation_train_features,
        y_train=frame[f"{OXIDATION_TARGET}__true"].to_numpy(dtype=float),
        groups=frame["scenario_id"].to_numpy(),
    )
    calibrated_oxidation_train = oxidation_model.predict(oxidation_train_features)
    oxidation_test_features = np.column_stack(
        [
            shipping_test[TARGET_COLUMNS[1]].to_numpy(dtype=float),
            deep_sets_v1_test[TARGET_COLUMNS[1]].to_numpy(dtype=float),
            test_severity,
        ]
    )
    calibrated_oxidation_test = oxidation_model.predict(oxidation_test_features)

    severity_threshold = float(np.quantile(train_severity, SEVERE_QUANTILE))
    oxidation_threshold = float(np.quantile(calibrated_oxidation_train, SEVERE_QUANTILE))
    severe_train_mask = (train_severity >= severity_threshold) | (calibrated_oxidation_train >= oxidation_threshold)
    severe_test_mask = (test_severity >= severity_threshold) | (calibrated_oxidation_test >= oxidation_threshold)

    base_viscosity_train = frame["hybrid_v2_viscosity_pred"].to_numpy(dtype=float)
    residual_targets = frame[f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float) - base_viscosity_train
    correction_train_features = np.column_stack(
        [
            base_viscosity_train,
            calibrated_oxidation_train,
            train_severity,
            calibrated_oxidation_train * train_severity,
        ]
    )
    correction_model = _fit_small_ridge(
        X_train=correction_train_features[severe_train_mask],
        y_train=residual_targets[severe_train_mask],
        groups=frame.loc[severe_train_mask, "scenario_id"].to_numpy(),
    )

    correction_test_features = np.column_stack(
        [
            shipping_test[TARGET_COLUMNS[0]].to_numpy(dtype=float),
            calibrated_oxidation_test,
            test_severity,
            calibrated_oxidation_test * test_severity,
        ]
    )
    corrected_viscosity = shipping_test[TARGET_COLUMNS[0]].to_numpy(dtype=float).copy()
    predicted_residual = correction_model.predict(correction_test_features)
    corrected_viscosity[severe_test_mask] += shrinkage * predicted_residual[severe_test_mask]

    return pd.DataFrame(
        {
            "scenario_id": shipping_test["scenario_id"].to_numpy(),
            TARGET_COLUMNS[0]: corrected_viscosity,
            TARGET_COLUMNS[1]: calibrated_oxidation_test,
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def build_report(results: pd.DataFrame, package_path: Path | None) -> str:
    shipping_row = results.loc[results["candidate_name"] == "shipping_reference"].iloc[0]
    corrected_rows = results.loc[results["candidate_name"] != "shipping_reference"].copy()
    best_row = corrected_rows.sort_values("combined_score__mean", ascending=True).iloc[0]

    severe_improved = best_row["severe_viscosity_rmse__mean"] < best_row["severe_shipping_viscosity_rmse__mean"]
    overall_improved = best_row["combined_score__mean"] < shipping_row["combined_score__mean"]
    safest_row = corrected_rows.sort_values(["combined_score__mean", "combined_score__std"]).iloc[0]

    lines = [
        "# Local Recalibration Report",
        "",
        "## Candidate Comparison",
        "",
        "```text",
        results[
            [
                "rank_combined_score",
                "candidate_name",
                "shrinkage",
                "combined_score__mean",
                "combined_score__std",
                f"{VISCOSITY_TARGET}__rmse__mean",
                f"{OXIDATION_TARGET}__rmse__mean",
                "severe_viscosity_rmse__mean",
                "severe_case_share__mean",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Decision",
        f"- Did local recalibration improve over shipping in OOF evaluation: `{'yes' if overall_improved else 'no'}`",
        f"- Was the improvement concentrated in severe viscosity cases: `{'yes' if severe_improved else 'no'}`",
        f"- Safest shrinkage level: `{safest_row['shrinkage']:.2f}`",
        (
            f"- Platform recommendation: `{'try the packaged candidate' if overall_improved else 'abandon for now'}`"
        ),
        "",
        "## Key Deltas Vs Shipping",
        f"- Shipping combined score: `{shipping_row['combined_score__mean']:.4f}`",
        f"- Best corrected combined score delta: `{best_row['combined_score__mean'] - shipping_row['combined_score__mean']:+.4f}`",
        f"- Best corrected viscosity RMSE delta: `{best_row[f'{VISCOSITY_TARGET}__rmse__mean'] - shipping_row[f'{VISCOSITY_TARGET}__rmse__mean']:+.4f}`",
        f"- Best corrected oxidation RMSE delta: `{best_row[f'{OXIDATION_TARGET}__rmse__mean'] - shipping_row[f'{OXIDATION_TARGET}__rmse__mean']:+.4f}`",
        f"- Best corrected severe-case viscosity RMSE delta vs shipping on the same severe cases: `{best_row['severe_viscosity_rmse__mean'] - best_row['severe_shipping_viscosity_rmse__mean']:+.4f}`",
    ]
    if package_path is not None:
        lines.extend(
            [
                "",
                "## Packaged Candidate",
                f"- Candidate ZIP: `{package_path}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Packaged Candidate",
                "- No candidate was packaged because the corrected variants did not look strong enough.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
    )

    oof_frame = _load_oof_frame(config=config, outer_splits=args.outer_splits)
    shipping_fold_metrics, shipping_severe_metrics = _evaluate_candidate(
        frame=oof_frame,
        candidate_name="shipping_reference",
        shrinkage=None,
    )

    fold_tables = [shipping_fold_metrics]
    severe_tables = [shipping_severe_metrics]
    for shrinkage in SHRINKAGES:
        fold_metrics, severe_metrics = _evaluate_candidate(
            frame=oof_frame,
            candidate_name="local_recalibration",
            shrinkage=shrinkage,
        )
        fold_tables.append(fold_metrics)
        severe_tables.append(severe_metrics)

    fold_results = pd.concat(fold_tables, ignore_index=True)
    severe_results = pd.concat(severe_tables, ignore_index=True)
    summary_results = _summarize_results(fold_results, severe_results)
    _write_csv(summary_results, LOCAL_RECALIBRATION_RESULTS_OUTPUT_PATH)

    packaged_zip: Path | None = None
    corrected_rows = summary_results.loc[summary_results["candidate_name"] == "local_recalibration"].copy()
    if not corrected_rows.empty and corrected_rows["combined_score__mean"].min() < summary_results.loc[
        summary_results["candidate_name"] == "shipping_reference", "combined_score__mean"
    ].iloc[0]:
        best_corrected = corrected_rows.sort_values("combined_score__mean", ascending=True).iloc[0]
        best_shrinkage = float(best_corrected["shrinkage"])

        shipping_test = _to_internal_columns(pd.read_csv(SHIPPING_PREDICTIONS_PATH))
        deep_sets_v1_test = _to_internal_columns(
            train_full_deep_sets_variant_ensemble_and_predict(
                prepared_data=load_deep_sets_data(),
                variant=HybridVariant(name="deep_sets_v1", use_component_embedding=True, use_tabular_branch=False),
                target_strategy_name="raw",
                seeds=[0, 1, 2, 3, 4],
                config=config,
            )
        )
        test_features = _load_recalibration_features(TEST_SCENARIO_FEATURES_OUTPUT_PATH)
        final_candidate = _fit_final_candidate(
            frame=oof_frame,
            shipping_test=shipping_test,
            deep_sets_v1_test=deep_sets_v1_test,
            test_features=test_features,
            shrinkage=best_shrinkage,
        )
        candidate_dir = OUTPUTS_DIR / "submissions" / CANDIDATE_STEM
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_predictions_path = candidate_dir / "predictions.csv"
        _to_submission_columns(final_candidate).to_csv(candidate_predictions_path, index=False, encoding="utf-8")
        validate_predictions_csv(candidate_predictions_path)
        packaged_zip = build_bundle_from_predictions(
            predictions_path=candidate_predictions_path,
            zip_name=CANDIDATE_ZIP_NAME,
        )

    report = build_report(summary_results, packaged_zip)
    _write_text(report, LOCAL_RECALIBRATION_REPORT_OUTPUT_PATH)

    print(f"local_recalibration_results: {LOCAL_RECALIBRATION_RESULTS_OUTPUT_PATH}")
    print(f"local_recalibration_report: {LOCAL_RECALIBRATION_REPORT_OUTPUT_PATH}")
    if packaged_zip is not None:
        print(f"local_recalibration_zip: {packaged_zip}")


if __name__ == "__main__":
    main()
