"""Low-retrain GP Stage 2 diagnostic sprint over existing Stage 1.5 and GP outputs."""

from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import set_config
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import build_bundle_from_predictions, validate_predictions_csv
from src.config import (
    CV_OUTPUTS_DIR,
    GP_STAGE2_OOF_PREDICTIONS_OUTPUT_PATH,
    GP_STAGE2_REGIME_AUDIT_REPORT_OUTPUT_PATH,
    META_STACK_SEARCH_REPORT_OUTPUT_PATH,
    META_STACK_SEARCH_RESULTS_OUTPUT_PATH,
    OUTPUTS_DIR,
    PAIRED_BOOTSTRAP_CI_OUTPUT_PATH,
    PAIRED_BOOTSTRAP_CI_REPORT_OUTPUT_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)
from src.eval.metrics import PLATFORM_TARGET_SCALES, evaluate_platform_predictions, evaluate_regression_predictions
from src.eval.run_gp_ensemble_stage2 import (
    BEST_BASELINE_FEATURE_SETTING,
    STAGE15_CANDIDATE_NAME,
    _build_kernel,
    _collect_stage15_oof_predictions,
    _fit_final_stack_models,
    _fit_full_gp_predictions,
    _fit_full_stage15_deep_sets_predictions,
    _fit_gp_preprocessor,
    _predict_with_final_stack_models,
    _to_submission_columns,
)
from src.models.train_baselines import (
    OXIDATION_TARGET,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    PreparedBaselineData,
    load_baseline_training_data,
    select_baseline_feature_columns,
)


CURRENT_STACK_SOURCE = "current_stack"
STACK_DOT_SOURCE = "stack_matern_white_dot"
SOURCE_ORDER = [
    "stage15",
    "gp_matern_white",
    "gp_matern_white_dot",
    CURRENT_STACK_SOURCE,
    STACK_DOT_SOURCE,
]
TOP_TARGET_BLEND_COUNT = 5
BOOTSTRAP_RESAMPLES = 10000
STRONG_BOOTSTRAP_PROBABILITY = 0.95
PACKAGE_STEM_PREFIX = "neftekod_dot_submission_gp_stage2_meta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--top-target-blends", type=int, default=TOP_TARGET_BLEND_COUNT)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--strong-bootstrap-probability", type=float, default=STRONG_BOOTSTRAP_PROBABILITY)
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _target_key(target_name: str) -> str:
    return "viscosity" if target_name == VISCOSITY_TARGET else "oxidation"


def _prediction_column(source_name: str, target_name: str) -> str:
    return f"{source_name}_{_target_key(target_name)}_pred"


def _std_column(source_name: str, target_name: str) -> str:
    return f"{source_name}_{_target_key(target_name)}_std"


def _stack_input_frame(
    frame: pd.DataFrame,
    gp_source_name: str,
) -> pd.DataFrame:
    return frame.rename(
        columns={
            _prediction_column(gp_source_name, VISCOSITY_TARGET): "gp_viscosity_pred",
            _prediction_column(gp_source_name, OXIDATION_TARGET): "gp_oxidation_pred",
            _std_column(gp_source_name, VISCOSITY_TARGET): "gp_viscosity_std",
            _std_column(gp_source_name, OXIDATION_TARGET): "gp_oxidation_std",
        }
    )


def _fit_gp_target_with_std(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    kernel_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianProcessRegressor(
        kernel=_build_kernel(kernel_name),
        normalize_y=True,
        n_restarts_optimizer=2,
        alpha=1e-8,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))
    pred_mean, pred_std = model.predict(X_valid, return_std=True)
    return np.asarray(pred_mean, dtype=float), np.asarray(pred_std, dtype=float)


def _collect_gp_oof_predictions_with_uncertainty(
    prepared_data: PreparedBaselineData,
    deep_oof: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    fold_assignments = deep_oof.loc[:, ["scenario_id", "fold_index"]].sort_values("scenario_id").reset_index(drop=True)
    scenario_ids = prepared_data.scenario_ids.reset_index(drop=True).copy()
    if not scenario_ids.sort_values(ignore_index=True).equals(fold_assignments["scenario_id"]):
        raise ValueError("Stage 1.5 OOF folds do not align with the baseline scenario IDs.")

    fold_lookup = fold_assignments.set_index("scenario_id")["fold_index"]
    aligned_folds = scenario_ids.map(fold_lookup)
    if aligned_folds.isnull().any():
        raise ValueError("Missing fold assignments for some baseline scenarios.")

    X = prepared_data.X.loc[:, feature_columns].reset_index(drop=True)
    y = prepared_data.y.reset_index(drop=True)
    kernel_frames: dict[str, pd.DataFrame] = {}

    for kernel_name in ["matern_white", "matern_white_dot"]:
        viscosity_predictions = np.zeros(len(X), dtype=float)
        oxidation_predictions = np.zeros(len(X), dtype=float)
        viscosity_stds = np.zeros(len(X), dtype=float)
        oxidation_stds = np.zeros(len(X), dtype=float)

        for fold_index in sorted(aligned_folds.unique()):
            train_mask = aligned_folds != fold_index
            valid_mask = aligned_folds == fold_index

            preprocessor, X_train = _fit_gp_preprocessor(X.loc[train_mask, :])
            X_valid = preprocessor.transform(X.loc[valid_mask, :])
            y_train = y.loc[train_mask, :]

            viscosity_mean, viscosity_std = _fit_gp_target_with_std(
                X_train=X_train,
                y_train=y_train[VISCOSITY_TARGET].to_numpy(dtype=float),
                X_valid=X_valid,
                kernel_name=kernel_name,
            )
            oxidation_mean, oxidation_std = _fit_gp_target_with_std(
                X_train=X_train,
                y_train=y_train[OXIDATION_TARGET].to_numpy(dtype=float),
                X_valid=X_valid,
                kernel_name=kernel_name,
            )

            viscosity_predictions[valid_mask] = viscosity_mean
            oxidation_predictions[valid_mask] = oxidation_mean
            viscosity_stds[valid_mask] = viscosity_std
            oxidation_stds[valid_mask] = oxidation_std

        source_name = f"gp_{kernel_name}"
        kernel_frames[kernel_name] = pd.DataFrame(
            {
                "scenario_id": scenario_ids.to_numpy(),
                "fold_index": aligned_folds.to_numpy(dtype=int),
                _prediction_column(source_name, VISCOSITY_TARGET): viscosity_predictions,
                _prediction_column(source_name, OXIDATION_TARGET): oxidation_predictions,
                _std_column(source_name, VISCOSITY_TARGET): viscosity_stds,
                _std_column(source_name, OXIDATION_TARGET): oxidation_stds,
            }
        ).sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)

    return kernel_frames


def _build_current_stack_oof_predictions(frame: pd.DataFrame, gp_source_name: str, stack_source_name: str) -> pd.DataFrame:
    temp_frame = _stack_input_frame(frame=frame, gp_source_name=gp_source_name)
    from src.eval.run_gp_ensemble_stage2 import _build_stack_predictions  # local import to avoid circular churn at module load

    viscosity_predictions, oxidation_predictions = _build_stack_predictions(temp_frame)
    return pd.DataFrame(
        {
            "scenario_id": frame["scenario_id"].to_numpy(),
            "fold_index": frame["fold_index"].to_numpy(dtype=int),
            _prediction_column(stack_source_name, VISCOSITY_TARGET): viscosity_predictions,
            _prediction_column(stack_source_name, OXIDATION_TARGET): oxidation_predictions,
        }
    )


def _compute_family_entropy(frame: pd.DataFrame) -> pd.Series:
    family_share_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("family__") and column.endswith("__mass_share")
    )
    family_mass = frame.loc[:, family_share_columns].to_numpy(dtype=float)
    family_mass = np.clip(family_mass, 0.0, None)
    family_mass_sum = family_mass.sum(axis=1, keepdims=True)
    family_probs = np.divide(
        family_mass,
        family_mass_sum,
        out=np.zeros_like(family_mass),
        where=family_mass_sum > 0,
    )
    entropy = -(family_probs * np.log(np.clip(family_probs, 1e-12, None))).sum(axis=1)
    normalizer = np.log(max(2, family_probs.shape[1]))
    return pd.Series(entropy / normalizer, index=frame.index, name="family_entropy")


def _compute_regime_descriptors() -> pd.DataFrame:
    feature_frame = pd.read_csv(TRAIN_SCENARIO_FEATURES_OUTPUT_PATH).sort_values("scenario_id").reset_index(drop=True)
    descriptors = feature_frame.loc[
        :,
        [
            "scenario_id",
            "component_row_count",
            "component_unique_count",
            "component_family_unique_count",
            "component_batch_unique_count",
            "batch_unique_count",
            "mass_fraction_top1_share",
            "mass_fraction_top3_share",
            "missing_all_props__mass_share",
            "property_join_source__missing__row_ratio",
            "property_cell_nonmissing_density",
            "property_column_nonmissing_any_count",
        ],
    ].copy()
    descriptors["family_entropy"] = _compute_family_entropy(feature_frame)
    descriptors["missingness_burden"] = (
        descriptors["missing_all_props__mass_share"].astype(float)
        + descriptors["property_join_source__missing__row_ratio"].astype(float)
        + (1.0 - descriptors["property_cell_nonmissing_density"].astype(float))
    )

    cluster_input_columns = [
        "component_row_count",
        "component_unique_count",
        "component_family_unique_count",
        "mass_fraction_top1_share",
        "mass_fraction_top3_share",
        "family_entropy",
        "missingness_burden",
        "property_cell_nonmissing_density",
    ]
    scaler = StandardScaler()
    cluster_input = scaler.fit_transform(descriptors.loc[:, cluster_input_columns].to_numpy(dtype=float))

    n_clusters = min(6, max(2, len(descriptors) // 25))
    cluster_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
    descriptors["regime_cluster_label"] = cluster_model.fit_predict(cluster_input)

    neighbor_count = min(6, len(descriptors))
    neighbor_model = NearestNeighbors(n_neighbors=neighbor_count)
    neighbor_model.fit(cluster_input)
    distances, _ = neighbor_model.kneighbors(cluster_input)
    if neighbor_count > 1:
        mean_neighbor_distance = distances[:, 1:].mean(axis=1)
    else:
        mean_neighbor_distance = np.zeros(len(descriptors), dtype=float)
    descriptors["regime_mean_5nn_distance"] = mean_neighbor_distance
    descriptors["regime_local_density_score"] = 1.0 / (1e-6 + mean_neighbor_distance)
    return descriptors.sort_values("scenario_id").reset_index(drop=True)


def _build_base_oof_frame(args: argparse.Namespace) -> pd.DataFrame:
    prepared_data = load_baseline_training_data()
    feature_columns = select_baseline_feature_columns(
        prepared_data=prepared_data,
        feature_setting=BEST_BASELINE_FEATURE_SETTING,
    )
    deep_oof = _collect_stage15_oof_predictions(args).rename(
        columns={
            "deep_sets_viscosity_pred": _prediction_column("stage15", VISCOSITY_TARGET),
            "deep_sets_oxidation_pred": _prediction_column("stage15", OXIDATION_TARGET),
        }
    )

    gp_frames = _collect_gp_oof_predictions_with_uncertainty(
        prepared_data=prepared_data,
        deep_oof=deep_oof.rename(
            columns={
                _prediction_column("stage15", VISCOSITY_TARGET): "deep_sets_viscosity_pred",
                _prediction_column("stage15", OXIDATION_TARGET): "deep_sets_oxidation_pred",
            }
        ),
        feature_columns=feature_columns,
    )

    merged = deep_oof.copy()
    for kernel_name, frame in gp_frames.items():
        merged = merged.merge(frame, on=["scenario_id", "fold_index"], how="inner", validate="one_to_one")

    current_stack = _build_current_stack_oof_predictions(
        frame=merged.rename(
            columns={
                _prediction_column("stage15", VISCOSITY_TARGET): "deep_sets_viscosity_pred",
                _prediction_column("stage15", OXIDATION_TARGET): "deep_sets_oxidation_pred",
            }
        ),
        gp_source_name="gp_matern_white",
        stack_source_name=CURRENT_STACK_SOURCE,
    )
    stack_dot = _build_current_stack_oof_predictions(
        frame=merged.rename(
            columns={
                _prediction_column("stage15", VISCOSITY_TARGET): "deep_sets_viscosity_pred",
                _prediction_column("stage15", OXIDATION_TARGET): "deep_sets_oxidation_pred",
            }
        ),
        gp_source_name="gp_matern_white_dot",
        stack_source_name=STACK_DOT_SOURCE,
    )

    merged = merged.merge(current_stack, on=["scenario_id", "fold_index"], how="inner", validate="one_to_one")
    merged = merged.merge(stack_dot, on=["scenario_id", "fold_index"], how="inner", validate="one_to_one")
    merged = merged.merge(_compute_regime_descriptors(), on="scenario_id", how="inner", validate="one_to_one")
    return merged.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _add_error_and_disagreement_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    source_names = SOURCE_ORDER

    for source_name in source_names:
        for target_name in TARGET_COLUMNS:
            key = _target_key(target_name)
            pred_column = _prediction_column(source_name, target_name)
            true_column = f"{target_name}__true"
            enriched[f"{source_name}_{key}_abs_error"] = (
                enriched[true_column].to_numpy(dtype=float) - enriched[pred_column].to_numpy(dtype=float)
            ).astype(float)
            enriched[f"{source_name}_{key}_abs_error"] = enriched[f"{source_name}_{key}_abs_error"].abs()

    for target_name in TARGET_COLUMNS:
        key = _target_key(target_name)
        prediction_columns = [_prediction_column(source_name, target_name) for source_name in source_names]
        values = enriched.loc[:, prediction_columns].to_numpy(dtype=float)
        enriched[f"{key}_prediction_std_across_sources"] = values.std(axis=1)
        enriched[f"{key}_prediction_range_across_sources"] = values.max(axis=1) - values.min(axis=1)
        pairwise_diffs: list[np.ndarray] = []
        for left_index, left_name in enumerate(source_names):
            for right_name in source_names[left_index + 1 :]:
                left_values = enriched[_prediction_column(left_name, target_name)].to_numpy(dtype=float)
                right_values = enriched[_prediction_column(right_name, target_name)].to_numpy(dtype=float)
                diff = np.abs(left_values - right_values)
                pairwise_diffs.append(diff)
                if left_name in {"stage15", CURRENT_STACK_SOURCE} or right_name in {"stage15", CURRENT_STACK_SOURCE}:
                    enriched[f"{key}_abs_diff__{left_name}__vs__{right_name}"] = diff
        enriched[f"{key}_mean_abs_pairwise_disagreement"] = np.mean(np.vstack(pairwise_diffs), axis=0)

    enriched["combined_prediction_std_across_sources"] = 0.5 * (
        enriched["viscosity_prediction_std_across_sources"] / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
        + enriched["oxidation_prediction_std_across_sources"] / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
    )
    return enriched


def _fit_nonnegative_mae_weights(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    X_array = np.asarray(X_train, dtype=float)
    y_array = np.asarray(y_train, dtype=float)
    n_models = X_array.shape[1]

    if n_models == 1:
        return np.ones(1, dtype=float)

    def _objective(weights: np.ndarray) -> float:
        prediction = X_array @ weights
        return float(np.mean(np.abs(y_array - prediction)))

    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    initial = np.full(n_models, 1.0 / n_models, dtype=float)
    result = minimize(
        _objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if result.success:
        weights = np.asarray(result.x, dtype=float)
        return weights / max(weights.sum(), 1e-12)

    fallback_index = int(np.argmin([mean_absolute_error(y_array, X_array[:, index]) for index in range(n_models)]))
    fallback = np.zeros(n_models, dtype=float)
    fallback[fallback_index] = 1.0
    return fallback


def _enumerate_source_subsets(source_names: list[str]) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for subset_size in range(2, len(source_names) + 1):
        subsets.extend(tuple(subset) for subset in combinations(source_names, subset_size))
    return subsets


def _crossfit_target_blend(
    frame: pd.DataFrame,
    target_name: str,
    source_subset: tuple[str, ...],
) -> dict[str, object]:
    prediction_columns = [_prediction_column(source_name, target_name) for source_name in source_subset]
    target_values = frame[f"{target_name}__true"].to_numpy(dtype=float)
    fold_predictions = np.zeros(len(frame), dtype=float)
    fold_weights: list[list[float]] = []

    for fold_index in sorted(frame["fold_index"].unique()):
        train_mask = frame["fold_index"] != fold_index
        valid_mask = frame["fold_index"] == fold_index
        X_train = frame.loc[train_mask, prediction_columns].to_numpy(dtype=float)
        X_valid = frame.loc[valid_mask, prediction_columns].to_numpy(dtype=float)
        y_train = frame.loc[train_mask, f"{target_name}__true"].to_numpy(dtype=float)
        weights = _fit_nonnegative_mae_weights(X_train=X_train, y_train=y_train)
        fold_predictions[valid_mask] = X_valid @ weights
        fold_weights.append(weights.tolist())

    full_weights = _fit_nonnegative_mae_weights(
        X_train=frame.loc[:, prediction_columns].to_numpy(dtype=float),
        y_train=target_values,
    )
    target_mae = float(mean_absolute_error(target_values, fold_predictions))
    return {
        "source_subset": source_subset,
        "prediction_columns": prediction_columns,
        "oof_predictions": fold_predictions,
        "fold_weights": fold_weights,
        "full_weights": full_weights,
        "target_mae": target_mae,
        "subset_tag": "+".join(source_subset),
    }


def _evaluate_candidate_predictions(
    frame: pd.DataFrame,
    candidate_name: str,
    candidate_family: str,
    viscosity_pred: np.ndarray,
    oxidation_pred: np.ndarray,
    viscosity_sources: str,
    oxidation_sources: str,
    viscosity_weights_json: str,
    oxidation_weights_json: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frame = frame.loc[
        :,
        [
            "scenario_id",
            "fold_index",
            "viscosity_scale",
            "oxidation_scale",
            f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true",
        ],
    ].copy()
    prediction_frame[f"{VISCOSITY_TARGET}__pred"] = np.asarray(viscosity_pred, dtype=float)
    prediction_frame[f"{OXIDATION_TARGET}__pred"] = np.asarray(oxidation_pred, dtype=float)

    fold_records: list[dict[str, object]] = []
    for fold_index, fold_frame in prediction_frame.groupby("fold_index", dropna=False):
        regression_metrics = evaluate_regression_predictions(
            y_true=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__pred", f"{OXIDATION_TARGET}__pred"]].to_numpy(dtype=float),
            target_names=TARGET_COLUMNS,
            target_scales={
                VISCOSITY_TARGET: float(fold_frame["viscosity_scale"].iloc[0]),
                OXIDATION_TARGET: float(fold_frame["oxidation_scale"].iloc[0]),
            },
        )
        platform_metrics = evaluate_platform_predictions(
            y_true=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__true", f"{OXIDATION_TARGET}__true"]].to_numpy(dtype=float),
            y_pred=fold_frame.loc[:, [f"{VISCOSITY_TARGET}__pred", f"{OXIDATION_TARGET}__pred"]].to_numpy(dtype=float),
            target_names=TARGET_COLUMNS,
        )
        fold_records.append(
            {
                "candidate_name": candidate_name,
                "candidate_family": candidate_family,
                "fold_index": int(fold_index),
                "viscosity_sources": viscosity_sources,
                "oxidation_sources": oxidation_sources,
                "viscosity_weights_json": viscosity_weights_json,
                "oxidation_weights_json": oxidation_weights_json,
                **regression_metrics,
                **platform_metrics,
            }
        )

    fold_metrics = pd.DataFrame.from_records(fold_records)
    summary = (
        fold_metrics.groupby(
            [
                "candidate_name",
                "candidate_family",
                "viscosity_sources",
                "oxidation_sources",
                "viscosity_weights_json",
                "oxidation_weights_json",
            ],
            dropna=False,
        )[
            [
                "combined_score",
                "platform_score",
                f"{VISCOSITY_TARGET}__mae",
                f"{OXIDATION_TARGET}__mae",
                f"{VISCOSITY_TARGET}__platform_mae",
                f"{OXIDATION_TARGET}__platform_mae",
                f"{VISCOSITY_TARGET}__platform_nmae",
                f"{OXIDATION_TARGET}__platform_nmae",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return fold_metrics, summary


def _run_meta_stack_search(
    frame: pd.DataFrame,
    top_target_blends: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, dict[str, dict[str, object]]]]:
    candidate_summaries: list[pd.DataFrame] = []
    candidate_predictions: dict[str, dict[str, object]] = {}

    anchor_name = STAGE15_CANDIDATE_NAME
    _, anchor_summary = _evaluate_candidate_predictions(
        frame=frame,
        candidate_name=anchor_name,
        candidate_family="reference_anchor",
        viscosity_pred=frame[_prediction_column("stage15", VISCOSITY_TARGET)].to_numpy(dtype=float),
        oxidation_pred=frame[_prediction_column("stage15", OXIDATION_TARGET)].to_numpy(dtype=float),
        viscosity_sources="stage15",
        oxidation_sources="stage15",
        viscosity_weights_json=json.dumps({"stage15": 1.0}, sort_keys=True),
        oxidation_weights_json=json.dumps({"stage15": 1.0}, sort_keys=True),
    )
    candidate_summaries.append(anchor_summary)
    candidate_predictions[anchor_name] = {
        "candidate_family": "reference_anchor",
        "viscosity_pred": frame[_prediction_column("stage15", VISCOSITY_TARGET)].to_numpy(dtype=float),
        "oxidation_pred": frame[_prediction_column("stage15", OXIDATION_TARGET)].to_numpy(dtype=float),
        "viscosity_sources": ("stage15",),
        "oxidation_sources": ("stage15",),
        "viscosity_full_weights": np.asarray([1.0], dtype=float),
        "oxidation_full_weights": np.asarray([1.0], dtype=float),
    }

    for viscosity_source, oxidation_source in product(SOURCE_ORDER, repeat=2):
        candidate_name = f"assembly__visc__{viscosity_source}__ox__{oxidation_source}"
        _, summary = _evaluate_candidate_predictions(
            frame=frame,
            candidate_name=candidate_name,
            candidate_family="simple_target_assembly",
            viscosity_pred=frame[_prediction_column(viscosity_source, VISCOSITY_TARGET)].to_numpy(dtype=float),
            oxidation_pred=frame[_prediction_column(oxidation_source, OXIDATION_TARGET)].to_numpy(dtype=float),
            viscosity_sources=viscosity_source,
            oxidation_sources=oxidation_source,
            viscosity_weights_json=json.dumps({viscosity_source: 1.0}, sort_keys=True),
            oxidation_weights_json=json.dumps({oxidation_source: 1.0}, sort_keys=True),
        )
        candidate_summaries.append(summary)
        candidate_predictions[candidate_name] = {
            "candidate_family": "simple_target_assembly",
            "viscosity_pred": frame[_prediction_column(viscosity_source, VISCOSITY_TARGET)].to_numpy(dtype=float),
            "oxidation_pred": frame[_prediction_column(oxidation_source, OXIDATION_TARGET)].to_numpy(dtype=float),
            "viscosity_sources": (viscosity_source,),
            "oxidation_sources": (oxidation_source,),
            "viscosity_full_weights": np.asarray([1.0], dtype=float),
            "oxidation_full_weights": np.asarray([1.0], dtype=float),
        }

    target_blend_results: dict[str, dict[str, dict[str, object]]] = {target_name: {} for target_name in TARGET_COLUMNS}
    for target_name in TARGET_COLUMNS:
        for source_subset in _enumerate_source_subsets(SOURCE_ORDER):
            result = _crossfit_target_blend(frame=frame, target_name=target_name, source_subset=source_subset)
            target_blend_results[target_name][result["subset_tag"]] = result

    top_viscosity = sorted(
        target_blend_results[VISCOSITY_TARGET].values(),
        key=lambda record: record["target_mae"],
    )[:top_target_blends]
    top_oxidation = sorted(
        target_blend_results[OXIDATION_TARGET].values(),
        key=lambda record: record["target_mae"],
    )[:top_target_blends]

    for viscosity_result, oxidation_result in product(top_viscosity, top_oxidation):
        candidate_name = (
            f"blend__visc__{viscosity_result['subset_tag']}__ox__{oxidation_result['subset_tag']}"
        )
        viscosity_weights = {
            source_name: round(float(weight), 8)
            for source_name, weight in zip(viscosity_result["source_subset"], viscosity_result["full_weights"], strict=False)
        }
        oxidation_weights = {
            source_name: round(float(weight), 8)
            for source_name, weight in zip(oxidation_result["source_subset"], oxidation_result["full_weights"], strict=False)
        }
        _, summary = _evaluate_candidate_predictions(
            frame=frame,
            candidate_name=candidate_name,
            candidate_family="nonnegative_mae_blend",
            viscosity_pred=viscosity_result["oof_predictions"],
            oxidation_pred=oxidation_result["oof_predictions"],
            viscosity_sources="+".join(viscosity_result["source_subset"]),
            oxidation_sources="+".join(oxidation_result["source_subset"]),
            viscosity_weights_json=json.dumps(viscosity_weights, sort_keys=True),
            oxidation_weights_json=json.dumps(oxidation_weights, sort_keys=True),
        )
        candidate_summaries.append(summary)
        candidate_predictions[candidate_name] = {
            "candidate_family": "nonnegative_mae_blend",
            "viscosity_pred": viscosity_result["oof_predictions"],
            "oxidation_pred": oxidation_result["oof_predictions"],
            "viscosity_sources": viscosity_result["source_subset"],
            "oxidation_sources": oxidation_result["source_subset"],
            "viscosity_full_weights": viscosity_result["full_weights"],
            "oxidation_full_weights": oxidation_result["full_weights"],
        }

    results = pd.concat(candidate_summaries, ignore_index=True).sort_values(
        [
            "platform_score__mean",
            f"{VISCOSITY_TARGET}__mae__mean",
            f"{OXIDATION_TARGET}__mae__mean",
        ]
    ).reset_index(drop=True)
    results["rank_platform_score"] = np.arange(1, len(results) + 1)
    return results, candidate_predictions, target_blend_results


def _build_oof_artifact(
    frame: pd.DataFrame,
    candidate_name: str,
    candidate_predictions: dict[str, dict[str, object]],
) -> pd.DataFrame:
    artifact = _add_error_and_disagreement_features(frame)
    best_candidate = candidate_predictions[candidate_name]
    artifact["best_meta_candidate_name"] = candidate_name
    artifact["best_meta_viscosity_pred"] = np.asarray(best_candidate["viscosity_pred"], dtype=float)
    artifact["best_meta_oxidation_pred"] = np.asarray(best_candidate["oxidation_pred"], dtype=float)
    artifact["best_meta_viscosity_abs_error"] = (
        artifact[f"{VISCOSITY_TARGET}__true"] - artifact["best_meta_viscosity_pred"]
    ).abs()
    artifact["best_meta_oxidation_abs_error"] = (
        artifact[f"{OXIDATION_TARGET}__true"] - artifact["best_meta_oxidation_pred"]
    ).abs()
    return artifact.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _paired_bootstrap_summary(
    frame: pd.DataFrame,
    anchor_name: str,
    candidate_name: str,
    candidate_predictions: dict[str, dict[str, object]],
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = len(frame)
    anchor = candidate_predictions[anchor_name]
    candidate = candidate_predictions[candidate_name]

    true_viscosity = frame[f"{VISCOSITY_TARGET}__true"].to_numpy(dtype=float)
    true_oxidation = frame[f"{OXIDATION_TARGET}__true"].to_numpy(dtype=float)
    anchor_viscosity = np.asarray(anchor["viscosity_pred"], dtype=float)
    anchor_oxidation = np.asarray(anchor["oxidation_pred"], dtype=float)
    candidate_viscosity = np.asarray(candidate["viscosity_pred"], dtype=float)
    candidate_oxidation = np.asarray(candidate["oxidation_pred"], dtype=float)

    platform_deltas = np.zeros(bootstrap_resamples, dtype=float)
    viscosity_mae_deltas = np.zeros(bootstrap_resamples, dtype=float)
    oxidation_mae_deltas = np.zeros(bootstrap_resamples, dtype=float)

    for index in range(bootstrap_resamples):
        sample_index = rng.integers(0, n_rows, size=n_rows)

        anchor_visc_mae = mean_absolute_error(true_viscosity[sample_index], anchor_viscosity[sample_index])
        anchor_ox_mae = mean_absolute_error(true_oxidation[sample_index], anchor_oxidation[sample_index])
        candidate_visc_mae = mean_absolute_error(true_viscosity[sample_index], candidate_viscosity[sample_index])
        candidate_ox_mae = mean_absolute_error(true_oxidation[sample_index], candidate_oxidation[sample_index])

        anchor_score = 0.5 * (
            anchor_visc_mae / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
            + anchor_ox_mae / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
        )
        candidate_score = 0.5 * (
            candidate_visc_mae / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
            + candidate_ox_mae / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
        )

        platform_deltas[index] = float(candidate_score - anchor_score)
        viscosity_mae_deltas[index] = float(candidate_visc_mae - anchor_visc_mae)
        oxidation_mae_deltas[index] = float(candidate_ox_mae - anchor_ox_mae)

    summary = pd.DataFrame.from_records(
        [
            {
                "anchor_candidate": anchor_name,
                "comparison_candidate": candidate_name,
                "bootstrap_resamples": bootstrap_resamples,
                "platform_delta__median": float(np.median(platform_deltas)),
                "platform_delta__q025": float(np.quantile(platform_deltas, 0.025)),
                "platform_delta__q975": float(np.quantile(platform_deltas, 0.975)),
                "probability_of_improvement": float(np.mean(platform_deltas < 0.0)),
                "viscosity_mae_delta__median": float(np.median(viscosity_mae_deltas)),
                "oxidation_mae_delta__median": float(np.median(oxidation_mae_deltas)),
                "probability_of_no_oxidation_regression": float(np.mean(oxidation_mae_deltas <= 0.0)),
            }
        ]
    )
    return summary


def _build_regime_audit_report(
    frame: pd.DataFrame,
    best_candidate_name: str,
) -> str:
    working = frame.copy()
    working["anchor_platform_component"] = 0.5 * (
        working["stage15_viscosity_abs_error"] / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
        + working["stage15_oxidation_abs_error"] / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
    )
    working["current_stack_platform_component"] = 0.5 * (
        working[f"{CURRENT_STACK_SOURCE}_viscosity_abs_error"] / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
        + working[f"{CURRENT_STACK_SOURCE}_oxidation_abs_error"] / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
    )
    working["best_meta_platform_component"] = 0.5 * (
        working["best_meta_viscosity_abs_error"] / PLATFORM_TARGET_SCALES[VISCOSITY_TARGET]
        + working["best_meta_oxidation_abs_error"] / PLATFORM_TARGET_SCALES[OXIDATION_TARGET]
    )
    working["best_meta_delta_vs_anchor"] = (
        working["best_meta_platform_component"] - working["anchor_platform_component"]
    )

    regime_specs = [
        ("high_component_count", working["component_row_count"] >= working["component_row_count"].quantile(0.75)),
        ("high_family_entropy", working["family_entropy"] >= working["family_entropy"].quantile(0.75)),
        ("high_missingness_burden", working["missingness_burden"] >= working["missingness_burden"].quantile(0.75)),
        ("low_local_density", working["regime_local_density_score"] <= working["regime_local_density_score"].quantile(0.25)),
        (
            "high_model_disagreement",
            working["combined_prediction_std_across_sources"] >= working["combined_prediction_std_across_sources"].quantile(0.75),
        ),
    ]

    records: list[dict[str, object]] = []
    for regime_name, mask in regime_specs:
        subset = working.loc[mask].copy()
        if subset.empty:
            continue
        records.append(
            {
                "regime": regime_name,
                "rows": int(len(subset)),
                "anchor_platform_component": float(subset["anchor_platform_component"].mean()),
                "current_stack_platform_component": float(subset["current_stack_platform_component"].mean()),
                "best_meta_platform_component": float(subset["best_meta_platform_component"].mean()),
                "best_meta_delta_vs_anchor": float(subset["best_meta_delta_vs_anchor"].mean()),
            }
        )

    largest_gains = working.nsmallest(5, "best_meta_delta_vs_anchor")[
        ["scenario_id", "best_meta_delta_vs_anchor", "component_row_count", "family_entropy", "missingness_burden"]
    ]
    largest_losses = working.nlargest(5, "best_meta_delta_vs_anchor")[
        ["scenario_id", "best_meta_delta_vs_anchor", "component_row_count", "family_entropy", "missingness_burden"]
    ]

    lines = [
        "# GP Stage 2 Regime Audit",
        "",
        "## Descriptor Layer",
        "- Included row-level OOF targets, Stage 1.5 predictions, both GP kernels, the current Deep+GP stack, best-meta predictions, absolute errors, disagreement features, GP uncertainty, and regime descriptors.",
        "- Regime descriptors include counts, family entropy, missingness burden, a k-means cluster label, and a 5-nearest-neighbor density proxy.",
        "",
        "## Regime Summary",
        "",
        "```text",
        pd.DataFrame.from_records(records).to_string(index=False),
        "```",
        "",
        "## Largest Gains",
        "",
        "```text",
        largest_gains.to_string(index=False),
        "```",
        "",
        "## Largest Losses",
        "",
        "```text",
        largest_losses.to_string(index=False),
        "```",
        "",
        "## Takeaway",
        f"- Best meta candidate under audit: `{best_candidate_name}`.",
        "- Negative `best_meta_delta_vs_anchor` means the meta candidate improved on the local Stage 1.5 anchor.",
    ]
    return "\n".join(lines) + "\n"


def _build_meta_stack_report(
    results: pd.DataFrame,
    best_new_candidate_name: str,
    local_gain: float,
    bootstrap_summary: pd.DataFrame,
    should_package: bool,
    packaged_paths: dict[str, str] | None,
) -> str:
    anchor_row = results.loc[results["candidate_name"] == STAGE15_CANDIDATE_NAME].iloc[0]
    best_row = results.loc[results["candidate_name"] == best_new_candidate_name].iloc[0]
    best_assembly_row = results.loc[results["candidate_family"] == "simple_target_assembly"].iloc[0]
    best_blend_row = results.loc[results["candidate_family"] == "nonnegative_mae_blend"].iloc[0]
    bootstrap_row = bootstrap_summary.iloc[0]

    lines = [
        "# Meta Stack Search Report",
        "",
        "## Candidate Ranking",
        "",
        "```text",
        results.head(15)[
            [
                "rank_platform_score",
                "candidate_name",
                "candidate_family",
                "platform_score__mean",
                "platform_score__std",
                f"{VISCOSITY_TARGET}__mae__mean",
                f"{OXIDATION_TARGET}__mae__mean",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Comparison",
        f"- Local Stage 1.5 anchor: `{anchor_row['platform_score__mean']:.6f}`",
        f"- Best simple assembly: `{best_assembly_row['candidate_name']}` at `{best_assembly_row['platform_score__mean']:.6f}`",
        f"- Best learned non-negative blend: `{best_blend_row['candidate_name']}` at `{best_blend_row['platform_score__mean']:.6f}`",
        f"- Best new candidate overall: `{best_new_candidate_name}` at `{best_row['platform_score__mean']:.6f}`",
        f"- Local gain vs Stage 1.5 anchor: `{local_gain:+.6f}`",
        f"- Bootstrap probability of improvement: `{bootstrap_row['probability_of_improvement']:.2%}`",
        f"- Bootstrap probability of no oxidation regression: `{bootstrap_row['probability_of_no_oxidation_regression']:.2%}`",
        f"- New platform attempt justified: `{'yes' if should_package else 'no'}`",
    ]
    if packaged_paths is not None:
        lines.extend(
            [
                f"- Packaged predictions path: `{packaged_paths['predictions_path']}`",
                f"- Packaged ZIP path: `{packaged_paths['zip_path']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_bootstrap_report(
    bootstrap_summary: pd.DataFrame,
    local_gain: float,
    should_package: bool,
) -> str:
    row = bootstrap_summary.iloc[0]
    lines = [
        "# Paired Bootstrap CI: Stage 1.5 vs Best Meta",
        "",
        f"- Anchor candidate: `{row['anchor_candidate']}`",
        f"- Comparison candidate: `{row['comparison_candidate']}`",
        f"- Bootstrap resamples: `{int(row['bootstrap_resamples'])}`",
        f"- Local gain vs Stage 1.5 anchor: `{local_gain:+.6f}`",
        f"- Median platform delta (candidate - anchor): `{row['platform_delta__median']:+.6f}`",
        f"- 95% CI for platform delta: `[{row['platform_delta__q025']:+.6f}, {row['platform_delta__q975']:+.6f}]`",
        f"- Probability of improvement: `{row['probability_of_improvement']:.2%}`",
        f"- Median viscosity MAE delta: `{row['viscosity_mae_delta__median']:+.6f}`",
        f"- Median oxidation MAE delta: `{row['oxidation_mae_delta__median']:+.6f}`",
        f"- Probability of no oxidation regression: `{row['probability_of_no_oxidation_regression']:.2%}`",
        f"- New platform attempt justified: `{'yes' if should_package else 'no'}`",
    ]
    return "\n".join(lines) + "\n"


def _build_current_stack_test_predictions(
    args: argparse.Namespace,
    stage15_test_frame: pd.DataFrame,
    gp_test_frame: pd.DataFrame,
    gp_source_name: str,
) -> pd.DataFrame:
    oof_stage15 = _collect_stage15_oof_predictions(args).rename(
        columns={
            "deep_sets_viscosity_pred": _prediction_column("stage15", VISCOSITY_TARGET),
            "deep_sets_oxidation_pred": _prediction_column("stage15", OXIDATION_TARGET),
        }
    )
    prepared_data = load_baseline_training_data()
    feature_columns = select_baseline_feature_columns(prepared_data, BEST_BASELINE_FEATURE_SETTING)
    gp_oof_frames = _collect_gp_oof_predictions_with_uncertainty(
        prepared_data=prepared_data,
        deep_oof=oof_stage15.rename(
            columns={
                _prediction_column("stage15", VISCOSITY_TARGET): "deep_sets_viscosity_pred",
                _prediction_column("stage15", OXIDATION_TARGET): "deep_sets_oxidation_pred",
            }
        ),
        feature_columns=feature_columns,
    )
    kernel_key = "matern_white" if gp_source_name == "gp_matern_white" else "matern_white_dot"
    merged_oof = oof_stage15.merge(gp_oof_frames[kernel_key], on=["scenario_id", "fold_index"], how="inner", validate="one_to_one")
    models = _fit_final_stack_models(
        _stack_input_frame(
            frame=merged_oof.rename(
                columns={
                    _prediction_column("stage15", VISCOSITY_TARGET): "deep_sets_viscosity_pred",
                    _prediction_column("stage15", OXIDATION_TARGET): "deep_sets_oxidation_pred",
                }
            ),
            gp_source_name=gp_source_name,
        )
    )
    return _predict_with_final_stack_models(
        deep_frame=stage15_test_frame.rename(
            columns={
                VISCOSITY_TARGET: "deep_sets_viscosity_pred",
                OXIDATION_TARGET: "deep_sets_oxidation_pred",
            }
        ),
        gp_frame=gp_test_frame.rename(
            columns={
                VISCOSITY_TARGET: "gp_viscosity_pred",
                OXIDATION_TARGET: "gp_oxidation_pred",
            }
        ),
        models=models,
    )


def _build_test_source_predictions(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    stage15_test = _fit_full_stage15_deep_sets_predictions(args)
    prepared_data = load_baseline_training_data()
    feature_columns = select_baseline_feature_columns(prepared_data, BEST_BASELINE_FEATURE_SETTING)
    gp_matern_white_test = _fit_full_gp_predictions(
        prepared_data=prepared_data,
        feature_columns=feature_columns,
        kernel_name="matern_white",
    )
    gp_matern_white_dot_test = _fit_full_gp_predictions(
        prepared_data=prepared_data,
        feature_columns=feature_columns,
        kernel_name="matern_white_dot",
    )
    current_stack_test = _build_current_stack_test_predictions(
        args=args,
        stage15_test_frame=stage15_test,
        gp_test_frame=gp_matern_white_test,
        gp_source_name="gp_matern_white",
    )
    stack_dot_test = _build_current_stack_test_predictions(
        args=args,
        stage15_test_frame=stage15_test,
        gp_test_frame=gp_matern_white_dot_test,
        gp_source_name="gp_matern_white_dot",
    )
    return {
        "stage15": stage15_test,
        "gp_matern_white": gp_matern_white_test,
        "gp_matern_white_dot": gp_matern_white_dot_test,
        CURRENT_STACK_SOURCE: current_stack_test,
        STACK_DOT_SOURCE: stack_dot_test,
    }


def _apply_candidate_to_test(
    candidate_name: str,
    candidate_predictions: dict[str, dict[str, object]],
    test_source_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    candidate = candidate_predictions[candidate_name]
    scenario_ids = test_source_predictions["stage15"]["scenario_id"].to_numpy()

    def _blend_target(target_name: str, source_names: tuple[str, ...], weights: np.ndarray) -> np.ndarray:
        stacked = np.column_stack(
            [
                test_source_predictions[source_name][target_name].to_numpy(dtype=float)
                for source_name in source_names
            ]
        )
        return stacked @ np.asarray(weights, dtype=float)

    viscosity = _blend_target(
        target_name=VISCOSITY_TARGET,
        source_names=tuple(candidate["viscosity_sources"]),
        weights=np.asarray(candidate["viscosity_full_weights"], dtype=float),
    )
    oxidation = _blend_target(
        target_name=OXIDATION_TARGET,
        source_names=tuple(candidate["oxidation_sources"]),
        weights=np.asarray(candidate["oxidation_full_weights"], dtype=float),
    )

    return pd.DataFrame(
        {
            "scenario_id": scenario_ids,
            TARGET_COLUMNS[0]: viscosity,
            TARGET_COLUMNS[1]: oxidation,
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _should_package_candidate(
    local_gain: float,
    best_row: pd.Series,
    anchor_row: pd.Series,
    bootstrap_summary: pd.DataFrame,
    strong_bootstrap_probability: float,
) -> bool:
    if local_gain >= 0.0020:
        return True

    bootstrap_row = bootstrap_summary.iloc[0]
    oxidation_regression = float(best_row[f"{OXIDATION_TARGET}__mae__mean"]) > float(anchor_row[f"{OXIDATION_TARGET}__mae__mean"])
    strong_bootstrap_support = (
        float(bootstrap_row["probability_of_improvement"]) >= strong_bootstrap_probability
        and float(bootstrap_row["platform_delta__q975"]) < 0.0
    )
    return 0.0010 <= local_gain < 0.0020 and strong_bootstrap_support and not oxidation_regression


def _package_best_candidate(
    args: argparse.Namespace,
    candidate_name: str,
    candidate_predictions: dict[str, dict[str, object]],
) -> dict[str, str]:
    test_source_predictions = _build_test_source_predictions(args)
    final_predictions = _apply_candidate_to_test(
        candidate_name=candidate_name,
        candidate_predictions=candidate_predictions,
        test_source_predictions=test_source_predictions,
    )
    candidate_slug = candidate_name.replace("__", "_")
    candidate_stem = f"{PACKAGE_STEM_PREFIX}_{candidate_slug}"
    candidate_dir = OUTPUTS_DIR / "submissions" / candidate_stem
    predictions_path = candidate_dir / "predictions.csv"
    zip_name = f"{candidate_stem}.zip"

    candidate_dir.mkdir(parents=True, exist_ok=True)
    submission_frame = _to_submission_columns(final_predictions)
    submission_frame.to_csv(predictions_path, index=False, encoding="utf-8")
    validate_predictions_csv(predictions_path)
    zip_path = build_bundle_from_predictions(predictions_path=predictions_path, zip_name=zip_name)
    return {
        "predictions_path": str(predictions_path),
        "zip_path": str(zip_path),
    }


def main() -> None:
    args = parse_args()
    set_config(enable_metadata_routing=True)
    CV_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_oof = _build_base_oof_frame(args)
    meta_results, candidate_predictions, _ = _run_meta_stack_search(
        frame=base_oof,
        top_target_blends=args.top_target_blends,
    )

    anchor_row = meta_results.loc[meta_results["candidate_name"] == STAGE15_CANDIDATE_NAME].iloc[0]
    best_new_row = meta_results.loc[meta_results["candidate_name"] != STAGE15_CANDIDATE_NAME].iloc[0]
    best_new_candidate_name = str(best_new_row["candidate_name"])
    local_gain = float(anchor_row["platform_score__mean"] - best_new_row["platform_score__mean"])

    oof_artifact = _build_oof_artifact(
        frame=base_oof,
        candidate_name=best_new_candidate_name,
        candidate_predictions=candidate_predictions,
    )
    bootstrap_summary = _paired_bootstrap_summary(
        frame=oof_artifact,
        anchor_name=STAGE15_CANDIDATE_NAME,
        candidate_name=best_new_candidate_name,
        candidate_predictions=candidate_predictions,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    should_package = _should_package_candidate(
        local_gain=local_gain,
        best_row=best_new_row,
        anchor_row=anchor_row,
        bootstrap_summary=bootstrap_summary,
        strong_bootstrap_probability=args.strong_bootstrap_probability,
    )
    packaged_paths = _package_best_candidate(
        args=args,
        candidate_name=best_new_candidate_name,
        candidate_predictions=candidate_predictions,
    ) if should_package else None

    regime_report = _build_regime_audit_report(
        frame=oof_artifact,
        best_candidate_name=best_new_candidate_name,
    )
    meta_report = _build_meta_stack_report(
        results=meta_results,
        best_new_candidate_name=best_new_candidate_name,
        local_gain=local_gain,
        bootstrap_summary=bootstrap_summary,
        should_package=should_package,
        packaged_paths=packaged_paths,
    )
    bootstrap_report = _build_bootstrap_report(
        bootstrap_summary=bootstrap_summary,
        local_gain=local_gain,
        should_package=should_package,
    )

    _write_csv(oof_artifact, GP_STAGE2_OOF_PREDICTIONS_OUTPUT_PATH)
    _write_csv(meta_results, META_STACK_SEARCH_RESULTS_OUTPUT_PATH)
    _write_csv(bootstrap_summary, PAIRED_BOOTSTRAP_CI_OUTPUT_PATH)
    _write_text(regime_report, GP_STAGE2_REGIME_AUDIT_REPORT_OUTPUT_PATH)
    _write_text(meta_report, META_STACK_SEARCH_REPORT_OUTPUT_PATH)
    _write_text(bootstrap_report, PAIRED_BOOTSTRAP_CI_REPORT_OUTPUT_PATH)

    print(f"gp_stage2_oof_predictions: {GP_STAGE2_OOF_PREDICTIONS_OUTPUT_PATH}")
    print(f"meta_stack_search_results: {META_STACK_SEARCH_RESULTS_OUTPUT_PATH}")
    print(f"paired_bootstrap_ci: {PAIRED_BOOTSTRAP_CI_OUTPUT_PATH}")
    print(f"gp_stage2_regime_audit_report: {GP_STAGE2_REGIME_AUDIT_REPORT_OUTPUT_PATH}")
    print(f"meta_stack_search_report: {META_STACK_SEARCH_REPORT_OUTPUT_PATH}")
    print(f"paired_bootstrap_ci_report: {PAIRED_BOOTSTRAP_CI_REPORT_OUTPUT_PATH}")
    print(f"best_meta_candidate: {best_new_candidate_name}")
    print(f"local_gain_vs_stage15: {local_gain:.6f}")
    print(f"bootstrap_probability_of_improvement: {bootstrap_summary.iloc[0]['probability_of_improvement']:.6f}")
    print(f"platform_attempt_justified: {'yes' if should_package else 'no'}")
    if packaged_paths is not None:
        print(f"meta_predictions: {packaged_paths['predictions_path']}")
        print(f"meta_zip: {packaged_paths['zip_path']}")


if __name__ == "__main__":
    main()
