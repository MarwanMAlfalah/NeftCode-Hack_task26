"""Scenario-level non-tree baseline training and grouped CV evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from src.config import (
    FEATURE_MANIFEST_OUTPUT_PATH,
    RANDOM_SEED,
    TEST_SCENARIO_FEATURES_OUTPUT_PATH,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
    TRAIN_TARGETS_OUTPUT_PATH,
    TRAIN_TARGET_COLUMNS,
)
from src.eval.metrics import compute_target_scales, evaluate_regression_predictions


VISCOSITY_TARGET = "target_delta_kinematic_viscosity_pct"
OXIDATION_TARGET = "target_oxidation_eot_a_per_cm"
TARGET_COLUMNS = [VISCOSITY_TARGET, OXIDATION_TARGET]
FEATURE_SETTING_TO_GROUPS = {
    "conditions_only": ["scenario_conditions"],
    "conditions_structure": ["scenario_conditions", "structure_and_mass"],
    "conditions_structure_family": ["scenario_conditions", "structure_and_mass", "component_families"],
    "conditions_structure_family_coverage": [
        "scenario_conditions",
        "structure_and_mass",
        "component_families",
        "coverage_and_missingness",
    ],
    "full_feature_set": [
        "scenario_conditions",
        "structure_and_mass",
        "component_families",
        "coverage_and_missingness",
        "weighted_numeric_properties",
    ],
}


@dataclass(frozen=True)
class PreparedBaselineData:
    """Scenario-level features, targets, and metadata for CV evaluation."""

    scenario_ids: pd.Series
    X: pd.DataFrame
    y: pd.DataFrame
    feature_manifest: dict[str, object]


@dataclass(frozen=True)
class TargetStrategy:
    """Target-space transformation strategy used during model fitting."""

    name: str
    description: str
    transform: Callable[[np.ndarray], np.ndarray]
    inverse_transform: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """A single baseline family plus its outer-fold tuning space."""

    name: str
    description: str
    build_estimator: Callable[[int], Pipeline]
    build_param_grid: Callable[[pd.DataFrame, np.ndarray], list[dict[str, object]]]


@dataclass(frozen=True)
class GroupedCVArtifacts:
    """Per-fold metrics plus out-of-fold predictions for one configuration."""

    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame


def load_baseline_training_data(
    train_features_path: Path = TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
    train_targets_path: Path = TRAIN_TARGETS_OUTPUT_PATH,
    feature_manifest_path: Path = FEATURE_MANIFEST_OUTPUT_PATH,
) -> PreparedBaselineData:
    """Load and validate the processed scenario-level feature and target tables."""

    train_features = pd.read_csv(train_features_path)
    train_targets = pd.read_csv(train_targets_path)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))

    if train_features["scenario_id"].duplicated().any():
        raise ValueError("Train features contain duplicate scenario_id values.")
    if train_targets["scenario_id"].duplicated().any():
        raise ValueError("Train targets contain duplicate scenario_id values.")

    merged = train_features.merge(
        train_targets.loc[:, ["scenario_id", *TARGET_COLUMNS]],
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("scenario_id").reset_index(drop=True)

    feature_target_overlap = [column for column in train_features.columns if column.startswith("target_")]
    if feature_target_overlap:
        raise ValueError(
            "Train feature table unexpectedly contains target columns: "
            + ", ".join(sorted(feature_target_overlap))
        )

    scenario_ids = merged["scenario_id"].copy()
    X = merged.drop(columns=["scenario_id", *TARGET_COLUMNS]).astype(float).copy()
    y = merged.loc[:, TARGET_COLUMNS].astype(float).copy()

    return PreparedBaselineData(
        scenario_ids=scenario_ids,
        X=X,
        y=y,
        feature_manifest=feature_manifest,
    )


def load_test_feature_table(path: Path = TEST_SCENARIO_FEATURES_OUTPUT_PATH) -> pd.DataFrame:
    """Load the processed test scenario feature table for schema validation."""

    frame = pd.read_csv(path)
    feature_columns = [column for column in frame.columns if column != "scenario_id"]
    frame = frame.astype({column: float for column in feature_columns}, copy=False)
    return frame


def select_baseline_feature_columns(
    prepared_data: PreparedBaselineData,
    feature_setting: str,
) -> list[str]:
    """Return one named feature subset from the saved manifest."""

    if feature_setting not in FEATURE_SETTING_TO_GROUPS:
        raise KeyError(f"Unknown baseline feature setting: {feature_setting}")

    manifest = prepared_data.feature_manifest["feature_group_columns"]
    columns: list[str] = []
    for group_name in FEATURE_SETTING_TO_GROUPS[feature_setting]:
        if group_name not in manifest:
            raise KeyError(f"Missing feature group `{group_name}` in manifest.")
        columns.extend(manifest[group_name])
    return list(dict.fromkeys(columns))


def build_target_strategies() -> list[TargetStrategy]:
    """Return the supported viscosity-target strategies."""

    def _identity(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def _viscosity_asinh(values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=float).copy()
        transformed[:, 0] = np.arcsinh(transformed[:, 0])
        return transformed

    def _viscosity_sinh(values: np.ndarray) -> np.ndarray:
        restored = np.asarray(values, dtype=float).copy()
        restored[:, 0] = np.sinh(restored[:, 0])
        return restored

    def _viscosity_log1p_signed(values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=float).copy()
        transformed[:, 0] = np.sign(transformed[:, 0]) * np.log1p(np.abs(transformed[:, 0]))
        return transformed

    def _viscosity_expm1_signed(values: np.ndarray) -> np.ndarray:
        restored = np.asarray(values, dtype=float).copy()
        restored[:, 0] = np.sign(restored[:, 0]) * np.expm1(np.abs(restored[:, 0]))
        return restored

    return [
        TargetStrategy(
            name="raw",
            description="Train both targets in their original scale.",
            transform=_identity,
            inverse_transform=_identity,
        ),
        TargetStrategy(
            name="viscosity_asinh",
            description="Train viscosity on asinh scale while keeping oxidation unchanged.",
            transform=_viscosity_asinh,
            inverse_transform=_viscosity_sinh,
        ),
        TargetStrategy(
            name="viscosity_log1p_signed",
            description="Train viscosity on signed log1p scale while keeping oxidation unchanged.",
            transform=_viscosity_log1p_signed,
            inverse_transform=_viscosity_expm1_signed,
        ),
    ]


def _pca_options() -> list[object]:
    """Return reusable PCA search options for linear baselines."""

    return [
        "passthrough",
        PCA(n_components=0.95, svd_solver="full"),
        PCA(n_components=0.99, svd_solver="full"),
    ]


def build_model_specs(include_mlp: bool = False) -> list[ModelSpec]:
    """Construct the required non-tree baseline families."""

    def _ridge_estimator(seed: int) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("reduce_dim", "passthrough"),
                ("model", MultiOutputRegressor(Ridge())),
            ]
        )

    def _ridge_grid(_: pd.DataFrame, __: np.ndarray) -> list[dict[str, object]]:
        return [
            {
                "reduce_dim": ["passthrough", PCA(n_components=0.95, svd_solver="full")],
                "model__estimator__alpha": [0.3, 1.0, 10.0, 30.0],
            }
        ]

    def _pls_estimator(seed: int) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("model", PLSRegression(scale=False)),
            ]
        )

    def _pls_grid(X_train: pd.DataFrame, _: np.ndarray) -> list[dict[str, object]]:
        max_components = max(1, min(X_train.shape[0] - 1, X_train.shape[1], 32))
        candidate_components = [1, 2, 4, 8, 12, 16]
        components = sorted({value for value in candidate_components if value <= max_components})
        return [{"model__n_components": components}]

    def _elasticnet_estimator(seed: int) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("reduce_dim", "passthrough"),
                (
                    "model",
                    MultiTaskElasticNet(
                        max_iter=50000,
                        tol=1e-3,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def _elasticnet_grid(_: pd.DataFrame, __: np.ndarray) -> list[dict[str, object]]:
        return [
            {
                "reduce_dim": ["passthrough"],
                "model__alpha": [0.1, 1.0],
                "model__l1_ratio": [0.2, 0.8],
            }
        ]

    specs = [
        ModelSpec(
            name="ridge_multioutput",
            description="Independent L2-regularized linear models after shared preprocessing.",
            build_estimator=_ridge_estimator,
            build_param_grid=_ridge_grid,
        ),
        ModelSpec(
            name="pls_regression",
            description="Latent-factor multi-output regression with tuned component count.",
            build_estimator=_pls_estimator,
            build_param_grid=_pls_grid,
        ),
        ModelSpec(
            name="multitask_elasticnet",
            description="Sparse shared-coefficient linear baseline using native multi-output ElasticNet.",
            build_estimator=_elasticnet_estimator,
            build_param_grid=_elasticnet_grid,
        ),
    ]

    if include_mlp:
        def _mlp_estimator(seed: int) -> Pipeline:
            return Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value=0.0,
                            add_indicator=True,
                            keep_empty_features=True,
                        ),
                    ),
                    ("variance", VarianceThreshold()),
                    ("scaler", StandardScaler()),
                    ("reduce_dim", PCA(n_components=0.95, svd_solver="full")),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            early_stopping=True,
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )

        def _mlp_grid(_: pd.DataFrame, __: np.ndarray) -> list[dict[str, object]]:
            return [
                {
                    "reduce_dim": [
                        PCA(n_components=0.90, svd_solver="full"),
                        PCA(n_components=0.95, svd_solver="full"),
                    ],
                    "model__hidden_layer_sizes": [(32,), (64, 32)],
                    "model__alpha": [1e-4, 1e-3],
                }
            ]

        specs.append(
            ModelSpec(
                name="compact_mlp",
                description="Optional small MLP with PCA compression and early stopping.",
                build_estimator=_mlp_estimator,
                build_param_grid=_mlp_grid,
            )
        )

    return specs


def get_model_spec_by_name(model_name: str, include_mlp: bool = False) -> ModelSpec:
    """Return a baseline model spec by its stable name."""

    for spec in build_model_specs(include_mlp=include_mlp):
        if spec.name == model_name:
            return spec
    raise KeyError(f"Unknown baseline model spec: {model_name}")


def get_target_strategy_by_name(target_strategy_name: str) -> TargetStrategy:
    """Return a target strategy by its stable name."""

    for strategy in build_target_strategies():
        if strategy.name == target_strategy_name:
            return strategy
    raise KeyError(f"Unknown target strategy: {target_strategy_name}")


def _make_inner_scorer(
    target_strategy: TargetStrategy,
    target_scales: dict[str, float],
) -> Callable[[BaseEstimator, pd.DataFrame, np.ndarray], float]:
    """Build a scorer that evaluates transformed-target models on original scale."""

    def _score(estimator: BaseEstimator, X_valid: pd.DataFrame, y_valid_transformed: np.ndarray) -> float:
        y_pred_transformed = np.asarray(estimator.predict(X_valid), dtype=float)
        y_true_raw = target_strategy.inverse_transform(y_valid_transformed)
        y_pred_raw = target_strategy.inverse_transform(y_pred_transformed)
        metrics = evaluate_regression_predictions(
            y_true=y_true_raw,
            y_pred=y_pred_raw,
            target_names=TARGET_COLUMNS,
            target_scales=target_scales,
        )
        return -metrics["combined_score"]

    return _score


def _serialize_best_params(best_params: dict[str, object]) -> str:
    """Serialize potentially non-JSON sklearn objects into a stable JSON string."""

    def _to_serializable(value: object) -> object:
        if isinstance(value, PCA):
            return repr(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    return json.dumps(
        {key: _to_serializable(value) for key, value in best_params.items()},
        sort_keys=True,
    )


def evaluate_single_baseline_configuration(
    prepared_data: PreparedBaselineData,
    model_spec: ModelSpec,
    target_strategy: TargetStrategy,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = RANDOM_SEED,
    extra_metadata: dict[str, object] | None = None,
) -> GroupedCVArtifacts:
    """Run grouped outer CV for one model family and one target strategy."""

    scenario_ids = prepared_data.scenario_ids.reset_index(drop=True)
    groups = scenario_ids.to_numpy()
    X = prepared_data.X.reset_index(drop=True)
    y = prepared_data.y.reset_index(drop=True)
    metadata = extra_metadata or {}

    outer_cv = GroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []

    for fold_index, (train_index, valid_index) in enumerate(
        outer_cv.split(X=X, y=y, groups=groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train_raw = y.iloc[train_index].to_numpy(dtype=float)
        y_valid_raw = y.iloc[valid_index].to_numpy(dtype=float)
        groups_train = groups[train_index]
        valid_scenarios = scenario_ids.iloc[valid_index].to_numpy()

        target_scales = compute_target_scales(y_train_raw, TARGET_COLUMNS)
        inner_n_splits = max(2, min(inner_splits, len(np.unique(groups_train))))
        inner_cv = GroupKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed + fold_index)

        estimator = model_spec.build_estimator(seed + fold_index)
        param_grid = model_spec.build_param_grid(X_train, target_strategy.transform(y_train_raw))
        scorer = _make_inner_scorer(target_strategy=target_strategy, target_scales=target_scales)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scorer,
            cv=inner_cv,
            n_jobs=None,
            refit=True,
            error_score="raise",
        )

        start_time = time.perf_counter()
        search.fit(X_train, target_strategy.transform(y_train_raw), groups=groups_train)
        fit_time = time.perf_counter() - start_time

        y_pred_transformed = np.asarray(search.best_estimator_.predict(X_valid), dtype=float)
        y_pred_raw = target_strategy.inverse_transform(y_pred_transformed)
        metrics = evaluate_regression_predictions(
            y_true=y_valid_raw,
            y_pred=y_pred_raw,
            target_names=TARGET_COLUMNS,
            target_scales=target_scales,
        )

        fold_record = {
            "fold_index": fold_index,
            "model_name": model_spec.name,
            "target_strategy": target_strategy.name,
            "n_train": len(train_index),
            "n_valid": len(valid_index),
            "fit_time_seconds": fit_time,
            "best_inner_cv_score": -float(search.best_score_),
            "best_params_json": _serialize_best_params(search.best_params_),
            "viscosity_scale": target_scales[VISCOSITY_TARGET],
            "oxidation_scale": target_scales[OXIDATION_TARGET],
            **metadata,
        }
        fold_record.update(metrics)
        fold_records.append(fold_record)

        for row_offset, scenario_id in enumerate(valid_scenarios):
            prediction_records.append(
                {
                    "fold_index": fold_index,
                    "model_name": model_spec.name,
                    "target_strategy": target_strategy.name,
                    "scenario_id": scenario_id,
                    f"{VISCOSITY_TARGET}__true": y_valid_raw[row_offset, 0],
                    f"{VISCOSITY_TARGET}__pred": y_pred_raw[row_offset, 0],
                    f"{OXIDATION_TARGET}__true": y_valid_raw[row_offset, 1],
                    f"{OXIDATION_TARGET}__pred": y_pred_raw[row_offset, 1],
                    "viscosity_scale": target_scales[VISCOSITY_TARGET],
                    "oxidation_scale": target_scales[OXIDATION_TARGET],
                    **metadata,
                }
            )

    return GroupedCVArtifacts(
        fold_metrics=pd.DataFrame.from_records(fold_records),
        oof_predictions=pd.DataFrame.from_records(prediction_records),
    )


def aggregate_cv_results(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate outer-fold metrics into a comparison table."""

    metric_columns = [
        column
        for column in fold_metrics.columns
        if column.endswith("__rmse")
        or column.endswith("__mae")
        or column.endswith("__r2")
        or column.endswith("__nrmse_iqr")
        or column in {"combined_score", "combined_r2_mean", "fit_time_seconds", "best_inner_cv_score"}
    ]
    summary = (
        fold_metrics.groupby(["model_name", "target_strategy"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.rename(columns={"model_name": "model_name", "target_strategy": "target_strategy"})
    summary = summary.sort_values(
        ["combined_score__mean", f"{VISCOSITY_TARGET}__rmse__mean", f"{OXIDATION_TARGET}__rmse__mean"]
    ).reset_index(drop=True)
    summary["rank_combined_score"] = np.arange(1, len(summary) + 1)
    return summary


def _build_error_analysis_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute residuals for the best outer-CV configuration."""

    residuals = predictions.copy()
    residuals["viscosity_abs_error"] = (
        residuals[f"{VISCOSITY_TARGET}__true"] - residuals[f"{VISCOSITY_TARGET}__pred"]
    ).abs()
    residuals["oxidation_abs_error"] = (
        residuals[f"{OXIDATION_TARGET}__true"] - residuals[f"{OXIDATION_TARGET}__pred"]
    ).abs()
    residuals["combined_normalized_abs_error"] = (
        residuals["viscosity_abs_error"] / residuals["viscosity_scale"]
        + residuals["oxidation_abs_error"] / residuals["oxidation_scale"]
    ) / 2.0
    return residuals.sort_values("combined_normalized_abs_error", ascending=False)


def build_baseline_report(
    summary_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    best_predictions: pd.DataFrame,
    feature_manifest: dict[str, object],
) -> str:
    """Create a concise markdown report for the baseline comparison."""

    best_row = summary_results.iloc[0]
    comparison_columns = [
        "rank_combined_score",
        "model_name",
        "target_strategy",
        "combined_score__mean",
        f"{VISCOSITY_TARGET}__rmse__mean",
        f"{OXIDATION_TARGET}__rmse__mean",
        f"{VISCOSITY_TARGET}__mae__mean",
        f"{OXIDATION_TARGET}__mae__mean",
    ]

    transform_comparison = summary_results.pivot(
        index="model_name",
        columns="target_strategy",
        values=["combined_score__mean", f"{VISCOSITY_TARGET}__rmse__mean"],
    )

    transform_lines: list[str] = []
    for model_name in sorted(summary_results["model_name"].unique()):
        raw_score = transform_comparison.get(("combined_score__mean", "raw"))
        asinh_score = transform_comparison.get(("combined_score__mean", "viscosity_asinh"))
        raw_rmse = transform_comparison.get((f"{VISCOSITY_TARGET}__rmse__mean", "raw"))
        asinh_rmse = transform_comparison.get((f"{VISCOSITY_TARGET}__rmse__mean", "viscosity_asinh"))
        if raw_score is None or asinh_score is None or raw_rmse is None or asinh_rmse is None:
            continue
        delta_score = asinh_score.loc[model_name] - raw_score.loc[model_name]
        delta_rmse = asinh_rmse.loc[model_name] - raw_rmse.loc[model_name]
        transform_lines.append(
            f"- `{model_name}`: combined score delta `{delta_score:+.4f}`, viscosity RMSE delta `{delta_rmse:+.4f}`"
        )

    hardest_cases = _build_error_analysis_table(best_predictions).head(5)
    hardest_case_lines = []
    for row in hardest_cases.to_dict(orient="records"):
        hardest_case_lines.append(
            f"- `{row['scenario_id']}`: combined normalized abs error `{row['combined_normalized_abs_error']:.3f}`, "
            f"viscosity true/pred `{row[f'{VISCOSITY_TARGET}__true']:.3f}` / `{row[f'{VISCOSITY_TARGET}__pred']:.3f}`, "
            f"oxidation true/pred `{row[f'{OXIDATION_TARGET}__true']:.3f}` / `{row[f'{OXIDATION_TARGET}__pred']:.3f}`"
        )

    lines = [
        "# Baseline CV Report",
        "",
        "## Data Snapshot",
        f"- Train scenarios: `{feature_manifest.get('train_scenarios', 'unknown')}`",
        f"- Feature columns evaluated: `{len(feature_manifest.get('feature_table_columns', [])) - 1}`",
        f"- Feature groups: `{feature_manifest.get('feature_group_column_counts', {})}`",
        "",
        "## Baseline Comparison",
        "",
        "```text",
        summary_results.loc[:, comparison_columns].to_string(index=False),
        "```",
        "",
        "## Best Current Baseline",
        (
            f"- Best configuration: `{best_row['model_name']}` with target strategy `{best_row['target_strategy']}`"
            f" and mean combined score `{best_row['combined_score__mean']:.4f}`"
        ),
        (
            f"- Mean RMSEs: viscosity `{best_row[f'{VISCOSITY_TARGET}__rmse__mean']:.4f}`, "
            f"oxidation `{best_row[f'{OXIDATION_TARGET}__rmse__mean']:.4f}`"
        ),
        "",
        "## Viscosity Transform Effect",
    ]

    if transform_lines:
        lines.extend(transform_lines)
    else:
        lines.append("- No paired raw/asinh comparison was available.")

    fold_dispersion = fold_metrics.groupby(["model_name", "target_strategy"])[
        ["combined_score", f"{VISCOSITY_TARGET}__rmse", f"{OXIDATION_TARGET}__rmse"]
    ].std().sort_values("combined_score", ascending=False)
    most_volatile = fold_dispersion.head(3)

    lines.extend(
        [
            "",
            "## Error Analysis",
            "- The viscosity target remains the main difficulty because it is highly skewed and outlier-heavy.",
            "- Large gaps between exact-key coverage and usable numeric coverage indicate that fallback and missingness still matter materially.",
            "- The hardest scenarios are dominated by large viscosity misses rather than oxidation misses.",
            "",
            "Hardest out-of-fold scenarios for the best baseline:",
            *hardest_case_lines,
            "",
            "Highest fold-to-fold volatility:",
        ]
    )

    for (model_name, target_strategy), row in most_volatile.iterrows():
        lines.append(
            f"- `{model_name}` / `{target_strategy}`: combined score std `{row['combined_score']:.4f}`, "
            f"viscosity RMSE std `{row[f'{VISCOSITY_TARGET}__rmse']:.4f}`, "
            f"oxidation RMSE std `{row[f'{OXIDATION_TARGET}__rmse']:.4f}`"
        )

    return "\n".join(lines) + "\n"


def run_baseline_cv(
    prepared_data: PreparedBaselineData,
    model_specs: list[ModelSpec] | None = None,
    target_strategies: list[TargetStrategy] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run grouped outer CV across all model and target-strategy combinations."""

    specs = model_specs or build_model_specs(include_mlp=False)
    strategies = target_strategies or build_target_strategies()

    scenario_ids = prepared_data.scenario_ids.reset_index(drop=True)
    groups = scenario_ids.to_numpy()
    X = prepared_data.X.reset_index(drop=True)
    y = prepared_data.y.reset_index(drop=True)

    outer_cv = GroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    fold_records: list[dict[str, object]] = []
    oof_prediction_records: list[dict[str, object]] = []

    for fold_index, (train_index, valid_index) in enumerate(
        outer_cv.split(X=X, y=y, groups=groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train_raw = y.iloc[train_index].to_numpy(dtype=float)
        y_valid_raw = y.iloc[valid_index].to_numpy(dtype=float)
        groups_train = groups[train_index]
        valid_scenarios = scenario_ids.iloc[valid_index].to_numpy()

        target_scales = compute_target_scales(y_train_raw, TARGET_COLUMNS)
        inner_n_splits = max(2, min(inner_splits, len(np.unique(groups_train))))
        inner_cv = GroupKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed + fold_index)

        for target_strategy in strategies:
            y_train_transformed = target_strategy.transform(y_train_raw)

            for model_spec in specs:
                estimator = model_spec.build_estimator(seed + fold_index)
                param_grid = model_spec.build_param_grid(X_train, y_train_transformed)
                scorer = _make_inner_scorer(target_strategy=target_strategy, target_scales=target_scales)

                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring=scorer,
                    cv=inner_cv,
                    n_jobs=None,
                    refit=True,
                    error_score="raise",
                )

                start_time = time.perf_counter()
                search.fit(X_train, y_train_transformed, groups=groups_train)
                fit_time = time.perf_counter() - start_time

                y_pred_transformed = np.asarray(search.best_estimator_.predict(X_valid), dtype=float)
                y_pred_raw = target_strategy.inverse_transform(y_pred_transformed)
                metrics = evaluate_regression_predictions(
                    y_true=y_valid_raw,
                    y_pred=y_pred_raw,
                    target_names=TARGET_COLUMNS,
                    target_scales=target_scales,
                )

                fold_record = {
                    "fold_index": fold_index,
                    "model_name": model_spec.name,
                    "target_strategy": target_strategy.name,
                    "n_train": len(train_index),
                    "n_valid": len(valid_index),
                    "fit_time_seconds": fit_time,
                    "best_inner_cv_score": -float(search.best_score_),
                    "best_params_json": _serialize_best_params(search.best_params_),
                    "viscosity_scale": target_scales[VISCOSITY_TARGET],
                    "oxidation_scale": target_scales[OXIDATION_TARGET],
                }
                fold_record.update(metrics)
                fold_records.append(fold_record)

                for row_offset, scenario_id in enumerate(valid_scenarios):
                    oof_prediction_records.append(
                        {
                            "fold_index": fold_index,
                            "model_name": model_spec.name,
                            "target_strategy": target_strategy.name,
                            "scenario_id": scenario_id,
                            f"{VISCOSITY_TARGET}__true": y_valid_raw[row_offset, 0],
                            f"{VISCOSITY_TARGET}__pred": y_pred_raw[row_offset, 0],
                            f"{OXIDATION_TARGET}__true": y_valid_raw[row_offset, 1],
                            f"{OXIDATION_TARGET}__pred": y_pred_raw[row_offset, 1],
                            "viscosity_scale": target_scales[VISCOSITY_TARGET],
                            "oxidation_scale": target_scales[OXIDATION_TARGET],
                        }
                    )

    fold_metrics = pd.DataFrame.from_records(fold_records)
    summary_results = aggregate_cv_results(fold_metrics)

    best_row = summary_results.iloc[0]
    best_predictions = pd.DataFrame.from_records(oof_prediction_records)
    best_predictions = best_predictions.loc[
        (best_predictions["model_name"] == best_row["model_name"])
        & (best_predictions["target_strategy"] == best_row["target_strategy"])
    ].copy()
    best_predictions = best_predictions.sort_values("scenario_id").reset_index(drop=True)

    report_markdown = build_baseline_report(
        summary_results=summary_results,
        fold_metrics=fold_metrics,
        best_predictions=best_predictions,
        feature_manifest=prepared_data.feature_manifest,
    )
    return summary_results, fold_metrics, report_markdown
