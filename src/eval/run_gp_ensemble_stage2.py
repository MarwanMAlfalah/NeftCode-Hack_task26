"""Standalone GP Stage 2 experiment against the Stage 1.5 Deep Sets anchor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import REQUIRED_COLUMNS, build_bundle_from_predictions, validate_predictions_csv
from src.config import (
    CV_OUTPUTS_DIR,
    GP_ENSEMBLE_REPORT_OUTPUT_PATH,
    GP_ENSEMBLE_RESULTS_OUTPUT_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.eval.metrics import evaluate_platform_predictions, evaluate_regression_predictions
from src.models.train_baselines import (
    OXIDATION_TARGET,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    PreparedBaselineData,
    load_baseline_training_data,
    load_test_feature_table,
    select_baseline_feature_columns,
)
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    PREDICTION_COLUMN_MAP,
    evaluate_single_deep_sets_configuration,
    get_target_strategy_by_name,
    load_deep_sets_data,
    train_full_deep_sets_variant_ensemble_and_predict,
)


BEST_BASELINE_FEATURE_SETTING = "conditions_structure_family"
TARGET_STRATEGY_NAME = "raw"
STAGE15_ANCHOR_SCORE = 0.107282
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]
CANDIDATE_STEM_PREFIX = "neftekod_dot_submission_gp_stage2"
STAGE15_CANDIDATE_NAME = "stage15_anchor__deep_sets"


@dataclass(frozen=True)
class GPPreprocessor:
    """Fold-local preprocessing for GP models."""

    imputer: SimpleImputer
    variance: VarianceThreshold
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.imputer.transform(frame)
        values = self.variance.transform(values)
        values = self.scaler.transform(values)
        return np.asarray(values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--package-threshold", type=float, default=0.0020)
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _to_internal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]]: TARGET_COLUMNS[0],
            PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]]: TARGET_COLUMNS[1],
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _to_submission_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            TARGET_COLUMNS[0]: PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]],
            TARGET_COLUMNS[1]: PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]],
        }
    ).loc[:, REQUIRED_COLUMNS]


def _build_stage15_config(args: argparse.Namespace) -> DeepSetsConfig:
    return DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_metric="platform_score",
    )


def _build_stage15_loss() -> LossConfig:
    return LossConfig(
        name="huber_huber",
        use_robust_viscosity_loss=True,
        use_robust_oxidation_loss=True,
        viscosity_delta=1.0,
        oxidation_delta=0.75,
    )


def _build_stage15_variant() -> HybridVariant:
    return HybridVariant(
        name="hybrid_deep_sets_v2_family_only",
        use_component_embedding=False,
        use_tabular_branch=True,
    )


def _collect_stage15_oof_predictions(args: argparse.Namespace) -> pd.DataFrame:
    prepared_data = load_deep_sets_data()
    artifacts = evaluate_single_deep_sets_configuration(
        prepared_data=prepared_data,
        config=_build_stage15_config(args),
        variant=_build_stage15_variant(),
        target_strategy=get_target_strategy_by_name(TARGET_STRATEGY_NAME),
        outer_splits=args.outer_splits,
        seed=RANDOM_SEED,
        loss_config=_build_stage15_loss(),
        extra_metadata={
            "candidate_name": STAGE15_CANDIDATE_NAME,
            "loss_name": "huber_huber",
            "checkpoint_metric": "platform_score",
        },
    )
    return artifacts.oof_predictions.rename(
        columns={
            f"{VISCOSITY_TARGET}__pred": "deep_sets_viscosity_pred",
            f"{OXIDATION_TARGET}__pred": "deep_sets_oxidation_pred",
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
            "deep_sets_viscosity_pred",
            "deep_sets_oxidation_pred",
        ],
    ].sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _build_kernel(kernel_name: str):
    base_kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )
    if kernel_name == "matern_white":
        return base_kernel
    if kernel_name == "matern_white_dot":
        return base_kernel + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    raise KeyError(f"Unknown GP kernel: {kernel_name}")


def _fit_gp_preprocessor(X_train: pd.DataFrame) -> tuple[GPPreprocessor, np.ndarray]:
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=False)
    variance = VarianceThreshold()
    scaler = StandardScaler()

    values = imputer.fit_transform(X_train)
    values = variance.fit_transform(values)
    if values.shape[1] == 0:
        raise ValueError("GP preprocessing removed every feature column.")
    values = scaler.fit_transform(values)
    return GPPreprocessor(imputer=imputer, variance=variance, scaler=scaler), np.asarray(values, dtype=float)


def _fit_gp_target(X_train: np.ndarray, y_train: np.ndarray, kernel_name: str) -> GaussianProcessRegressor:
    model = GaussianProcessRegressor(
        kernel=_build_kernel(kernel_name),
        normalize_y=True,
        n_restarts_optimizer=2,
        alpha=1e-8,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))
    return model


def _collect_gp_oof_predictions(
    prepared_data: PreparedBaselineData,
    deep_oof: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    fold_assignments = deep_oof.loc[:, ["scenario_id", "fold_index"]].sort_values("scenario_id").reset_index(drop=True)
    scenario_ids = prepared_data.scenario_ids.reset_index(drop=True).copy()
    if not scenario_ids.sort_values(ignore_index=True).equals(fold_assignments["scenario_id"]):
        raise ValueError("Deep Sets OOF folds do not align with the baseline scenario IDs.")

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

        for fold_index in sorted(aligned_folds.unique()):
            train_mask = aligned_folds != fold_index
            valid_mask = aligned_folds == fold_index

            preprocessor, X_train = _fit_gp_preprocessor(X.loc[train_mask, :])
            X_valid = preprocessor.transform(X.loc[valid_mask, :])
            y_train = y.loc[train_mask, :]

            viscosity_model = _fit_gp_target(
                X_train=X_train,
                y_train=y_train[VISCOSITY_TARGET].to_numpy(dtype=float),
                kernel_name=kernel_name,
            )
            oxidation_model = _fit_gp_target(
                X_train=X_train,
                y_train=y_train[OXIDATION_TARGET].to_numpy(dtype=float),
                kernel_name=kernel_name,
            )

            viscosity_predictions[valid_mask] = viscosity_model.predict(X_valid)
            oxidation_predictions[valid_mask] = oxidation_model.predict(X_valid)

        kernel_frames[kernel_name] = pd.DataFrame(
            {
                "scenario_id": scenario_ids.to_numpy(),
                "fold_index": aligned_folds.to_numpy(dtype=int),
                "gp_viscosity_pred": viscosity_predictions,
                "gp_oxidation_pred": oxidation_predictions,
            }
        ).sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)

    return kernel_frames


def _fit_target_ridgecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
) -> tuple[StandardScaler, RidgeCV]:
    inner_splits = max(2, min(5, len(np.unique(groups))))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    grouped_cv = GroupKFold(n_splits=inner_splits)
    cv_splits = list(grouped_cv.split(X_scaled, y_train.to_numpy(dtype=float), groups=np.asarray(groups, dtype=object)))
    model = RidgeCV(
        alphas=RIDGE_ALPHA_GRID,
        cv=cv_splits,
    )
    model.fit(X_scaled, y_train.to_numpy(dtype=float))
    return scaler, model


def _build_stack_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    viscosity_predictions = np.zeros(len(frame), dtype=float)
    oxidation_predictions = np.zeros(len(frame), dtype=float)

    for fold_index in sorted(frame["fold_index"].unique()):
        train_mask = frame["fold_index"] != fold_index
        valid_mask = frame["fold_index"] == fold_index

        viscosity_scaler, viscosity_model = _fit_target_ridgecv(
            X_train=frame.loc[train_mask, ["deep_sets_viscosity_pred", "gp_viscosity_pred"]],
            y_train=frame.loc[train_mask, f"{VISCOSITY_TARGET}__true"],
            groups=frame.loc[train_mask, "scenario_id"].to_numpy(),
        )
        oxidation_scaler, oxidation_model = _fit_target_ridgecv(
            X_train=frame.loc[train_mask, ["deep_sets_oxidation_pred", "gp_oxidation_pred"]],
            y_train=frame.loc[train_mask, f"{OXIDATION_TARGET}__true"],
            groups=frame.loc[train_mask, "scenario_id"].to_numpy(),
        )

        viscosity_predictions[valid_mask] = viscosity_model.predict(
            viscosity_scaler.transform(frame.loc[valid_mask, ["deep_sets_viscosity_pred", "gp_viscosity_pred"]])
        )
        oxidation_predictions[valid_mask] = oxidation_model.predict(
            oxidation_scaler.transform(frame.loc[valid_mask, ["deep_sets_oxidation_pred", "gp_oxidation_pred"]])
        )

    return viscosity_predictions, oxidation_predictions


def _fit_final_stack_models(frame: pd.DataFrame) -> dict[str, object]:
    viscosity_scaler, viscosity_model = _fit_target_ridgecv(
        X_train=frame.loc[:, ["deep_sets_viscosity_pred", "gp_viscosity_pred"]],
        y_train=frame.loc[:, f"{VISCOSITY_TARGET}__true"],
        groups=frame.loc[:, "scenario_id"].to_numpy(),
    )
    oxidation_scaler, oxidation_model = _fit_target_ridgecv(
        X_train=frame.loc[:, ["deep_sets_oxidation_pred", "gp_oxidation_pred"]],
        y_train=frame.loc[:, f"{OXIDATION_TARGET}__true"],
        groups=frame.loc[:, "scenario_id"].to_numpy(),
    )
    return {
        "viscosity_scaler": viscosity_scaler,
        "viscosity_model": viscosity_model,
        "oxidation_scaler": oxidation_scaler,
        "oxidation_model": oxidation_model,
    }


def _predict_with_final_stack_models(
    deep_frame: pd.DataFrame,
    gp_frame: pd.DataFrame,
    models: dict[str, object],
) -> pd.DataFrame:
    merged = deep_frame.merge(
        gp_frame,
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("scenario_id").reset_index(drop=True)
    viscosity_predictions = models["viscosity_model"].predict(
        models["viscosity_scaler"].transform(merged.loc[:, ["deep_sets_viscosity_pred", "gp_viscosity_pred"]])
    )
    oxidation_predictions = models["oxidation_model"].predict(
        models["oxidation_scaler"].transform(merged.loc[:, ["deep_sets_oxidation_pred", "gp_oxidation_pred"]])
    )
    return pd.DataFrame(
        {
            "scenario_id": merged["scenario_id"].to_numpy(),
            TARGET_COLUMNS[0]: viscosity_predictions,
            TARGET_COLUMNS[1]: oxidation_predictions,
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _evaluate_candidate(
    frame: pd.DataFrame,
    candidate_name: str,
    kernel_name: str,
    candidate_type: str,
    viscosity_pred: np.ndarray,
    oxidation_pred: np.ndarray,
) -> pd.DataFrame:
    fold_records: list[dict[str, object]] = []

    prediction_frame = frame.loc[
        :,
        [
            "scenario_id",
            "fold_index",
            f"{VISCOSITY_TARGET}__true",
            f"{OXIDATION_TARGET}__true",
            "viscosity_scale",
            "oxidation_scale",
        ],
    ].copy()
    prediction_frame[f"{VISCOSITY_TARGET}__pred"] = np.asarray(viscosity_pred, dtype=float)
    prediction_frame[f"{OXIDATION_TARGET}__pred"] = np.asarray(oxidation_pred, dtype=float)

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
                "kernel_name": kernel_name,
                "candidate_type": candidate_type,
                "fold_index": int(fold_index),
                **regression_metrics,
                **platform_metrics,
            }
        )

    return pd.DataFrame.from_records(fold_records)


def _summarize_candidates(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "combined_score",
        "platform_score",
        f"{VISCOSITY_TARGET}__mae",
        f"{OXIDATION_TARGET}__mae",
        f"{VISCOSITY_TARGET}__platform_mae",
        f"{OXIDATION_TARGET}__platform_mae",
        f"{VISCOSITY_TARGET}__platform_nmae",
        f"{OXIDATION_TARGET}__platform_nmae",
    ]
    summary = (
        fold_metrics.groupby(["candidate_name", "kernel_name", "candidate_type"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "__".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.sort_values(
        [
            "platform_score__mean",
            f"{VISCOSITY_TARGET}__mae__mean",
            f"{OXIDATION_TARGET}__mae__mean",
        ]
    ).reset_index(drop=True)
    summary["rank_platform_score"] = np.arange(1, len(summary) + 1)
    return summary


def _build_report(
    results: pd.DataFrame,
    feature_columns: list[str],
    threshold_score: float,
    packaged_paths: dict[str, str] | None,
) -> str:
    anchor_row = results.loc[results["candidate_name"] == STAGE15_CANDIDATE_NAME].iloc[0]
    best_row = results.iloc[0]
    best_gp_row = results.loc[results["candidate_type"] == "gp_only"].sort_values("platform_score__mean").iloc[0]
    best_stack_row = results.loc[results["candidate_type"] == "stack"].sort_values("platform_score__mean").iloc[0]
    packaging_triggered = packaged_paths is not None

    lines = [
        "# GP Ensemble Stage 2 Report",
        "",
        "## Experiment Setup",
        "- Shipping path remained untouched; this run only produced a standalone experiment and optional challenger bundle.",
        "- Stage 1.5 anchor OOF was rebuilt with `hybrid_deep_sets_v2_family_only / raw / huber_huber / checkpoint_metric=platform_score`.",
        f"- GP used the validated `{BEST_BASELINE_FEATURE_SETTING}` subset with `{len(feature_columns)}` feature columns.",
        "- Compared both GP kernels end-to-end: `Constant * Matern + WhiteKernel` and `Constant * Matern + WhiteKernel + DotProduct`.",
        "- Built target-wise `RidgeCV` stackers over `[deep_sets_pred, gp_pred]` for viscosity and oxidation separately.",
        "- Ranking metric: `0.5 * (visc_MAE / 2439.25 + ox_MAE / 160.62)`.",
        "",
        "## Candidate Comparison",
        "",
        "```text",
        results[
            [
                "rank_platform_score",
                "candidate_name",
                "kernel_name",
                "candidate_type",
                "platform_score__mean",
                "platform_score__std",
                f"{VISCOSITY_TARGET}__mae__mean",
                f"{OXIDATION_TARGET}__mae__mean",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Decision",
        f"- Stage 1.5 anchor platform score: `{STAGE15_ANCHOR_SCORE:.6f}`",
        f"- Packaging threshold score: `{threshold_score:.6f}`",
        f"- Best overall candidate: `{best_row['candidate_name']}` at `{best_row['platform_score__mean']:.6f}`",
        (
            f"- Best GP kernel: `{best_gp_row['kernel_name']}` via `{best_gp_row['candidate_name']}` "
            f"at `{best_gp_row['platform_score__mean']:.6f}`"
        ),
        (
            f"- Best stack result: `{best_stack_row['candidate_name']}` "
            f"at `{best_stack_row['platform_score__mean']:.6f}`"
        ),
        (
            f"- Best overall delta vs Stage 1.5 anchor: "
            f"`{best_row['platform_score__mean'] - STAGE15_ANCHOR_SCORE:+.6f}`"
        ),
        (
            f"- Best stack delta vs Stage 1.5 anchor: "
            f"`{best_stack_row['platform_score__mean'] - STAGE15_ANCHOR_SCORE:+.6f}`"
        ),
        (
            f"- Package separate challenger ZIP: `{'yes' if packaging_triggered else 'no'}`"
        ),
    ]
    if packaging_triggered and packaged_paths is not None:
        lines.extend(
            [
                f"- Packaged predictions path: `{packaged_paths['predictions_path']}`",
                f"- Packaged ZIP path: `{packaged_paths['zip_path']}`",
            ]
        )
    else:
        lines.append("- No challenger ZIP was produced because the strict platform threshold was not met.")

    lines.extend(
        [
            "",
            "## Anchor Comparison",
            f"- Anchor row from this rerun: `{anchor_row['platform_score__mean']:.6f}` mean platform score.",
            f"- GP-only delta vs anchor: `{best_gp_row['platform_score__mean'] - anchor_row['platform_score__mean']:+.6f}`",
            f"- Stack delta vs anchor: `{best_stack_row['platform_score__mean'] - anchor_row['platform_score__mean']:+.6f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _fit_full_gp_predictions(
    prepared_data: PreparedBaselineData,
    feature_columns: list[str],
    kernel_name: str,
) -> pd.DataFrame:
    test_features = load_test_feature_table()
    X_train = prepared_data.X.loc[:, feature_columns].copy()
    y_train = prepared_data.y.copy()
    X_test = test_features.loc[:, feature_columns].copy()

    preprocessor, X_train_scaled = _fit_gp_preprocessor(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    viscosity_model = _fit_gp_target(
        X_train=X_train_scaled,
        y_train=y_train[VISCOSITY_TARGET].to_numpy(dtype=float),
        kernel_name=kernel_name,
    )
    oxidation_model = _fit_gp_target(
        X_train=X_train_scaled,
        y_train=y_train[OXIDATION_TARGET].to_numpy(dtype=float),
        kernel_name=kernel_name,
    )

    return pd.DataFrame(
        {
            "scenario_id": test_features["scenario_id"].to_numpy(),
            TARGET_COLUMNS[0]: viscosity_model.predict(X_test_scaled),
            TARGET_COLUMNS[1]: oxidation_model.predict(X_test_scaled),
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _fit_full_stage15_deep_sets_predictions(args: argparse.Namespace) -> pd.DataFrame:
    prepared_data = load_deep_sets_data()
    return _to_internal_columns(
        train_full_deep_sets_variant_ensemble_and_predict(
            prepared_data=prepared_data,
            variant=_build_stage15_variant(),
            target_strategy_name=TARGET_STRATEGY_NAME,
            seeds=ENSEMBLE_SEEDS,
            config=_build_stage15_config(args),
            loss_config=_build_stage15_loss(),
        )
    )


def _package_best_candidate(
    args: argparse.Namespace,
    best_row: pd.Series,
    gp_oof_frames: dict[str, pd.DataFrame],
    deep_oof: pd.DataFrame,
    prepared_data: PreparedBaselineData,
    feature_columns: list[str],
) -> dict[str, str] | None:
    threshold_score = STAGE15_ANCHOR_SCORE - args.package_threshold
    if float(best_row["platform_score__mean"]) > threshold_score:
        return None

    candidate_slug = str(best_row["candidate_name"]).replace("__", "_")
    candidate_stem = f"{CANDIDATE_STEM_PREFIX}_{candidate_slug}"
    candidate_dir = OUTPUTS_DIR / "submissions" / candidate_stem
    predictions_path = candidate_dir / "predictions.csv"
    zip_name = f"{candidate_stem}.zip"

    if best_row["candidate_type"] == "gp_only":
        final_predictions = _fit_full_gp_predictions(
            prepared_data=prepared_data,
            feature_columns=feature_columns,
            kernel_name=str(best_row["kernel_name"]),
        )
    elif best_row["candidate_type"] == "stack":
        gp_oof = gp_oof_frames[str(best_row["kernel_name"])]
        merged_oof = deep_oof.merge(
            gp_oof,
            on=["scenario_id", "fold_index"],
            how="inner",
            validate="one_to_one",
        ).sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)
        stack_models = _fit_final_stack_models(merged_oof)
        deep_test_predictions = _fit_full_stage15_deep_sets_predictions(args).rename(
            columns={
                TARGET_COLUMNS[0]: "deep_sets_viscosity_pred",
                TARGET_COLUMNS[1]: "deep_sets_oxidation_pred",
            }
        )
        gp_test_predictions = _fit_full_gp_predictions(
            prepared_data=prepared_data,
            feature_columns=feature_columns,
            kernel_name=str(best_row["kernel_name"]),
        ).rename(
            columns={
                TARGET_COLUMNS[0]: "gp_viscosity_pred",
                TARGET_COLUMNS[1]: "gp_oxidation_pred",
            }
        )
        final_predictions = _predict_with_final_stack_models(
            deep_frame=deep_test_predictions,
            gp_frame=gp_test_predictions,
            models=stack_models,
        )
    else:
        raise ValueError(f"Unexpected candidate type for packaging: {best_row['candidate_type']}")

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

    prepared_data = load_baseline_training_data()
    feature_columns = select_baseline_feature_columns(
        prepared_data=prepared_data,
        feature_setting=BEST_BASELINE_FEATURE_SETTING,
    )

    deep_oof = _collect_stage15_oof_predictions(args)
    gp_oof_frames = _collect_gp_oof_predictions(
        prepared_data=prepared_data,
        deep_oof=deep_oof,
        feature_columns=feature_columns,
    )

    fold_metrics_frames: list[pd.DataFrame] = []
    fold_metrics_frames.append(
        _evaluate_candidate(
            frame=deep_oof,
            candidate_name=STAGE15_CANDIDATE_NAME,
            kernel_name="none",
            candidate_type="deep_sets_anchor",
            viscosity_pred=deep_oof["deep_sets_viscosity_pred"].to_numpy(dtype=float),
            oxidation_pred=deep_oof["deep_sets_oxidation_pred"].to_numpy(dtype=float),
        )
    )

    for kernel_name, gp_oof in gp_oof_frames.items():
        merged = deep_oof.merge(
            gp_oof,
            on=["scenario_id", "fold_index"],
            how="inner",
            validate="one_to_one",
        ).sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)
        if len(merged) != len(deep_oof):
            raise ValueError(f"GP OOF merge changed row count for kernel `{kernel_name}`.")

        fold_metrics_frames.append(
            _evaluate_candidate(
                frame=merged,
                candidate_name=f"gp_only__{kernel_name}",
                kernel_name=kernel_name,
                candidate_type="gp_only",
                viscosity_pred=merged["gp_viscosity_pred"].to_numpy(dtype=float),
                oxidation_pred=merged["gp_oxidation_pred"].to_numpy(dtype=float),
            )
        )

        stacked_viscosity, stacked_oxidation = _build_stack_predictions(merged)
        fold_metrics_frames.append(
            _evaluate_candidate(
                frame=merged,
                candidate_name=f"stack__deep_plus_gp__{kernel_name}",
                kernel_name=kernel_name,
                candidate_type="stack",
                viscosity_pred=stacked_viscosity,
                oxidation_pred=stacked_oxidation,
            )
        )

    fold_metrics = pd.concat(fold_metrics_frames, ignore_index=True)
    results = _summarize_candidates(fold_metrics)
    best_row = results.iloc[0]
    packaged_paths = _package_best_candidate(
        args=args,
        best_row=best_row,
        gp_oof_frames=gp_oof_frames,
        deep_oof=deep_oof,
        prepared_data=prepared_data,
        feature_columns=feature_columns,
    )
    report = _build_report(
        results=results,
        feature_columns=feature_columns,
        threshold_score=STAGE15_ANCHOR_SCORE - args.package_threshold,
        packaged_paths=packaged_paths,
    )

    _write_csv(results, GP_ENSEMBLE_RESULTS_OUTPUT_PATH)
    _write_text(report, GP_ENSEMBLE_REPORT_OUTPUT_PATH)

    print(f"gp_ensemble_results: {GP_ENSEMBLE_RESULTS_OUTPUT_PATH}")
    print(f"gp_ensemble_report: {GP_ENSEMBLE_REPORT_OUTPUT_PATH}")
    print(f"gp_ensemble_feature_count: {len(feature_columns)}")
    print(f"gp_ensemble_best_candidate: {best_row['candidate_name']}")
    if packaged_paths is not None:
        print(f"gp_ensemble_predictions: {packaged_paths['predictions_path']}")
        print(f"gp_ensemble_zip: {packaged_paths['zip_path']}")


if __name__ == "__main__":
    main()
