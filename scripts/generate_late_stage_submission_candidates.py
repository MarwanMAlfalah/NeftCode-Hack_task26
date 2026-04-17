"""Generate late-stage chemistry submission candidates without touching the shipping path."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from package_submission import build_bundle_from_predictions, validate_predictions_csv
from src.config import (
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEST_SCENARIO_FEATURES_OUTPUT_PATH,
    TRAIN_SCENARIO_FEATURES_OUTPUT_PATH,
)
from src.eval.metrics import compute_target_scales
from src.eval.run_chemistry_ensemble import (
    BASELINE_VISCOSITY_COLUMNS,
    OXIDATION_COLUMNS,
    ROBUST_VISCOSITY_COLUMN,
    _fit_convex_weights,
    _fit_gated_viscosity_model,
    _load_severity_features,
    _predict_convex,
    _predict_gated_viscosity,
    build_experiment_frame,
)
from src.eval.run_target_specialist import (
    BEST_BASELINE_FEATURE_SETTING,
    BEST_BASELINE_MODEL_NAME,
    TARGET_STRATEGY_NAME,
    _select_baseline_columns,
)
from src.models.train_baselines import (
    TARGET_COLUMNS,
    _make_inner_scorer,
    get_model_spec_by_name,
    get_target_strategy_by_name,
    load_baseline_training_data,
)
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    PREDICTION_COLUMN_MAP,
    load_deep_sets_data,
    train_full_deep_sets_variant_ensemble_and_predict,
)


SHIPPING_PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
REPORT_PATH = OUTPUTS_DIR / "reports" / "late_stage_submission_candidates.md"
SEEDS = [0, 1, 2, 3, 4]

CHEMISTRY_ZIP_NAME = "neftekod_dot_submission_chemistry_gate_v1.zip"
BLEND_75_25_ZIP_NAME = "neftekod_dot_submission_blend_75_25.zip"
BLEND_65_35_ZIP_NAME = "neftekod_dot_submission_blend_65_35.zip"


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
    ).loc[:, ["scenario_id", PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]], PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]]]]


def _write_candidate_predictions(candidate_stem: str, frame: pd.DataFrame) -> Path:
    candidate_dir = SUBMISSIONS_DIR / candidate_stem
    candidate_dir.mkdir(parents=True, exist_ok=True)
    path = candidate_dir / "predictions.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    validate_predictions_csv(path)
    return path


def _fit_full_baseline_predictions() -> pd.DataFrame:
    prepared = load_baseline_training_data()
    feature_columns = _select_baseline_columns(prepared, BEST_BASELINE_FEATURE_SETTING)
    test_features = pd.read_csv(TEST_SCENARIO_FEATURES_OUTPUT_PATH).copy()
    feature_columns = [column for column in test_features.columns if column != "scenario_id"]
    for column in feature_columns:
        test_features[column] = test_features[column].astype(float)
    X_train = prepared.X.loc[:, feature_columns].copy()
    y_train = prepared.y.to_numpy(dtype=float)
    groups = prepared.scenario_ids.to_numpy()
    X_test = test_features.loc[:, feature_columns].copy()

    model_spec = get_model_spec_by_name(BEST_BASELINE_MODEL_NAME)
    target_strategy = get_target_strategy_by_name(TARGET_STRATEGY_NAME)
    target_scales = compute_target_scales(y_train, TARGET_COLUMNS)
    scorer = _make_inner_scorer(target_strategy=target_strategy, target_scales=target_scales)
    inner_cv = GroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        estimator=model_spec.build_estimator(RANDOM_SEED),
        param_grid=model_spec.build_param_grid(X_train, target_strategy.transform(y_train)),
        scoring=scorer,
        cv=inner_cv,
        n_jobs=None,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, target_strategy.transform(y_train), groups=groups)
    predictions = target_strategy.inverse_transform(np.asarray(search.best_estimator_.predict(X_test), dtype=float))
    return pd.DataFrame(
        {
            "scenario_id": test_features["scenario_id"].to_numpy(),
            TARGET_COLUMNS[0]: predictions[:, 0],
            TARGET_COLUMNS[1]: predictions[:, 1],
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _fit_full_deep_sets_predictions() -> dict[str, pd.DataFrame]:
    prepared = load_deep_sets_data()
    config = DeepSetsConfig()
    outputs: dict[str, pd.DataFrame] = {}

    outputs["deep_sets_v1"] = _to_internal_columns(
        train_full_deep_sets_variant_ensemble_and_predict(
            prepared_data=prepared,
            variant=HybridVariant(
                name="deep_sets_v1",
                use_component_embedding=True,
                use_tabular_branch=False,
            ),
            target_strategy_name=TARGET_STRATEGY_NAME,
            seeds=SEEDS,
            config=config,
        )
    )
    outputs["hybrid_v2_robust"] = _to_internal_columns(
        train_full_deep_sets_variant_ensemble_and_predict(
            prepared_data=prepared,
            variant=HybridVariant(
                name="hybrid_deep_sets_v2_family_only",
                use_component_embedding=False,
                use_tabular_branch=True,
            ),
            target_strategy_name=TARGET_STRATEGY_NAME,
            seeds=SEEDS,
            config=config,
            loss_config=LossConfig(name="robust_viscosity", use_robust_viscosity_loss=True, viscosity_delta=1.0),
        )
    )
    return outputs


def _fit_full_meta_models(include_robust_viscosity: bool = True) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    oof_frame = build_experiment_frame(
        outer_splits=5,
        config=DeepSetsConfig(),
        include_robust_viscosity=include_robust_viscosity,
    )

    viscosity_feature_columns = BASELINE_VISCOSITY_COLUMNS.copy()
    if include_robust_viscosity and ROBUST_VISCOSITY_COLUMN in oof_frame.columns:
        viscosity_feature_columns.append(ROBUST_VISCOSITY_COLUMN)
    oxidation_weights = _fit_convex_weights(
        predictions=oof_frame.loc[:, OXIDATION_COLUMNS].to_numpy(dtype=float),
        target=oof_frame[TARGET_COLUMNS[1] + "__true"].to_numpy(dtype=float),
    )
    convex_viscosity_weights = _fit_convex_weights(
        predictions=oof_frame.loc[:, viscosity_feature_columns].to_numpy(dtype=float),
        target=oof_frame[TARGET_COLUMNS[0] + "__true"].to_numpy(dtype=float),
    )

    severity_feature_columns = [
        column
        for column in oof_frame.columns
        if column in set(_load_severity_features(TRAIN_SCENARIO_FEATURES_OUTPUT_PATH).columns) - {"scenario_id"}
    ]
    gate_model = _fit_gated_viscosity_model(
        low_predictions=oof_frame.loc[:, BASELINE_VISCOSITY_COLUMNS].to_numpy(dtype=float),
        high_predictions=oof_frame.loc[:, viscosity_feature_columns].to_numpy(dtype=float),
        severity_features=oof_frame.loc[:, severity_feature_columns].to_numpy(dtype=float),
        target=oof_frame[TARGET_COLUMNS[0] + "__true"].to_numpy(dtype=float),
    )
    gate_model["feature_names"] = np.asarray(severity_feature_columns, dtype=object)
    gate_model["convex_viscosity_weights"] = convex_viscosity_weights
    gate_model["convex_oxidation_weights"] = oxidation_weights
    gate_model["viscosity_feature_columns"] = np.asarray(viscosity_feature_columns, dtype=object)
    return gate_model, oof_frame


def _build_chemistry_test_predictions(
    shipping_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    deep_sets_v1_frame: pd.DataFrame,
    robust_frame: pd.DataFrame,
    gate_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    test_severity = _load_severity_features(TEST_SCENARIO_FEATURES_OUTPUT_PATH)
    merged = shipping_frame.merge(
        baseline_frame.rename(columns={TARGET_COLUMNS[0]: "pls_viscosity_pred", TARGET_COLUMNS[1]: "pls_oxidation_pred"}),
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        deep_sets_v1_frame.rename(columns={TARGET_COLUMNS[0]: "deep_sets_v1_viscosity_pred", TARGET_COLUMNS[1]: "deep_sets_v1_oxidation_pred"}),
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        robust_frame.rename(columns={TARGET_COLUMNS[0]: ROBUST_VISCOSITY_COLUMN, TARGET_COLUMNS[1]: "robust_unused_oxidation"}),
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        test_severity,
        on="scenario_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.rename(
        columns={
            TARGET_COLUMNS[0]: "hybrid_v2_viscosity_pred",
            TARGET_COLUMNS[1]: "hybrid_v2_oxidation_pred",
        }
    ).sort_values("scenario_id").reset_index(drop=True)

    oxidation_predictions = _predict_convex(
        predictions=merged.loc[:, OXIDATION_COLUMNS].to_numpy(dtype=float),
        weights=gate_model["convex_oxidation_weights"],
    )
    gated_viscosity_predictions = _predict_gated_viscosity(
        low_predictions=merged.loc[:, BASELINE_VISCOSITY_COLUMNS].to_numpy(dtype=float),
        high_predictions=merged.loc[:, gate_model["viscosity_feature_columns"].tolist()].to_numpy(dtype=float),
        severity_features=merged.loc[:, gate_model["feature_names"].tolist()].to_numpy(dtype=float),
        model=gate_model,
    )
    return pd.DataFrame(
        {
            "scenario_id": merged["scenario_id"].to_numpy(),
            TARGET_COLUMNS[0]: gated_viscosity_predictions,
            TARGET_COLUMNS[1]: oxidation_predictions,
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _blend_predictions(
    shipping_frame: pd.DataFrame,
    chemistry_frame: pd.DataFrame,
    shipping_weight: float,
) -> pd.DataFrame:
    chemistry_weight = 1.0 - shipping_weight
    merged = shipping_frame.merge(
        chemistry_frame,
        on="scenario_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_ship", "_chem"),
    ).sort_values("scenario_id").reset_index(drop=True)
    return pd.DataFrame(
        {
            "scenario_id": merged["scenario_id"].to_numpy(),
            TARGET_COLUMNS[0]: shipping_weight * merged[f"{TARGET_COLUMNS[0]}_ship"] + chemistry_weight * merged[f"{TARGET_COLUMNS[0]}_chem"],
            TARGET_COLUMNS[1]: shipping_weight * merged[f"{TARGET_COLUMNS[1]}_ship"] + chemistry_weight * merged[f"{TARGET_COLUMNS[1]}_chem"],
        }
    )


def _validate_and_package(candidate_stem: str, zip_name: str, frame: pd.DataFrame) -> dict[str, object]:
    submission_frame = _to_submission_columns(frame)
    predictions_path = _write_candidate_predictions(candidate_stem, submission_frame)
    validation = validate_predictions_csv(predictions_path)
    zip_path = build_bundle_from_predictions(predictions_path=predictions_path, zip_name=zip_name)
    return {
        "candidate_name": candidate_stem,
        "predictions_path": str(predictions_path),
        "zip_path": str(zip_path),
        "validation_status": "passed",
        "row_count": validation["row_count"],
    }


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    shipping_frame = _to_internal_columns(pd.read_csv(SHIPPING_PREDICTIONS_PATH))
    baseline_frame = _fit_full_baseline_predictions()
    deep_sets_outputs = _fit_full_deep_sets_predictions()
    gate_model, _ = _fit_full_meta_models(include_robust_viscosity=True)
    chemistry_frame = _build_chemistry_test_predictions(
        shipping_frame=shipping_frame,
        baseline_frame=baseline_frame,
        deep_sets_v1_frame=deep_sets_outputs["deep_sets_v1"],
        robust_frame=deep_sets_outputs["hybrid_v2_robust"],
        gate_model=gate_model,
    )

    blend_75_25_frame = _blend_predictions(
        shipping_frame=shipping_frame,
        chemistry_frame=chemistry_frame,
        shipping_weight=0.75,
    )
    blend_65_35_frame = _blend_predictions(
        shipping_frame=shipping_frame,
        chemistry_frame=chemistry_frame,
        shipping_weight=0.65,
    )

    records = [
        {
            **_validate_and_package(
                candidate_stem="neftekod_dot_submission_chemistry_gate_v1",
                zip_name=CHEMISTRY_ZIP_NAME,
                frame=chemistry_frame,
            ),
            "source_predictions": "shipping_hybrid_v2_raw + pls_regression + deep_sets_v1_raw + hybrid_v2_raw_robust_viscosity",
            "blending_weights": "gated viscosity, convex oxidation",
        },
        {
            **_validate_and_package(
                candidate_stem="neftekod_dot_submission_blend_75_25",
                zip_name=BLEND_75_25_ZIP_NAME,
                frame=blend_75_25_frame,
            ),
            "source_predictions": "shipping_hybrid_v2_raw + chemistry_gate_v1",
            "blending_weights": "0.75 shipping / 0.25 chemistry",
        },
        {
            **_validate_and_package(
                candidate_stem="neftekod_dot_submission_blend_65_35",
                zip_name=BLEND_65_35_ZIP_NAME,
                frame=blend_65_35_frame,
            ),
            "source_predictions": "shipping_hybrid_v2_raw + chemistry_gate_v1",
            "blending_weights": "0.65 shipping / 0.35 chemistry",
        },
    ]

    report_lines = [
        "# Late Stage Submission Candidates",
        "",
        "## Candidate Summary",
        "",
        "```text",
        pd.DataFrame.from_records(records)[
            ["candidate_name", "source_predictions", "blending_weights", "validation_status", "zip_path"]
        ].to_string(index=False),
        "```",
        "",
        "## Recommended Upload Order",
        "1. `neftekod_dot_submission_blend_75_25.zip`",
        "2. `neftekod_dot_submission_blend_65_35.zip`",
        "3. `neftekod_dot_submission_chemistry_gate_v1.zip`",
        "",
        "## Rationale",
        "- `blend_75_25` is the most conservative move away from the stable shipping anchor.",
        "- `blend_65_35` increases exposure to the promising chemistry candidate while still hedging with the platform-proven shipping path.",
        "- `chemistry_gate_v1` is the highest-upside but also the highest-variance submission candidate.",
        "",
        "## Meta Details",
        f"- Oxidation convex weights: `{np.round(gate_model['convex_oxidation_weights'], 4).tolist()}`",
        f"- Viscosity low-severity weights: `{np.round(gate_model['low_weights'], 4).tolist()}`",
        f"- Viscosity high-severity weights: `{np.round(gate_model['high_weights'], 4).tolist()}`",
        f"- Gate coefficients: `{json.dumps({name: round(float(value), 4) for name, value in zip(gate_model['feature_names'].tolist(), gate_model['gate_beta'], strict=False)}, sort_keys=True)}`",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"late_stage_submission_report: {REPORT_PATH}")
    for record in records:
        print(f"{record['candidate_name']}: {record['zip_path']}")


if __name__ == "__main__":
    main()
