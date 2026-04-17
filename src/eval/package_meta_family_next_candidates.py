"""Package the next runner-up submission candidates from the successful meta-stack family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn import set_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import build_bundle_from_predictions, validate_bundle, validate_predictions_csv
from src.eval.run_gp_ensemble_stage2 import (
    BEST_BASELINE_FEATURE_SETTING,
    _fit_final_stack_models,
    _fit_full_gp_predictions,
    _predict_with_final_stack_models,
    _to_internal_columns,
    _to_submission_columns,
)
from src.eval.run_gp_stage2_diagnostic_sprint import CURRENT_STACK_SOURCE, STACK_DOT_SOURCE
from src.models.train_baselines import (
    OXIDATION_TARGET,
    TARGET_COLUMNS,
    VISCOSITY_TARGET,
    load_baseline_training_data,
    select_baseline_feature_columns,
)


META_RESULTS_PATH = REPO_ROOT / "outputs" / "cv" / "meta_stack_search_results.csv"
BOOTSTRAP_PATH = REPO_ROOT / "outputs" / "cv" / "paired_bootstrap_ci_stage15_vs_best_meta.csv"
META_REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "meta_stack_search_report.md"
OOF_PATH = REPO_ROOT / "outputs" / "cv" / "gp_stage2_oof_predictions.csv"
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "meta_family_next_candidates.md"
STAGE15_PREDICTIONS_PATH = (
    REPO_ROOT / "outputs" / "submissions" / "neftekod_dot_submission_stage15_huber_fixedmetric" / "predictions.csv"
)
PACKAGE_PREFIX = "neftekod_dot_submission_gp_stage2_meta_runnerup"
MAX_CANDIDATES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    return parser.parse_args()


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _effective_weight_signature(weights_json: str) -> dict[str, float]:
    weights = json.loads(weights_json)
    return {
        source_name: round(float(weight), 6)
        for source_name, weight in weights.items()
        if abs(float(weight)) > 1e-6
    }


def _candidate_signature(row: pd.Series) -> str:
    signature = {
        "viscosity": _effective_weight_signature(str(row["viscosity_weights_json"])),
        "oxidation": _effective_weight_signature(str(row["oxidation_weights_json"])),
    }
    return json.dumps(signature, sort_keys=True)


def _load_results() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    results = pd.read_csv(META_RESULTS_PATH)
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)
    if results.empty:
        raise ValueError(f"No meta-stack candidates found in {META_RESULTS_PATH}")
    winner = results.iloc[0]
    working = results.loc[results["candidate_family"] == "nonnegative_mae_blend"].copy()
    working = working.loc[working["candidate_name"] != winner["candidate_name"]].copy()
    working["signature"] = working.apply(_candidate_signature, axis=1)
    working["delta_vs_winner"] = working["platform_score__mean"] - float(winner["platform_score__mean"])
    return working, winner, bootstrap


def _select_runner_ups(
    results: pd.DataFrame,
    winner: pd.Series,
    max_candidates: int,
) -> pd.DataFrame:
    winner_signature = _candidate_signature(winner)
    working = results.drop_duplicates("signature").copy()
    working = working.loc[working["signature"] != winner_signature].copy()
    winner_oxidation = float(winner[f"{OXIDATION_TARGET}__mae__mean"])
    working = working.loc[working["delta_vs_winner"] <= 0.0004].copy()
    working = working.loc[working[f"{OXIDATION_TARGET}__mae__mean"] <= winner_oxidation + 0.12].copy()
    working = working.sort_values(
        [
            "platform_score__mean",
            f"{OXIDATION_TARGET}__mae__mean",
            f"{VISCOSITY_TARGET}__mae__mean",
        ]
    ).reset_index(drop=True)
    if working.empty:
        raise ValueError("No runner-up candidates satisfied the selection criteria.")
    return working.head(max_candidates).copy()


def _load_stage15_test_predictions() -> pd.DataFrame:
    return _to_internal_columns(pd.read_csv(STAGE15_PREDICTIONS_PATH))


def _load_oof_frame() -> pd.DataFrame:
    frame = pd.read_csv(OOF_PATH)
    required = {
        "scenario_id",
        "fold_index",
        f"{VISCOSITY_TARGET}__true",
        f"{OXIDATION_TARGET}__true",
        "stage15_viscosity_pred",
        "stage15_oxidation_pred",
        "gp_matern_white_viscosity_pred",
        "gp_matern_white_oxidation_pred",
        "gp_matern_white_dot_viscosity_pred",
        "gp_matern_white_dot_oxidation_pred",
        f"{CURRENT_STACK_SOURCE}_viscosity_pred",
        f"{CURRENT_STACK_SOURCE}_oxidation_pred",
        f"{STACK_DOT_SOURCE}_viscosity_pred",
        f"{STACK_DOT_SOURCE}_oxidation_pred",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF artifact is missing required columns: {missing}")
    return frame.sort_values(["fold_index", "scenario_id"]).reset_index(drop=True)


def _fit_current_stack_predictions(
    oof_frame: pd.DataFrame,
    stage15_test_predictions: pd.DataFrame,
    gp_test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    train_frame = oof_frame.rename(
        columns={
            "stage15_viscosity_pred": "deep_sets_viscosity_pred",
            "stage15_oxidation_pred": "deep_sets_oxidation_pred",
            "gp_matern_white_viscosity_pred": "gp_viscosity_pred",
            "gp_matern_white_oxidation_pred": "gp_oxidation_pred",
        }
    )
    models = _fit_final_stack_models(train_frame)
    return _predict_with_final_stack_models(
        deep_frame=stage15_test_predictions.rename(
            columns={
                TARGET_COLUMNS[0]: "deep_sets_viscosity_pred",
                TARGET_COLUMNS[1]: "deep_sets_oxidation_pred",
            }
        ),
        gp_frame=gp_test_predictions.rename(
            columns={
                TARGET_COLUMNS[0]: "gp_viscosity_pred",
                TARGET_COLUMNS[1]: "gp_oxidation_pred",
            }
        ),
        models=models,
    )


def _fit_stack_dot_predictions(
    oof_frame: pd.DataFrame,
    stage15_test_predictions: pd.DataFrame,
    gp_test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    train_frame = oof_frame.rename(
        columns={
            "stage15_viscosity_pred": "deep_sets_viscosity_pred",
            "stage15_oxidation_pred": "deep_sets_oxidation_pred",
            "gp_matern_white_dot_viscosity_pred": "gp_viscosity_pred",
            "gp_matern_white_dot_oxidation_pred": "gp_oxidation_pred",
        }
    )
    models = _fit_final_stack_models(train_frame)
    return _predict_with_final_stack_models(
        deep_frame=stage15_test_predictions.rename(
            columns={
                TARGET_COLUMNS[0]: "deep_sets_viscosity_pred",
                TARGET_COLUMNS[1]: "deep_sets_oxidation_pred",
            }
        ),
        gp_frame=gp_test_predictions.rename(
            columns={
                TARGET_COLUMNS[0]: "gp_viscosity_pred",
                TARGET_COLUMNS[1]: "gp_oxidation_pred",
            }
        ),
        models=models,
    )


def _build_test_source_predictions() -> dict[str, pd.DataFrame]:
    prepared_data = load_baseline_training_data()
    feature_columns = select_baseline_feature_columns(prepared_data, BEST_BASELINE_FEATURE_SETTING)
    oof_frame = _load_oof_frame()
    stage15_test_predictions = _load_stage15_test_predictions()
    gp_matern_white_predictions = _fit_full_gp_predictions(
        prepared_data=prepared_data,
        feature_columns=feature_columns,
        kernel_name="matern_white",
    )
    gp_matern_white_dot_predictions = _fit_full_gp_predictions(
        prepared_data=prepared_data,
        feature_columns=feature_columns,
        kernel_name="matern_white_dot",
    )
    current_stack_predictions = _fit_current_stack_predictions(
        oof_frame=oof_frame,
        stage15_test_predictions=stage15_test_predictions,
        gp_test_predictions=gp_matern_white_predictions,
    )
    stack_dot_predictions = _fit_stack_dot_predictions(
        oof_frame=oof_frame,
        stage15_test_predictions=stage15_test_predictions,
        gp_test_predictions=gp_matern_white_dot_predictions,
    )
    return {
        "stage15": stage15_test_predictions,
        "gp_matern_white": gp_matern_white_predictions,
        "gp_matern_white_dot": gp_matern_white_dot_predictions,
        CURRENT_STACK_SOURCE: current_stack_predictions,
        STACK_DOT_SOURCE: stack_dot_predictions,
    }


def _blend_test_predictions(
    source_predictions: dict[str, pd.DataFrame],
    viscosity_weights_json: str,
    oxidation_weights_json: str,
) -> pd.DataFrame:
    scenario_ids = source_predictions["stage15"]["scenario_id"].to_numpy()
    viscosity_weights = _effective_weight_signature(viscosity_weights_json)
    oxidation_weights = _effective_weight_signature(oxidation_weights_json)

    viscosity = np.zeros(len(scenario_ids), dtype=float)
    for source_name, weight in viscosity_weights.items():
        viscosity += float(weight) * source_predictions[source_name][TARGET_COLUMNS[0]].to_numpy(dtype=float)

    oxidation = np.zeros(len(scenario_ids), dtype=float)
    for source_name, weight in oxidation_weights.items():
        oxidation += float(weight) * source_predictions[source_name][TARGET_COLUMNS[1]].to_numpy(dtype=float)

    return pd.DataFrame(
        {
            "scenario_id": scenario_ids,
            TARGET_COLUMNS[0]: viscosity,
            TARGET_COLUMNS[1]: oxidation,
        }
    ).sort_values("scenario_id").reset_index(drop=True)


def _label_for_candidate(row: pd.Series, index: int) -> str:
    oxidation_signature = _effective_weight_signature(str(row["oxidation_weights_json"]))
    oxidation_keys = tuple(sorted(oxidation_signature.keys()))
    if oxidation_keys == ("current_stack",):
        return "current_stack_only_oxidation"
    if oxidation_keys == ("gp_matern_white_dot", "stage15"):
        return "stage15_gpdot_oxidation"
    return f"runnerup_{index + 1}"


def _load_live_winner_predictions(winner: pd.Series) -> pd.DataFrame:
    candidate_slug = str(winner["candidate_name"]).replace("__", "_")
    predictions_path = REPO_ROOT / "outputs" / "submissions" / f"neftekod_dot_submission_gp_stage2_meta_{candidate_slug}" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing live winner predictions file: {predictions_path}")
    return _to_internal_columns(pd.read_csv(predictions_path))


def _package_candidate(
    row: pd.Series,
    source_predictions: dict[str, pd.DataFrame],
    label: str,
) -> dict[str, object]:
    final_predictions = _blend_test_predictions(
        source_predictions=source_predictions,
        viscosity_weights_json=str(row["viscosity_weights_json"]),
        oxidation_weights_json=str(row["oxidation_weights_json"]),
    )
    candidate_stem = f"{PACKAGE_PREFIX}_{label}"
    candidate_dir = REPO_ROOT / "outputs" / "submissions" / candidate_stem
    predictions_path = candidate_dir / "predictions.csv"
    zip_name = f"{candidate_stem}.zip"

    candidate_dir.mkdir(parents=True, exist_ok=True)
    submission_frame = _to_submission_columns(final_predictions)
    submission_frame.to_csv(predictions_path, index=False, encoding="utf-8")
    predictions_validation = validate_predictions_csv(predictions_path)
    zip_path = build_bundle_from_predictions(predictions_path=predictions_path, zip_name=zip_name)
    bundle_validation = validate_bundle(zip_path)

    return {
        "candidate_name": str(row["candidate_name"]),
        "label": label,
        "predictions_path": str(predictions_path),
        "zip_path": str(zip_path),
        "predictions_validation": predictions_validation,
        "bundle_validation": bundle_validation,
    }


def _build_report(
    selected_rows: pd.DataFrame,
    packaged_records: list[dict[str, object]],
    winner: pd.Series,
    bootstrap: pd.DataFrame,
    source_predictions: dict[str, pd.DataFrame],
) -> str:
    winner_predictions = _load_live_winner_predictions(winner)
    winner_score = float(winner["platform_score__mean"])
    bootstrap_row = bootstrap.iloc[0] if not bootstrap.empty else None

    lines = [
        "# Meta Family Next Candidates",
        "",
        "## Selection Basis",
        "- Ranked candidates from `outputs/cv/meta_stack_search_results.csv` and removed weight-equivalent duplicates.",
        "- Kept only runner-ups from the same successful `nonnegative_mae_blend` meta family.",
        "- Required very small local score drop vs the current meta winner, no obvious oxidation collapse, and distinct effective blend structure.",
        (
            f"- Available family-level bootstrap context from the current live winner: "
            f"`{bootstrap_row['probability_of_improvement']:.2%}` probability of improvement vs Stage 1.5."
            if bootstrap_row is not None
            else "- No bootstrap context file was available."
        ),
        "",
        "## Recommended Upload Order",
    ]

    for upload_index, (row, packaged) in enumerate(zip(selected_rows.to_dict(orient="records"), packaged_records, strict=False), start=1):
        candidate_predictions = _blend_test_predictions(
            source_predictions=source_predictions,
            viscosity_weights_json=str(row["viscosity_weights_json"]),
            oxidation_weights_json=str(row["oxidation_weights_json"]),
        )
        merged = winner_predictions.merge(
            candidate_predictions,
            on="scenario_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_winner", "_candidate"),
        )
        viscosity_shift = float(
            np.mean(
                np.abs(
                    merged[f"{TARGET_COLUMNS[0]}_winner"].to_numpy(dtype=float)
                    - merged[f"{TARGET_COLUMNS[0]}_candidate"].to_numpy(dtype=float)
                )
            )
        )
        oxidation_shift = float(
            np.mean(
                np.abs(
                    merged[f"{TARGET_COLUMNS[1]}_winner"].to_numpy(dtype=float)
                    - merged[f"{TARGET_COLUMNS[1]}_candidate"].to_numpy(dtype=float)
                )
            )
        )
        delta_vs_winner = float(row["platform_score__mean"]) - winner_score
        oxidation_delta = float(row[f"{OXIDATION_TARGET}__mae__mean"]) - float(winner[f"{OXIDATION_TARGET}__mae__mean"])

        lines.extend(
            [
                f"{upload_index}. `{Path(str(packaged['zip_path'])).name}`",
                f"Candidate name: `{row['candidate_name']}`",
                f"Local score: `{row['platform_score__mean']:.6f}`",
                f"Delta vs current meta winner: `{delta_vs_winner:+.6f}`",
                (
                    "Why it is worth a platform test: "
                    f"effective viscosity blend stays on the winning Stage15+GP(matern_white) pattern, "
                    f"while oxidation shifts to `{_effective_weight_signature(str(row['oxidation_weights_json']))}`. "
                    f"Local oxidation MAE delta is `{oxidation_delta:+.6f}` and mean prediction shift vs the live winner is "
                    f"`{viscosity_shift:.4f}` for viscosity / `{oxidation_shift:.4f}` for oxidation."
                ),
                f"ZIP validation: `passed`",
                "",
            ]
        )

    lines.extend(
        [
            "## Candidate Table",
            "",
            "```text",
            selected_rows[
                [
                    "candidate_name",
                    "platform_score__mean",
                    "delta_vs_winner",
                    f"{VISCOSITY_TARGET}__mae__mean",
                    f"{OXIDATION_TARGET}__mae__mean",
                ]
            ].to_string(index=False),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    set_config(enable_metadata_routing=True)
    results, winner, bootstrap = _load_results()
    selected = _select_runner_ups(results=results, winner=winner, max_candidates=args.max_candidates)
    source_predictions = _build_test_source_predictions()

    packaged_records: list[dict[str, object]] = []
    for index, row in enumerate(selected.to_dict(orient="records")):
        packaged_records.append(
            _package_candidate(
                row=pd.Series(row),
                source_predictions=source_predictions,
                label=_label_for_candidate(pd.Series(row), index),
            )
        )

    report = _build_report(
        selected_rows=selected,
        packaged_records=packaged_records,
        winner=winner,
        bootstrap=bootstrap,
        source_predictions=source_predictions,
    )
    _write_text(report, REPORT_PATH)

    print(f"meta_family_next_candidates_report: {REPORT_PATH}")
    for packaged in packaged_records:
        print(f"{packaged['candidate_name']}: {packaged['zip_path']}")


if __name__ == "__main__":
    main()
