"""Build final ultra-conservative prediction-space blend challengers from two live submissions."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import REQUIRED_COLUMNS, build_bundle_from_predictions, validate_predictions_csv
from src.config import OUTPUTS_DIR, REPORTS_DIR


BEST7_SCORE = 0.104084
BEST9_SCORE = 0.104137
BEST7_LABEL = "submission #7"
BEST9_LABEL = "submission #9"
BEST7_PREDICTIONS_PATH = (
    REPO_ROOT / "outputs" / "submissions" / "neftekod_dot_submission_stage15_huber_fixedmetric" / "predictions.csv"
)
BEST9_PREDICTIONS_PATH = (
    REPO_ROOT
    / "outputs"
    / "submissions"
    / "neftekod_dot_submission_huber_weight_grid_vd100_od050_joint_light"
    / "predictions.csv"
)
REPORT_PATH = REPORTS_DIR / "final_prediction_blend_candidates.md"
BLEND_SPECS = [
    {
        "name": "blend_best7_best9_80_20",
        "zip_name": "neftekod_dot_submission_blend_best7_best9_80_20.zip",
        "candidate_dir": "neftekod_dot_submission_blend_best7_best9_80_20",
        "best7_weight": 0.80,
        "best9_weight": 0.20,
        "recommended_order": 1,
        "risk_note": "Closest to the incumbent best live probe, so it is the safest final submission-space test.",
    },
    {
        "name": "blend_best7_best9_60_40",
        "zip_name": "neftekod_dot_submission_blend_best7_best9_60_40.zip",
        "candidate_dir": "neftekod_dot_submission_blend_best7_best9_60_40",
        "best7_weight": 0.60,
        "best9_weight": 0.40,
        "recommended_order": 2,
        "risk_note": "Still conservative, but it leans farther toward the newer #9 challenger and is therefore slightly less safe.",
    },
]


def _load_submission_frame(path: Path) -> pd.DataFrame:
    validate_predictions_csv(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Unexpected submission columns in {path}: {list(frame.columns)}")
    return frame.copy()


def _verify_inputs(best7_frame: pd.DataFrame, best9_frame: pd.DataFrame) -> dict[str, object]:
    if len(best7_frame) != 40 or len(best9_frame) != 40:
        raise ValueError("Both source submissions must contain exactly 40 rows.")
    if not best7_frame["scenario_id"].equals(best9_frame["scenario_id"]):
        raise ValueError("Source submissions do not share identical scenario_id ordering.")
    return {
        "same_order": True,
        "row_count": int(len(best7_frame)),
        "column_names": list(best7_frame.columns),
    }


def _build_blend_frame(
    best7_frame: pd.DataFrame,
    best9_frame: pd.DataFrame,
    best7_weight: float,
    best9_weight: float,
) -> pd.DataFrame:
    blend_frame = best7_frame.copy()
    target_columns = REQUIRED_COLUMNS[1:]
    for column in target_columns:
        blend_frame[column] = (
            float(best7_weight) * best7_frame[column].to_numpy(dtype=float)
            + float(best9_weight) * best9_frame[column].to_numpy(dtype=float)
        )
    return blend_frame


def _write_predictions(frame: pd.DataFrame, candidate_dir_name: str) -> Path:
    candidate_dir = OUTPUTS_DIR / "submissions" / candidate_dir_name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = candidate_dir / "predictions.csv"
    frame.to_csv(predictions_path, index=False, encoding="utf-8")
    validate_predictions_csv(predictions_path)
    return predictions_path


def _build_report(input_validation: dict[str, object], built_candidates: list[dict[str, object]]) -> str:
    lines = [
        "# Final Prediction Blend Candidates",
        "",
        "## Scope",
        "- Pure prediction-space blend only.",
        "- No retraining and no new CV were run.",
        "- Official shipping path remained untouched.",
        f"- {BEST7_LABEL} source: `{BEST7_PREDICTIONS_PATH}` with live platform score `{BEST7_SCORE:.6f}`",
        f"- {BEST9_LABEL} source: `{BEST9_PREDICTIONS_PATH}` with live platform score `{BEST9_SCORE:.6f}`",
        "",
        "## Input Validation",
        f"- Same scenario_id ordering: `{'yes' if input_validation['same_order'] else 'no'}`",
        f"- Same 40 rows: `{'yes' if input_validation['row_count'] == 40 else 'no'}`",
        f"- Required columns present: `{'yes' if input_validation['column_names'] == REQUIRED_COLUMNS else 'no'}`",
        "",
        "## Candidates",
    ]
    for candidate in built_candidates:
        lines.extend(
            [
                f"- `{candidate['name']}`: `{candidate['best7_weight']:.2f} * #7 + {candidate['best9_weight']:.2f} * #9`",
                f"  predictions: `{candidate['predictions_path']}`",
                f"  zip: `{candidate['zip_path']}`",
                f"  validation: `{candidate['validation_status']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended Upload Order",
            f"1. `{built_candidates[0]['zip_name']}`",
            f"2. `{built_candidates[1]['zip_name']}`",
            "",
            "## Recommendation",
            f"- Safer final probe: `{built_candidates[0]['zip_name']}`",
            f"- Why: {built_candidates[0]['risk_note']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best7_frame = _load_submission_frame(BEST7_PREDICTIONS_PATH)
    best9_frame = _load_submission_frame(BEST9_PREDICTIONS_PATH)
    input_validation = _verify_inputs(best7_frame=best7_frame, best9_frame=best9_frame)

    built_candidates: list[dict[str, object]] = []
    for spec in BLEND_SPECS:
        blend_frame = _build_blend_frame(
            best7_frame=best7_frame,
            best9_frame=best9_frame,
            best7_weight=spec["best7_weight"],
            best9_weight=spec["best9_weight"],
        )
        if blend_frame.isnull().any().any():
            raise ValueError(f"Blend {spec['name']} contains null values.")
        if blend_frame["scenario_id"].duplicated().any():
            raise ValueError(f"Blend {spec['name']} contains duplicate scenario_id values.")

        predictions_path = _write_predictions(frame=blend_frame, candidate_dir_name=spec["candidate_dir"])
        zip_path = build_bundle_from_predictions(
            predictions_path=predictions_path,
            zip_name=spec["zip_name"],
        )
        built_candidates.append(
            {
                **spec,
                "predictions_path": str(predictions_path),
                "zip_path": str(zip_path),
                "validation_status": "passed",
            }
        )

    report_text = _build_report(input_validation=input_validation, built_candidates=built_candidates)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"best7_predictions: {BEST7_PREDICTIONS_PATH}")
    print(f"best9_predictions: {BEST9_PREDICTIONS_PATH}")
    print(f"input_same_order: {'yes' if input_validation['same_order'] else 'no'}")
    print(f"input_row_count: {input_validation['row_count']}")
    print(f"blend_report: {REPORT_PATH}")
    for candidate in built_candidates:
        print(f"{candidate['name']}_predictions: {candidate['predictions_path']}")
        print(f"{candidate['name']}_zip: {candidate['zip_path']}")


if __name__ == "__main__":
    main()
