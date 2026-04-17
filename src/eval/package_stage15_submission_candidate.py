"""Package a separate Stage 1.5 submission candidate without touching the official bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from package_submission import REQUIRED_COLUMNS, build_bundle_from_predictions, validate_predictions_csv
from src.config import OUTPUTS_DIR
from src.models.train_baselines import TARGET_COLUMNS
from src.models.train_deep_sets import (
    DeepSetsConfig,
    HybridVariant,
    LossConfig,
    PREDICTION_COLUMN_MAP,
    load_deep_sets_data,
    train_full_deep_sets_variant_ensemble_and_predict,
)


CANDIDATE_STEM = "neftekod_dot_submission_stage15_huber_fixedmetric"
CANDIDATE_ZIP_NAME = f"{CANDIDATE_STEM}.zip"
CANDIDATE_DIR = OUTPUTS_DIR / "submissions" / CANDIDATE_STEM
CANDIDATE_PREDICTIONS_PATH = CANDIDATE_DIR / "predictions.csv"
CANDIDATE_ZIP_PATH = OUTPUTS_DIR / "submissions" / CANDIDATE_ZIP_NAME
REPORT_PATH = OUTPUTS_DIR / "reports" / "stage15_submission_candidate.md"
OFFICIAL_ANCHOR_PATH = OUTPUTS_DIR / "submissions" / "neftekod_dot_submission_final.zip"
OFFICIAL_ANCHOR_SCORE = 0.109256
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    return parser.parse_args()


def _write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _to_submission_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            TARGET_COLUMNS[0]: PREDICTION_COLUMN_MAP[TARGET_COLUMNS[0]],
            TARGET_COLUMNS[1]: PREDICTION_COLUMN_MAP[TARGET_COLUMNS[1]],
        }
    ).loc[:, REQUIRED_COLUMNS]


def _build_validation_details(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count": int(len(frame)),
        "exact_required_columns": list(frame.columns) == REQUIRED_COLUMNS,
        "no_nulls": bool(not frame.isnull().any().any()),
        "no_duplicate_scenario_id": bool(not frame["scenario_id"].duplicated().any()),
    }


def build_report(validation: dict[str, object], zip_path: Path) -> str:
    all_passed = (
        validation["row_count"] == 40
        and validation["exact_required_columns"]
        and validation["no_nulls"]
        and validation["no_duplicate_scenario_id"]
    )
    lines = [
        "# Stage 1.5 Submission Candidate",
        "",
        "## Candidate Description",
        "- Variant: `hybrid_deep_sets_v2_family_only`",
        "- Target strategy: `raw`",
        "- Training loss: Huber-style on both targets",
        "- Checkpoint metric: fixed platform score",
        "- Fixed platform score formula: `0.5 * (visc_MAE / 2439.25 + ox_MAE / 160.62)`",
        "- Ensemble seeds for full-data prediction: `[0, 1, 2, 3, 4]`",
        "",
        "## Validation Status",
        f"- Overall status: `{'passed' if all_passed else 'failed'}`",
        f"- Exactly 40 rows: `{'passed' if validation['row_count'] == 40 else 'failed'}` (`{validation['row_count']}` rows)",
        f"- Exact required column names: `{'passed' if validation['exact_required_columns'] else 'failed'}`",
        f"- No nulls: `{'passed' if validation['no_nulls'] else 'failed'}`",
        f"- No duplicate `scenario_id`: `{'passed' if validation['no_duplicate_scenario_id'] else 'failed'}`",
        "",
        "## Packaging",
        f"- Candidate predictions path: `{CANDIDATE_PREDICTIONS_PATH}`",
        f"- Final ZIP path: `{zip_path}`",
        "",
        "## Recommended Upload Order",
        f"1. Keep the current official anchor first: `neftekod_dot_submission_final.zip` at `{OFFICIAL_ANCHOR_SCORE:.6f}`",
        f"2. Upload this separate challenger second: `{CANDIDATE_ZIP_NAME}`",
        "",
        "## Rationale",
        "- The Stage 1.5 winner improved the fixed platform-aligned validation objective, but it is still a new challenger rather than the incumbent shipped anchor.",
        "- Uploading it after the current anchor preserves the stable benchmark while giving the fixed-metric candidate a clean comparison slot on platform.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    prepared_data = load_deep_sets_data()
    config = DeepSetsConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_metric="platform_score",
    )
    loss_config = LossConfig(
        name="huber_huber",
        use_robust_viscosity_loss=True,
        use_robust_oxidation_loss=True,
        viscosity_delta=1.0,
        oxidation_delta=0.75,
    )
    prediction_frame = train_full_deep_sets_variant_ensemble_and_predict(
        prepared_data=prepared_data,
        variant=HybridVariant(
            name="hybrid_deep_sets_v2_family_only",
            use_component_embedding=False,
            use_tabular_branch=True,
        ),
        target_strategy_name="raw",
        seeds=ENSEMBLE_SEEDS,
        config=config,
        loss_config=loss_config,
    )

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    submission_frame = _to_submission_columns(prediction_frame)
    submission_frame.to_csv(CANDIDATE_PREDICTIONS_PATH, index=False, encoding="utf-8")

    validation_details = _build_validation_details(submission_frame)
    validate_predictions_csv(CANDIDATE_PREDICTIONS_PATH)
    zip_path = build_bundle_from_predictions(
        predictions_path=CANDIDATE_PREDICTIONS_PATH,
        zip_name=CANDIDATE_ZIP_NAME,
    )
    _write_text(build_report(validation_details, zip_path), REPORT_PATH)

    print(f"stage15_predictions: {CANDIDATE_PREDICTIONS_PATH}")
    print(f"stage15_zip: {zip_path}")
    print(f"stage15_report: {REPORT_PATH}")
    print(f"stage15_rows: {validation_details['row_count']}")


if __name__ == "__main__":
    main()
