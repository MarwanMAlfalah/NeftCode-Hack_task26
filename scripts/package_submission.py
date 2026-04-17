"""Create and validate the final competition submission bundle."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import TEST_SCENARIO_FEATURES_OUTPUT_PATH


PREDICTIONS_PATH = REPO_ROOT / "outputs" / "predictions.csv"
NOTEBOOK_PATH = REPO_ROOT / "inference.ipynb"
SUBMISSIONS_DIR = REPO_ROOT / "outputs" / "submissions"
DEFAULT_ZIP_NAME = "neftekod_dot_submission_final.zip"
REQUIRED_ROOT_FILES = ("predictions.csv", "inference.ipynb")
REQUIRED_COLUMNS = [
    "scenario_id",
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm",
]


def _assert_ascii_name(path: Path) -> None:
    try:
        path.name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Filename must be ASCII-safe: {path.name}") from exc


def _load_expected_test_ids() -> pd.Series:
    frame = pd.read_csv(TEST_SCENARIO_FEATURES_OUTPUT_PATH)
    if frame["scenario_id"].duplicated().any():
        raise ValueError("Processed test feature table contains duplicate scenario_id values.")
    return frame["scenario_id"].sort_values(ignore_index=True)


def validate_predictions_csv(path: Path = PREDICTIONS_PATH) -> dict[str, object]:
    """Validate the official predictions CSV against submission requirements."""

    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    _assert_ascii_name(path)

    raw_bytes = path.read_bytes()
    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("predictions.csv is not valid UTF-8.") from exc

    frame = pd.read_csv(path)
    if list(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            "predictions.csv columns do not match the required submission schema. "
            f"Expected {REQUIRED_COLUMNS}, got {list(frame.columns)}."
        )
    if frame.isnull().any().any():
        raise ValueError("predictions.csv contains null values.")
    if frame["scenario_id"].duplicated().any():
        raise ValueError("predictions.csv contains duplicate scenario_id values.")

    expected_ids = _load_expected_test_ids()
    actual_ids = frame["scenario_id"].sort_values(ignore_index=True)
    if len(frame) != len(expected_ids):
        raise ValueError(
            f"predictions.csv row count mismatch. Expected {len(expected_ids)}, got {len(frame)}."
        )
    if not actual_ids.equals(expected_ids):
        missing_ids = expected_ids[~expected_ids.isin(actual_ids)].tolist()
        extra_ids = actual_ids[~actual_ids.isin(expected_ids)].tolist()
        raise ValueError(
            "predictions.csv test IDs do not match the expected test scenarios. "
            f"Missing: {missing_ids[:5]}, extra: {extra_ids[:5]}."
        )

    return {
        "path": str(path),
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "encoding": "utf-8",
    }


def validate_bundle(zip_path: Path) -> dict[str, object]:
    """Validate the final ZIP structure and the bundled predictions CSV."""

    if not zip_path.exists():
        raise FileNotFoundError(f"Bundle not found: {zip_path}")
    _assert_ascii_name(zip_path)

    with ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        root_names = [name for name in names if not name.endswith("/")]
        csv_names = [name for name in root_names if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Bundle must contain exactly one CSV file, found {csv_names}.")
        if csv_names[0] != "predictions.csv":
            raise ValueError(f"The only CSV at ZIP root must be predictions.csv, found {csv_names[0]}.")
        required_missing = [name for name in REQUIRED_ROOT_FILES if name not in root_names]
        if required_missing:
            raise ValueError(f"Bundle is missing required root files: {required_missing}.")
        nested_entries = [name for name in root_names if "/" in name.strip("/")]
        if nested_entries:
            raise ValueError(f"Bundle contains nested files; required files must be at ZIP root: {nested_entries}")
        for name in root_names:
            try:
                name.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(f"Non-ASCII filename found in bundle: {name}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_predictions = Path(temp_dir) / "predictions.csv"
            extracted_predictions.write_bytes(archive.read("predictions.csv"))
            csv_info = validate_predictions_csv(extracted_predictions)

    return {
        "zip_path": str(zip_path),
        "files": list(REQUIRED_ROOT_FILES),
        "csv_validation": csv_info,
    }


def build_bundle(zip_name: str = DEFAULT_ZIP_NAME) -> Path:
    """Assemble the official root-level bundle and write a final ZIP."""

    predictions_info = validate_predictions_csv(PREDICTIONS_PATH)
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Missing notebook: {NOTEBOOK_PATH}")
    _assert_ascii_name(NOTEBOOK_PATH)

    if not zip_name.lower().endswith(".zip"):
        raise ValueError("Final bundle filename must end with .zip")
    try:
        zip_name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Final bundle filename must be ASCII-safe.") from exc

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    final_zip_path = SUBMISSIONS_DIR / zip_name

    with tempfile.TemporaryDirectory() as temp_dir:
        bundle_dir = Path(temp_dir) / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PREDICTIONS_PATH, bundle_dir / "predictions.csv")
        shutil.copy2(NOTEBOOK_PATH, bundle_dir / "inference.ipynb")

        with ZipFile(final_zip_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.write(bundle_dir / "predictions.csv", arcname="predictions.csv")
            archive.write(bundle_dir / "inference.ipynb", arcname="inference.ipynb")

    validate_bundle(final_zip_path)
    print(f"predictions_rows: {predictions_info['row_count']}")
    print(f"submission_zip: {final_zip_path}")
    return final_zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-name",
        default=DEFAULT_ZIP_NAME,
        help="ASCII-safe ZIP filename to create inside outputs/submissions.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing ZIP instead of building a new one.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="ZIP path to validate when using --validate-only. Defaults to the standard output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        zip_path = args.zip_path or (SUBMISSIONS_DIR / args.zip_name)
        info = validate_bundle(zip_path)
        print(f"validated_zip: {info['zip_path']}")
        return

    build_bundle(zip_name=args.zip_name)


if __name__ == "__main__":
    main()
