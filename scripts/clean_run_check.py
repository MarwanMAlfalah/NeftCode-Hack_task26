"""Run a clean local regeneration check for the official submission artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.package_submission import PREDICTIONS_PATH, validate_predictions_csv


def main() -> None:
    backup_path: Path | None = None
    if PREDICTIONS_PATH.exists():
        temp_dir = Path(tempfile.mkdtemp(prefix="neftekod_predictions_backup_"))
        backup_path = temp_dir / PREDICTIONS_PATH.name
        shutil.move(str(PREDICTIONS_PATH), str(backup_path))

    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "inference.ipynb",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        info = validate_predictions_csv(PREDICTIONS_PATH)
        print(f"clean_run_predictions: {info['path']}")
        print(f"clean_run_rows: {info['row_count']}")
    except Exception:
        if backup_path is not None and not PREDICTIONS_PATH.exists():
            shutil.move(str(backup_path), str(PREDICTIONS_PATH))
        raise
    else:
        if backup_path is not None and backup_path.exists():
            backup_path.unlink()
            backup_path.parent.rmdir()


if __name__ == "__main__":
    main()
