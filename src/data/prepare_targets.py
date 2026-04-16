"""Scenario-level target table preparation for Daimler DOT train data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import SCENARIO_CONDITION_COLUMNS, TRAIN_TARGETS_OUTPUT_PATH, TRAIN_TARGET_COLUMNS


def _validate_constant_within_scenario(
    frame: pd.DataFrame, scenario_column: str, columns: Iterable[str]
) -> None:
    """Ensure scenario-level columns do not vary inside the same scenario."""

    invalid_columns: list[str] = []
    grouped = frame.groupby(scenario_column, dropna=False)
    for column in columns:
        cardinality = grouped[column].nunique(dropna=False)
        if (cardinality > 1).any():
            invalid_columns.append(column)

    if invalid_columns:
        joined = ", ".join(sorted(invalid_columns))
        raise ValueError(f"Scenario-level columns vary within scenario_id: {joined}")


def build_train_scenario_targets(train_mixtures: pd.DataFrame) -> pd.DataFrame:
    """Create one clean row per scenario_id with only targets and scenario conditions."""

    required = {"scenario_id", *SCENARIO_CONDITION_COLUMNS, *TRAIN_TARGET_COLUMNS}
    missing = required.difference(train_mixtures.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"Train mixtures are missing required columns: {joined}")

    retained_columns = ["scenario_id", *SCENARIO_CONDITION_COLUMNS, *TRAIN_TARGET_COLUMNS]
    _validate_constant_within_scenario(train_mixtures[retained_columns], "scenario_id", retained_columns[1:])

    scenario_targets = (
        train_mixtures.loc[:, retained_columns]
        .drop_duplicates(subset=["scenario_id"])
        .sort_values("scenario_id")
        .reset_index(drop=True)
    )

    if scenario_targets["scenario_id"].duplicated().any():
        raise ValueError("Prepared target table still contains duplicate scenario_id values.")

    return scenario_targets


def save_train_scenario_targets(
    scenario_targets: pd.DataFrame, output_path: Path = TRAIN_TARGETS_OUTPUT_PATH
) -> Path:
    """Persist the scenario-level training targets to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_targets.to_csv(output_path, index=False)
    return output_path
