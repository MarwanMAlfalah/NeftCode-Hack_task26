"""Utilities for loading and normalizing raw Daimler DOT CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd

from src.config import (
    COMPONENT_PROPERTIES_PATH,
    MIXTURE_COLUMN_RENAMES,
    MIXTURE_NUMERIC_COLUMNS,
    PROPERTY_COLUMN_RENAMES,
    TEST_MIXTURES_PATH,
    TRAIN_MIXTURES_PATH,
    TRAIN_TARGET_COLUMN_RENAMES,
    TRAIN_TARGET_COLUMNS,
)


@dataclass(frozen=True)
class RawDatasets:
    """Container for the three normalized raw input tables."""

    train_mixtures: pd.DataFrame
    test_mixtures: pd.DataFrame
    component_properties: pd.DataFrame


def strip_utf8_bom(value: str) -> str:
    """Remove a UTF-8 BOM from the beginning of a string if present."""

    return value.lstrip("\ufeff")


def to_snake_case(value: str) -> str:
    """Convert an arbitrary label to a stable snake_case identifier."""

    text = strip_utf8_bom(str(value)).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed_column"


def normalize_headers(columns: Iterable[str], rename_map: Mapping[str, str]) -> list[str]:
    """Normalize raw headers using an explicit map with a snake_case fallback."""

    normalized: list[str] = []
    for column in columns:
        stripped = strip_utf8_bom(str(column)).strip()
        normalized.append(rename_map.get(stripped, to_snake_case(stripped)))
    return normalized


def normalize_string_series(series: pd.Series) -> pd.Series:
    """Trim whitespace from string cells and convert blanks to missing values."""

    normalized = series.astype("string").str.strip()
    return normalized.replace(to_replace=r"^\s*$", value=pd.NA, regex=True)


def coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Safely coerce selected columns to numeric dtype."""

    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def load_csv_with_normalized_columns(
    path: Path, rename_map: Mapping[str, str], numeric_columns: Iterable[str]
) -> pd.DataFrame:
    """Read a raw CSV and return a normalized DataFrame with safe type coercion."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    frame.columns = normalize_headers(frame.columns, rename_map)

    for column in frame.columns:
        if str(frame[column].dtype) in {"object", "string", "str"}:
            frame[column] = normalize_string_series(frame[column])

    frame = coerce_numeric_columns(frame, numeric_columns)
    return frame


def load_raw_train_mixtures(path: Path = TRAIN_MIXTURES_PATH) -> pd.DataFrame:
    """Load and normalize the raw training mixture rows."""

    numeric_columns = [*MIXTURE_NUMERIC_COLUMNS, *TRAIN_TARGET_COLUMNS]
    rename_map = {**MIXTURE_COLUMN_RENAMES, **TRAIN_TARGET_COLUMN_RENAMES}
    return load_csv_with_normalized_columns(path, rename_map, numeric_columns)


def load_raw_test_mixtures(path: Path = TEST_MIXTURES_PATH) -> pd.DataFrame:
    """Load and normalize the raw test mixture rows."""

    return load_csv_with_normalized_columns(path, MIXTURE_COLUMN_RENAMES, MIXTURE_NUMERIC_COLUMNS)


def load_raw_component_properties(path: Path = COMPONENT_PROPERTIES_PATH) -> pd.DataFrame:
    """Load and normalize the raw component-properties table."""

    frame = load_csv_with_normalized_columns(path, PROPERTY_COLUMN_RENAMES, numeric_columns=[])
    frame["property_value_numeric"] = pd.to_numeric(frame["property_value"], errors="coerce")
    return frame


def load_all_raw_datasets() -> RawDatasets:
    """Load all raw datasets with a shared normalization policy."""

    return RawDatasets(
        train_mixtures=load_raw_train_mixtures(),
        test_mixtures=load_raw_test_mixtures(),
        component_properties=load_raw_component_properties(),
    )
