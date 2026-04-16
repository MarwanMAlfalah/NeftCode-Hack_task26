"""Tests for grouped-CV baseline training and target strategies."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.train_baselines import (
    ModelSpec,
    PreparedBaselineData,
    build_target_strategies,
    load_baseline_training_data,
    run_baseline_cv,
)


class BaselinePipelineTest(unittest.TestCase):
    def test_target_strategies_roundtrip_and_keep_oxidation_scale(self) -> None:
        y = np.array([[0.0, 10.0], [5.0, 20.0], [100.0, 30.0]], dtype=float)
        strategies = {strategy.name: strategy for strategy in build_target_strategies()}

        raw = strategies["raw"]
        np.testing.assert_allclose(raw.inverse_transform(raw.transform(y)), y)

        asinh = strategies["viscosity_asinh"]
        transformed = asinh.transform(y)
        np.testing.assert_allclose(transformed[:, 1], y[:, 1])
        np.testing.assert_allclose(asinh.inverse_transform(transformed), y)

    def test_run_baseline_cv_returns_results_for_each_strategy(self) -> None:
        scenario_ids = [f"s{i:02d}" for i in range(12)]
        x1 = np.linspace(-2.0, 2.0, num=12)
        x2 = np.cos(x1)
        x3 = np.sin(x1)
        X = pd.DataFrame(
            {
                "feature_a": x1,
                "feature_b": x2,
                "feature_c": x3,
            }
        )
        y = pd.DataFrame(
            {
                "target_delta_kinematic_viscosity_pct": 2.0 * x1 + 0.5 * x2,
                "target_oxidation_eot_a_per_cm": -1.5 * x1 + 0.25 * x3,
            }
        )

        prepared = PreparedBaselineData(
            scenario_ids=pd.Series(scenario_ids),
            X=X,
            y=y,
            feature_manifest={"train_scenarios": 12, "feature_table_columns": ["scenario_id", *X.columns.tolist()], "feature_group_column_counts": {"synthetic": 3}},
        )

        ridge_spec = ModelSpec(
            name="ridge_smoke",
            description="Small deterministic ridge smoke test.",
            build_estimator=lambda seed: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("variance", VarianceThreshold()),
                    ("scaler", StandardScaler()),
                    ("model", MultiOutputRegressor(Ridge())),
                ]
            ),
            build_param_grid=lambda X_train, y_train: [{"model__estimator__alpha": [1.0]}],
        )

        summary, fold_metrics, report = run_baseline_cv(
            prepared_data=prepared,
            model_specs=[ridge_spec],
            target_strategies=build_target_strategies(),
            outer_splits=3,
            inner_splits=2,
            seed=7,
        )

        self.assertEqual(set(summary["target_strategy"]), {"raw", "viscosity_asinh"})
        self.assertEqual(len(fold_metrics), 6)
        self.assertIn("combined_score", fold_metrics.columns)
        self.assertIn("target_delta_kinematic_viscosity_pct__rmse", fold_metrics.columns)
        self.assertIn("target_oxidation_eot_a_per_cm__rmse", fold_metrics.columns)
        self.assertIn("Best Current Baseline", report)

    def test_load_baseline_training_data_rejects_target_columns_in_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_features_path = root / "train_features.csv"
            train_targets_path = root / "train_targets.csv"
            manifest_path = root / "feature_manifest.json"

            pd.DataFrame(
                {
                    "scenario_id": ["s1", "s2"],
                    "feature_a": [1.0, 2.0],
                    "target_delta_kinematic_viscosity_pct": [99.0, 100.0],
                }
            ).to_csv(train_features_path, index=False)
            pd.DataFrame(
                {
                    "scenario_id": ["s1", "s2"],
                    "target_delta_kinematic_viscosity_pct": [1.0, 2.0],
                    "target_oxidation_eot_a_per_cm": [3.0, 4.0],
                }
            ).to_csv(train_targets_path, index=False)
            manifest_path.write_text(json.dumps({}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contains target columns"):
                load_baseline_training_data(
                    train_features_path=train_features_path,
                    train_targets_path=train_targets_path,
                    feature_manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
