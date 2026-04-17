"""Shape tests for the compact Deep Sets pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.models.deep_sets import (
    DeepSetsRegressor,
    FeatureNormalizer,
    TargetScaler,
    align_tabular_features,
    build_deep_sets_schema,
    build_scenario_tensor_data,
)


def _build_frame(include_targets: bool) -> pd.DataFrame:
    rows = [
        {
            "scenario_id": "s1",
            "component_id": "BaseOil_1",
            "batch_id": "b1",
            "mass_fraction_pct": 70.0,
            "test_temperature_c": 160.0,
            "test_duration_h": 168.0,
            "biofuel_mass_fraction_pct": 0.0,
            "catalyst_dosage_category": 1,
            "batch_id_was_missing": False,
            "used_exact_batch_props": True,
            "has_usable_property_coverage": True,
            "used_typical_fallback": False,
            "missing_all_props": False,
            "property_join_source": "exact_only",
            "prop__density": 0.90,
            "prop__vis40": 50.0,
        },
        {
            "scenario_id": "s1",
            "component_id": "Additive_1",
            "batch_id": "b2",
            "mass_fraction_pct": 30.0,
            "test_temperature_c": 160.0,
            "test_duration_h": 168.0,
            "biofuel_mass_fraction_pct": 0.0,
            "catalyst_dosage_category": 1,
            "batch_id_was_missing": False,
            "used_exact_batch_props": False,
            "has_usable_property_coverage": True,
            "used_typical_fallback": True,
            "missing_all_props": False,
            "property_join_source": "typical_only",
            "prop__density": 1.10,
            "prop__vis40": np.nan,
        },
        {
            "scenario_id": "s2",
            "component_id": "BaseOil_2",
            "batch_id": "b3",
            "mass_fraction_pct": 100.0,
            "test_temperature_c": 150.0,
            "test_duration_h": 96.0,
            "biofuel_mass_fraction_pct": 5.0,
            "catalyst_dosage_category": 2,
            "batch_id_was_missing": True,
            "used_exact_batch_props": False,
            "has_usable_property_coverage": False,
            "used_typical_fallback": False,
            "missing_all_props": True,
            "property_join_source": "missing",
            "prop__density": np.nan,
            "prop__vis40": np.nan,
        },
    ]
    frame = pd.DataFrame(rows)
    if include_targets:
        frame["target_delta_kinematic_viscosity_pct"] = [10.0, 10.0, 20.0]
        frame["target_oxidation_eot_a_per_cm"] = [90.0, 90.0, 80.0]
    return frame


class DeepSetsShapeTest(unittest.TestCase):
    def test_tensor_builder_and_hybrid_model_forward_shapes(self) -> None:
        train_frame = _build_frame(include_targets=True)
        test_frame = _build_frame(include_targets=False)
        tabular_feature_frame = pd.DataFrame(
            {
                "scenario_id": ["s1", "s2"],
                "test_temperature_c": [160.0, 150.0],
                "component_row_count": [2.0, 1.0],
                "family__baseoil__mass_share": [0.7, 1.0],
            }
        )
        tabular_columns = ["test_temperature_c", "component_row_count", "family__baseoil__mass_share"]
        schema = build_deep_sets_schema(
            train_frame=train_frame,
            test_frame=test_frame,
            tabular_feature_columns=tabular_columns,
        )
        data = build_scenario_tensor_data(train_frame, schema=schema, include_targets=True).with_tabular_features(
            align_tabular_features(
                scenario_ids=np.asarray(["s1", "s2"], dtype=object),
                feature_frame=tabular_feature_frame,
                feature_columns=tabular_columns,
            )
        )

        self.assertEqual(data.family_ids.shape, (2, 2))
        self.assertEqual(data.component_ids.shape, (2, 2))
        self.assertEqual(data.mass_fraction.shape, (2, 2, 1))
        self.assertEqual(data.property_values.shape, (2, 2, 2))
        self.assertEqual(data.property_mask.shape, (2, 2, 2))
        self.assertEqual(data.component_flags.shape[0:2], (2, 2))
        self.assertEqual(data.conditions.shape, (2, 3))
        self.assertEqual(data.tabular_features.shape, (2, 3))
        self.assertEqual(data.targets.shape, (2, 2))

        normalized = FeatureNormalizer.fit(data).transform(data)
        target_scaler = TargetScaler.fit(normalized.targets)
        normalized = normalized.with_targets(target_scaler.transform(normalized.targets))

        model = DeepSetsRegressor(schema=schema, use_component_embedding=False, use_tabular_branch=True)
        batch = {
            "family_ids": torch.as_tensor(normalized.family_ids, dtype=torch.long),
            "component_ids": torch.as_tensor(normalized.component_ids, dtype=torch.long),
            "mass_fraction": torch.as_tensor(normalized.mass_fraction, dtype=torch.float32),
            "property_values": torch.as_tensor(normalized.property_values, dtype=torch.float32),
            "property_mask": torch.as_tensor(normalized.property_mask, dtype=torch.float32),
            "component_flags": torch.as_tensor(normalized.component_flags, dtype=torch.float32),
            "component_mask": torch.as_tensor(normalized.component_mask, dtype=torch.float32),
            "conditions": torch.as_tensor(normalized.conditions, dtype=torch.float32),
            "catalyst_ids": torch.as_tensor(normalized.catalyst_ids, dtype=torch.long),
            "tabular_features": torch.as_tensor(normalized.tabular_features, dtype=torch.float32),
        }
        predictions = model(batch)
        self.assertEqual(tuple(predictions.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
