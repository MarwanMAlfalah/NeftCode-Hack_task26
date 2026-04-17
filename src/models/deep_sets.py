"""Compact Deep Sets and hybrid tensor utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import TRAIN_TARGET_COLUMNS
from src.features.build_scenario_features import extract_component_family, validate_prepared_scenarios


PROPERTY_PREFIX = "prop__"
ROW_FLAG_COLUMNS = [
    "batch_id_was_missing",
    "used_exact_batch_props",
    "has_usable_property_coverage",
    "used_typical_fallback",
    "missing_all_props",
]
PROPERTY_SOURCE_COLUMNS = [
    "property_source_exact_only",
    "property_source_exact_plus_typical",
    "property_source_typical_only",
]
COMPONENT_FLAG_COLUMNS = [*ROW_FLAG_COLUMNS, *PROPERTY_SOURCE_COLUMNS]


@dataclass(frozen=True)
class DeepSetsSchema:
    """Stable categorical vocabularies and component layout limits."""

    property_columns: list[str]
    family_to_index: dict[str, int]
    component_to_index: dict[str, int]
    catalyst_to_index: dict[int, int]
    max_components: int
    tabular_feature_columns: list[str]

    @property
    def family_vocab_size(self) -> int:
        return max(self.family_to_index.values(), default=0) + 1

    @property
    def component_vocab_size(self) -> int:
        return max(self.component_to_index.values(), default=0) + 1

    @property
    def catalyst_vocab_size(self) -> int:
        return max(self.catalyst_to_index.values(), default=0) + 1

    @property
    def tabular_input_dim(self) -> int:
        return len(self.tabular_feature_columns)


@dataclass(frozen=True)
class ScenarioTensorData:
    """Padded scenario-level tensors stored as numpy arrays."""

    scenario_ids: np.ndarray
    family_ids: np.ndarray
    component_ids: np.ndarray
    mass_fraction: np.ndarray
    property_values: np.ndarray
    property_mask: np.ndarray
    component_flags: np.ndarray
    component_mask: np.ndarray
    conditions: np.ndarray
    catalyst_ids: np.ndarray
    tabular_features: np.ndarray | None = None
    targets: np.ndarray | None = None

    def subset(self, indices: Sequence[int]) -> "ScenarioTensorData":
        """Return a scenario subset without mutating the source arrays."""

        array_indices = np.asarray(indices, dtype=int)
        tabular_features = None
        if self.tabular_features is not None:
            tabular_features = self.tabular_features[array_indices].copy()
        targets = None if self.targets is None else self.targets[array_indices].copy()
        return ScenarioTensorData(
            scenario_ids=self.scenario_ids[array_indices].copy(),
            family_ids=self.family_ids[array_indices].copy(),
            component_ids=self.component_ids[array_indices].copy(),
            mass_fraction=self.mass_fraction[array_indices].copy(),
            property_values=self.property_values[array_indices].copy(),
            property_mask=self.property_mask[array_indices].copy(),
            component_flags=self.component_flags[array_indices].copy(),
            component_mask=self.component_mask[array_indices].copy(),
            conditions=self.conditions[array_indices].copy(),
            catalyst_ids=self.catalyst_ids[array_indices].copy(),
            tabular_features=tabular_features,
            targets=targets,
        )

    def with_targets(self, targets: np.ndarray | None) -> "ScenarioTensorData":
        """Return a copy with replaced target values."""

        return ScenarioTensorData(
            scenario_ids=self.scenario_ids.copy(),
            family_ids=self.family_ids.copy(),
            component_ids=self.component_ids.copy(),
            mass_fraction=self.mass_fraction.copy(),
            property_values=self.property_values.copy(),
            property_mask=self.property_mask.copy(),
            component_flags=self.component_flags.copy(),
            component_mask=self.component_mask.copy(),
            conditions=self.conditions.copy(),
            catalyst_ids=self.catalyst_ids.copy(),
            tabular_features=None if self.tabular_features is None else self.tabular_features.copy(),
            targets=None if targets is None else np.asarray(targets, dtype=np.float32).copy(),
        )

    def with_tabular_features(self, tabular_features: np.ndarray | None) -> "ScenarioTensorData":
        """Return a copy with scenario-level tabular features attached."""

        return ScenarioTensorData(
            scenario_ids=self.scenario_ids.copy(),
            family_ids=self.family_ids.copy(),
            component_ids=self.component_ids.copy(),
            mass_fraction=self.mass_fraction.copy(),
            property_values=self.property_values.copy(),
            property_mask=self.property_mask.copy(),
            component_flags=self.component_flags.copy(),
            component_mask=self.component_mask.copy(),
            conditions=self.conditions.copy(),
            catalyst_ids=self.catalyst_ids.copy(),
            tabular_features=None if tabular_features is None else np.asarray(tabular_features, dtype=np.float32).copy(),
            targets=None if self.targets is None else self.targets.copy(),
        )

    def __len__(self) -> int:
        return int(len(self.scenario_ids))


@dataclass(frozen=True)
class FeatureNormalizer:
    """Fold-local normalization parameters for numeric model inputs."""

    mass_mean: float
    mass_std: float
    property_mean: np.ndarray
    property_std: np.ndarray
    condition_mean: np.ndarray
    condition_std: np.ndarray
    tabular_mean: np.ndarray | None
    tabular_std: np.ndarray | None

    @classmethod
    def fit(cls, data: ScenarioTensorData) -> "FeatureNormalizer":
        """Estimate normalization statistics from training scenarios only."""

        valid_components = data.component_mask > 0
        mass_values = data.mass_fraction[..., 0][valid_components]
        if mass_values.size == 0:
            mass_mean = 0.0
            mass_std = 1.0
        else:
            mass_mean = float(np.nanmean(mass_values))
            mass_std = float(np.nanstd(mass_values))
            if not np.isfinite(mass_std) or mass_std <= 0:
                mass_std = 1.0

        property_mean = np.zeros(data.property_values.shape[-1], dtype=np.float32)
        property_std = np.ones(data.property_values.shape[-1], dtype=np.float32)
        for feature_index in range(data.property_values.shape[-1]):
            observed = data.property_values[..., feature_index][data.property_mask[..., feature_index] > 0]
            if observed.size == 0:
                continue
            mean = float(np.nanmean(observed))
            std = float(np.nanstd(observed))
            property_mean[feature_index] = mean if np.isfinite(mean) else 0.0
            property_std[feature_index] = std if np.isfinite(std) and std > 0 else 1.0

        condition_mean = np.nanmean(data.conditions, axis=0).astype(np.float32)
        condition_std = np.nanstd(data.conditions, axis=0).astype(np.float32)
        condition_mean = np.where(np.isfinite(condition_mean), condition_mean, 0.0)
        condition_std = np.where(np.isfinite(condition_std) & (condition_std > 0), condition_std, 1.0)

        tabular_mean = None
        tabular_std = None
        if data.tabular_features is not None:
            tabular_mean = np.nanmean(data.tabular_features, axis=0).astype(np.float32)
            tabular_std = np.nanstd(data.tabular_features, axis=0).astype(np.float32)
            tabular_mean = np.where(np.isfinite(tabular_mean), tabular_mean, 0.0)
            tabular_std = np.where(np.isfinite(tabular_std) & (tabular_std > 0), tabular_std, 1.0)

        return cls(
            mass_mean=mass_mean,
            mass_std=mass_std,
            property_mean=property_mean,
            property_std=property_std,
            condition_mean=condition_mean,
            condition_std=condition_std,
            tabular_mean=tabular_mean,
            tabular_std=tabular_std,
        )

    def transform(self, data: ScenarioTensorData) -> ScenarioTensorData:
        """Apply training-fold normalization while preserving masks."""

        mass = data.mass_fraction.astype(np.float32).copy()
        mass[..., 0] = ((mass[..., 0] - self.mass_mean) / self.mass_std) * data.component_mask

        properties = data.property_values.astype(np.float32).copy()
        properties = (properties - self.property_mean.reshape(1, 1, -1)) / self.property_std.reshape(1, 1, -1)
        properties = properties * data.property_mask

        conditions = data.conditions.astype(np.float32).copy()
        conditions = (conditions - self.condition_mean.reshape(1, -1)) / self.condition_std.reshape(1, -1)
        conditions = np.nan_to_num(conditions, nan=0.0, posinf=0.0, neginf=0.0)

        tabular_features = None
        if data.tabular_features is not None and self.tabular_mean is not None and self.tabular_std is not None:
            tabular_features = data.tabular_features.astype(np.float32).copy()
            tabular_features = (tabular_features - self.tabular_mean.reshape(1, -1)) / self.tabular_std.reshape(1, -1)
            tabular_features = np.nan_to_num(tabular_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return ScenarioTensorData(
            scenario_ids=data.scenario_ids.copy(),
            family_ids=data.family_ids.copy(),
            component_ids=data.component_ids.copy(),
            mass_fraction=mass,
            property_values=properties.astype(np.float32),
            property_mask=data.property_mask.astype(np.float32).copy(),
            component_flags=data.component_flags.astype(np.float32).copy(),
            component_mask=data.component_mask.astype(np.float32).copy(),
            conditions=conditions.astype(np.float32),
            catalyst_ids=data.catalyst_ids.copy(),
            tabular_features=tabular_features,
            targets=None if data.targets is None else data.targets.astype(np.float32).copy(),
        )


@dataclass(frozen=True)
class TargetScaler:
    """Standardize transformed targets for stable optimization."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        mean = np.nanmean(targets, axis=0).astype(np.float32)
        std = np.nanstd(targets, axis=0).astype(np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return cls(mean=mean, std=std)

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return ((targets - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)).astype(np.float32)

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        return (targets * self.std.reshape(1, -1) + self.mean.reshape(1, -1)).astype(np.float32)


def set_torch_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducible CPU training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_deep_sets_schema(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    tabular_feature_columns: Sequence[str] | None = None,
) -> DeepSetsSchema:
    """Infer stable vocabularies and padding limits from prepared row-level data."""

    combined = pd.concat([train_frame, test_frame], ignore_index=True, sort=False)
    property_columns = sorted(column for column in combined.columns if column.startswith(PROPERTY_PREFIX))

    family_values = sorted(
        {
            extract_component_family(component_id)
            for component_id in combined["component_id"].dropna().tolist()
        }
    )
    family_to_index = {"__padding__": 0, "__unknown__": 1}
    for value in family_values:
        if value not in family_to_index:
            family_to_index[value] = len(family_to_index)

    component_values = sorted({str(component_id) for component_id in combined["component_id"].dropna().tolist()})
    component_to_index = {"__padding__": 0, "__unknown__": 1}
    for value in component_values:
        if value not in component_to_index:
            component_to_index[value] = len(component_to_index)

    catalyst_values = sorted(
        {
            int(value)
            for value in pd.to_numeric(combined["catalyst_dosage_category"], errors="coerce").dropna().tolist()
        }
    )
    catalyst_to_index = {-1: 0}
    for value in catalyst_values:
        catalyst_to_index[value] = len(catalyst_to_index)

    max_components = int(
        max(
            train_frame.groupby("scenario_id", dropna=False).size().max(),
            test_frame.groupby("scenario_id", dropna=False).size().max(),
        )
    )
    return DeepSetsSchema(
        property_columns=property_columns,
        family_to_index=family_to_index,
        component_to_index=component_to_index,
        catalyst_to_index=catalyst_to_index,
        max_components=max_components,
        tabular_feature_columns=list(tabular_feature_columns or []),
    )


def align_tabular_features(
    scenario_ids: np.ndarray,
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Align scenario-level tabular features to the padded tensor scenario order."""

    aligned = (
        feature_frame.loc[:, ["scenario_id", *feature_columns]]
        .drop_duplicates(subset=["scenario_id"])
        .set_index("scenario_id")
        .reindex(scenario_ids)
    )
    numeric = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


def build_scenario_tensor_data(
    frame: pd.DataFrame,
    schema: DeepSetsSchema,
    include_targets: bool = True,
) -> ScenarioTensorData:
    """Convert prepared component rows into padded scenario-level arrays."""

    validate_prepared_scenarios(frame)
    prepared = frame.copy()
    prepared["component_family"] = prepared["component_id"].map(extract_component_family)
    prepared["catalyst_dosage_category"] = pd.to_numeric(
        prepared["catalyst_dosage_category"], errors="coerce"
    )

    for column in ROW_FLAG_COLUMNS:
        if column in prepared.columns:
            prepared[column] = (
                prepared[column]
                .astype("string")
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
                .astype(bool)
            )
        else:
            prepared[column] = False

    join_source = prepared.get("property_join_source", pd.Series(["missing"] * len(prepared), index=prepared.index))
    prepared["property_source_exact_only"] = join_source.eq("exact_only")
    prepared["property_source_exact_plus_typical"] = join_source.eq("exact_plus_typical")
    prepared["property_source_typical_only"] = join_source.eq("typical_only")

    scenario_ids = np.asarray(sorted(prepared["scenario_id"].dropna().unique()), dtype=object)
    n_scenarios = len(scenario_ids)
    max_components = schema.max_components
    n_properties = len(schema.property_columns)
    n_flags = len(COMPONENT_FLAG_COLUMNS)

    family_ids = np.zeros((n_scenarios, max_components), dtype=np.int64)
    component_ids = np.zeros((n_scenarios, max_components), dtype=np.int64)
    mass_fraction = np.zeros((n_scenarios, max_components, 1), dtype=np.float32)
    property_values = np.zeros((n_scenarios, max_components, n_properties), dtype=np.float32)
    property_mask = np.zeros((n_scenarios, max_components, n_properties), dtype=np.float32)
    component_flags = np.zeros((n_scenarios, max_components, n_flags), dtype=np.float32)
    component_mask = np.zeros((n_scenarios, max_components), dtype=np.float32)
    conditions = np.zeros((n_scenarios, 3), dtype=np.float32)
    catalyst_ids = np.zeros(n_scenarios, dtype=np.int64)
    targets = None
    if include_targets:
        targets = np.zeros((n_scenarios, len(TRAIN_TARGET_COLUMNS)), dtype=np.float32)

    property_column_index = {column: index for index, column in enumerate(schema.property_columns)}

    for scenario_offset, scenario_id in enumerate(scenario_ids):
        scenario_rows = prepared.loc[prepared["scenario_id"] == scenario_id].copy()
        scenario_rows["component_sort_key"] = scenario_rows["component_id"].astype("string")
        scenario_rows["batch_sort_key"] = scenario_rows["batch_id"].astype("string")
        scenario_rows = scenario_rows.sort_values(
            ["mass_fraction_pct", "component_sort_key", "batch_sort_key"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        first_row = scenario_rows.iloc[0]
        conditions[scenario_offset] = np.asarray(
            [
                pd.to_numeric(first_row["test_temperature_c"], errors="coerce"),
                pd.to_numeric(first_row["test_duration_h"], errors="coerce"),
                pd.to_numeric(first_row["biofuel_mass_fraction_pct"], errors="coerce"),
            ],
            dtype=np.float32,
        )
        catalyst_value = pd.to_numeric(first_row["catalyst_dosage_category"], errors="coerce")
        catalyst_ids[scenario_offset] = schema.catalyst_to_index.get(int(catalyst_value), 0) if pd.notna(catalyst_value) else 0

        if include_targets and targets is not None:
            targets[scenario_offset] = scenario_rows.loc[0, TRAIN_TARGET_COLUMNS].to_numpy(dtype=np.float32)

        for component_offset, (_, row) in enumerate(scenario_rows.iterrows()):
            if component_offset >= max_components:
                break

            family_key = row["component_family"] if pd.notna(row["component_family"]) else "__unknown__"
            component_key = str(row["component_id"]) if pd.notna(row["component_id"]) else "__unknown__"
            family_ids[scenario_offset, component_offset] = schema.family_to_index.get(family_key, 1)
            component_ids[scenario_offset, component_offset] = schema.component_to_index.get(component_key, 1)
            mass_fraction[scenario_offset, component_offset, 0] = float(
                pd.to_numeric(row["mass_fraction_pct"], errors="coerce")
            )
            component_mask[scenario_offset, component_offset] = 1.0

            for flag_offset, flag_column in enumerate(COMPONENT_FLAG_COLUMNS):
                component_flags[scenario_offset, component_offset, flag_offset] = float(bool(row.get(flag_column, False)))

            for column, property_index in property_column_index.items():
                value = pd.to_numeric(row.get(column), errors="coerce")
                if pd.notna(value):
                    property_values[scenario_offset, component_offset, property_index] = float(value)
                    property_mask[scenario_offset, component_offset, property_index] = 1.0

    return ScenarioTensorData(
        scenario_ids=scenario_ids,
        family_ids=family_ids,
        component_ids=component_ids,
        mass_fraction=mass_fraction,
        property_values=property_values,
        property_mask=property_mask,
        component_flags=component_flags,
        component_mask=component_mask,
        conditions=np.nan_to_num(conditions, nan=0.0, posinf=0.0, neginf=0.0),
        catalyst_ids=catalyst_ids,
        tabular_features=None,
        targets=targets,
    )


class ScenarioDataset(Dataset):
    """Torch dataset wrapper around padded scenario tensors."""

    def __init__(self, data: ScenarioTensorData) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "family_ids": torch.as_tensor(self.data.family_ids[index], dtype=torch.long),
            "component_ids": torch.as_tensor(self.data.component_ids[index], dtype=torch.long),
            "mass_fraction": torch.as_tensor(self.data.mass_fraction[index], dtype=torch.float32),
            "property_values": torch.as_tensor(self.data.property_values[index], dtype=torch.float32),
            "property_mask": torch.as_tensor(self.data.property_mask[index], dtype=torch.float32),
            "component_flags": torch.as_tensor(self.data.component_flags[index], dtype=torch.float32),
            "component_mask": torch.as_tensor(self.data.component_mask[index], dtype=torch.float32),
            "conditions": torch.as_tensor(self.data.conditions[index], dtype=torch.float32),
            "catalyst_ids": torch.as_tensor(self.data.catalyst_ids[index], dtype=torch.long),
        }
        if self.data.tabular_features is not None:
            item["tabular_features"] = torch.as_tensor(self.data.tabular_features[index], dtype=torch.float32)
        if self.data.targets is not None:
            item["targets"] = torch.as_tensor(self.data.targets[index], dtype=torch.float32)
        return item


def build_dataloader(
    data: ScenarioTensorData,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a dataloader for scenario tensors."""

    return DataLoader(
        ScenarioDataset(data),
        batch_size=min(batch_size, max(1, len(data))),
        shuffle=shuffle,
        drop_last=False,
    )


class MLPBlock(nn.Module):
    """Small feed-forward block used across the model."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DeepSetsRegressor(nn.Module):
    """Compact Deep Sets regressor with optional tabular and component-id branches."""

    def __init__(
        self,
        schema: DeepSetsSchema,
        property_projection_dim: int = 16,
        family_embedding_dim: int = 8,
        component_embedding_dim: int = 8,
        element_hidden_dim: int = 64,
        condition_hidden_dim: int = 32,
        tabular_hidden_dim: int = 48,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_component_embedding: bool = True,
        use_tabular_branch: bool = True,
    ) -> None:
        super().__init__()
        self.use_component_embedding = use_component_embedding
        self.use_tabular_branch = use_tabular_branch and schema.tabular_input_dim > 0

        self.family_embedding = nn.Embedding(schema.family_vocab_size, family_embedding_dim, padding_idx=0)
        if self.use_component_embedding:
            self.component_embedding = nn.Embedding(schema.component_vocab_size, component_embedding_dim, padding_idx=0)
        else:
            self.component_embedding = None
            component_embedding_dim = 0

        self.catalyst_embedding = nn.Embedding(schema.catalyst_vocab_size, 4, padding_idx=0)

        property_input_dim = len(schema.property_columns) * 2 + len(COMPONENT_FLAG_COLUMNS)
        self.property_encoder = MLPBlock(
            input_dim=property_input_dim,
            hidden_dims=[max(property_projection_dim, 16)],
            output_dim=property_projection_dim,
            dropout=dropout,
        )

        element_input_dim = family_embedding_dim + component_embedding_dim + 1 + property_projection_dim + len(COMPONENT_FLAG_COLUMNS)
        self.element_encoder = MLPBlock(
            input_dim=element_input_dim,
            hidden_dims=[element_hidden_dim],
            output_dim=element_hidden_dim,
            dropout=dropout,
        )

        self.condition_encoder = MLPBlock(
            input_dim=3 + 4,
            hidden_dims=[condition_hidden_dim],
            output_dim=condition_hidden_dim,
            dropout=dropout,
        )

        if self.use_tabular_branch:
            self.tabular_encoder = MLPBlock(
                input_dim=schema.tabular_input_dim,
                hidden_dims=[tabular_hidden_dim],
                output_dim=tabular_hidden_dim,
                dropout=dropout,
            )
            tabular_output_dim = tabular_hidden_dim
        else:
            self.tabular_encoder = None
            tabular_output_dim = 0

        self.fusion_head = MLPBlock(
            input_dim=2 * element_hidden_dim + condition_hidden_dim + tabular_output_dim,
            hidden_dims=[fusion_hidden_dim],
            output_dim=len(TRAIN_TARGET_COLUMNS),
            dropout=dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        family_embedding = self.family_embedding(batch["family_ids"])
        element_pieces = [family_embedding]

        if self.use_component_embedding and self.component_embedding is not None:
            element_pieces.append(self.component_embedding(batch["component_ids"]))

        property_input = torch.cat(
            [
                batch["property_values"],
                batch["property_mask"],
                batch["component_flags"],
            ],
            dim=-1,
        )
        property_latent = self.property_encoder(property_input)

        element_pieces.extend(
            [
                batch["mass_fraction"],
                property_latent,
                batch["component_flags"],
            ]
        )
        element_inputs = torch.cat(element_pieces, dim=-1)
        element_latent = self.element_encoder(element_inputs)

        component_mask = batch["component_mask"].unsqueeze(-1)
        masked_latent = element_latent * component_mask

        count = component_mask.sum(dim=1).clamp(min=1.0)
        pooled_mean = masked_latent.sum(dim=1) / count
        masked_for_max = element_latent.masked_fill(component_mask <= 0, -1e9)
        pooled_max = masked_for_max.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        catalyst_embedding = self.catalyst_embedding(batch["catalyst_ids"])
        condition_latent = self.condition_encoder(torch.cat([batch["conditions"], catalyst_embedding], dim=-1))

        fusion_inputs = [pooled_mean, pooled_max, condition_latent]
        if self.use_tabular_branch and self.tabular_encoder is not None and "tabular_features" in batch:
            fusion_inputs.append(self.tabular_encoder(batch["tabular_features"]))
        fused = torch.cat(fusion_inputs, dim=-1)
        return self.fusion_head(fused)
