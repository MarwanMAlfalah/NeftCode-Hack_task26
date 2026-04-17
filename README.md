# Neftekod DOT Challenge

This repository contains a reproducible solution pipeline for the Daimler Oxidation Test (DOT) scenario-level multi-output regression task. The target is to predict two scenario-level outcomes from variable-length component mixtures and scenario conditions:

- `Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %`
- `Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm`

## Task Summary

Each train scenario contains a variable number of component rows. Every scenario shares one condition tuple and two regression targets. The modeling pipeline must respect grouped evaluation by `scenario_id` and remain non-tree-based.

## Approach Summary

The current production path is a hybrid Deep Sets model:

- Component-level set encoder over prepared mixture rows.
- Explicit scenario-condition branch for temperature, duration, biofuel, and catalyst category.
- Compact tabular side branch using the strongest engineered feature subset from the best baseline feature ablation: `conditions_structure_family`.
- Fusion head producing both regression targets jointly.
- Submission inference uses the winning variant with a 5-seed ensemble for launch safety.

## Current Best Model

Current best validated model:

- Variant: `hybrid_deep_sets_v2_family_only`
- Target strategy: `raw`
- Mean grouped-CV combined score: `1.5507`
- Mean grouped-CV viscosity RMSE: `122.2518`
- Mean grouped-CV oxidation RMSE: `29.9569`

Reference points:

- Best tabular baseline `conditions_structure_family`: combined score `1.7163`
- Deep Sets v1 best: combined score `1.6813`

## Repository Layout

- `src/data/`: raw loading, target preparation, deterministic property joins
- `src/features/`: scenario-level feature generation
- `src/models/`: Deep Sets / hybrid model code and training helpers
- `src/eval/`: grouped-CV runners and reports
- `tests/`: unit tests for joins, targets, features, and tensor/model shapes
- `inference.ipynb`: root-level end-to-end submission notebook
- `outputs/`: generated reports, CV outputs, and submission predictions

## Quickstart

### Local Python

1. Create and activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Execute the inference notebook end to end:

```bash
jupyter nbconvert --to notebook --execute --inplace inference.ipynb
```

4. The submission file will be written to:

```text
outputs/predictions.csv
```

### Docker

Build and run the notebook in a container:

```bash
docker build -t neftekod-dot .
docker run --rm -v "$(pwd)/outputs:/app/outputs" neftekod-dot
```

### Docker Compose

Run the same flow with compose:

```bash
docker compose up --build
```

The bind-mounted submission file will appear in:

```text
./outputs/predictions.csv
```

## Reproducibility Notes

- Python runtime is pinned to `3.11` in Docker.
- Core libraries are pinned in `requirements.txt`.
- Data preparation and feature generation are deterministic.
- The hybrid model uses deterministic seeding across Python, NumPy, PyTorch, and DataLoader shuffling.
- Submission inference uses a fixed 5-seed ensemble: `0, 1, 2, 3, 4`.
- The notebook validates the final CSV schema, row count, uniqueness, and null safety before writing.

## Method-Restriction Compliance

- Final predictor remains non-tree-based.
- Grouped handling is by `scenario_id`.
- The winning submission path reuses the validated hybrid Deep Sets v2 family-only variant.
- This delivery path does not alter the property join policy or introduce a new model family.

## How `predictions.csv` Is Generated

`inference.ipynb` performs the following steps:

1. Runs the deterministic data preparation pipeline.
2. Runs the scenario-level feature pipeline.
3. Loads the prepared hybrid Deep Sets data.
4. Trains the winning `hybrid_deep_sets_v2_family_only / raw` model as a 5-seed ensemble.
5. Predicts one row per test `scenario_id`.
6. Validates the output schema and writes `outputs/predictions.csv` in UTF-8.
