# Final Submission Checklist

Use this checklist immediately before creating the upload ZIP.

## Required Artifacts
- `outputs/predictions.csv` exists locally.
- `inference.ipynb` exists at repository root.
- The final ZIP contains `predictions.csv` at ZIP root.
- The final ZIP contains `inference.ipynb` at ZIP root.

## CSV Integrity
- `predictions.csv` is UTF-8 encoded.
- `predictions.csv` contains exactly these columns, in order:
  - `scenario_id`
  - `Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %`
  - `Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm`
- `predictions.csv` has exactly one row per test `scenario_id`.
- `predictions.csv` has no duplicate test IDs.
- `predictions.csv` has no extra test IDs.
- `predictions.csv` has no missing values.

## ZIP Hygiene
- Exactly one CSV is included in the upload ZIP.
- No extra CSV files are present anywhere else in the ZIP.
- Required files are placed at ZIP root, not in nested folders.
- ZIP filename is ASCII-only.
- All included filenames are ASCII-only and contain no Cyrillic characters.

## Local Validation Steps
1. Regenerate the official predictions file if needed:

```bash
jupyter nbconvert --to notebook --execute --inplace inference.ipynb
```

2. Run the packaging helper:

```bash
python3 scripts/package_submission.py
```

3. Re-run ZIP validation only:

```bash
python3 scripts/package_submission.py --validate-only
```

4. Optional clean-run check:

```bash
python3 scripts/clean_run_check.py
```

## Final Manual Review
- Open the ZIP and confirm only `predictions.csv` and `inference.ipynb` are present at root.
- Confirm the official shipping path remains unchanged:
  - `hybrid_deep_sets_v2_family_only / raw / current loss`
- Confirm you are uploading the ASCII-safe final ZIP from `outputs/submissions/`.
