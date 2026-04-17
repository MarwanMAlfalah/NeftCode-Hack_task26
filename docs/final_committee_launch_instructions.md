# Final Committee Launch Instructions

## Purpose

This note gives the shortest reproducible path for committee review.

The frozen winning platform artifact is:

- `outputs/submissions/neftekod_dot_submission_gp_stage2_meta_runnerup_current_stack_only_oxidation.zip`

That ZIP already contains:

- `predictions.csv`
- `inference.ipynb`

## Local Python Reproduction

Use Python `3.11`.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Execute the inference notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace inference.ipynb
```

3. The regenerated prediction file will appear at:

```text
outputs/predictions.csv
```

## Docker Reproduction

```bash
docker build -t neftekod-dot .
docker run --rm -v "$(pwd)/outputs:/app/outputs" neftekod-dot
```

The output file will appear at:

```text
outputs/predictions.csv
```

## Docker Compose Reproduction

```bash
docker compose up --build
```

The output file will appear at:

```text
outputs/predictions.csv
```

## Frozen Artifact Validation

To inspect the final locked competition ZIP without rebuilding it:

```bash
unzip -l outputs/submissions/neftekod_dot_submission_gp_stage2_meta_runnerup_current_stack_only_oxidation.zip
```

The expected root contents are:

- `predictions.csv`
- `inference.ipynb`

## Freeze Rule

Do not replace the locked winning artifact unless a better live platform score is already confirmed in writing.

