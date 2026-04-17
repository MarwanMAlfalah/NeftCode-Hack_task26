# Final Submission Readiness

## Locked Artifact

- Final official ZIP: `outputs/submissions/neftekod_dot_submission_meta_blend_v1.zip`
- Final official CSV lineage file: `outputs/submissions/neftekod_dot_submission_gp_stage2_meta_blend_visc_stage15+gp_matern_white+current_stack+stack_matern_white_dot_ox_current_stack+stack_matern_white_dot/predictions.csv`
- Submission lineage: `submission #7`, current live meta-lineage winner
- Official best score: `0.104084`
- Shipping status: `frozen`

## Checklist

- Exact final ZIP path confirmed: `yes`
- Exactly one CSV in the ZIP: `yes`
- ZIP root contains `predictions.csv` and `inference.ipynb`: `yes`
- Exact CSV schema confirmed: `yes`
- Schema order confirmed:
  - `scenario_id`
  - `Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %`
  - `Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm`
- No nulls in required columns: `yes`
- No duplicate IDs: `yes`
- Exactly 40 prediction rows: `yes`
- Notebook present: `yes`
- ASCII filenames: `yes`
- CSV is UTF-8 compatible text: `yes`
- Official best score recorded as `0.104084`: `yes`

## Final Rule

Do not replace this artifact with:

- any other GP/meta ZIP
- any Huber-weight challenger
- any blend of `#7` and `#9`
- any external-data sidecar

Use the locked ZIP above for the official final package and presentation references.
