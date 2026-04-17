# Final Presentation Outline

## Slide 1. Challenge And Goal

- Predict two DOT outputs for each lubricant scenario
- Deliver a non-tree, competition-compliant, reproducible solution
- Final locked best live score: `0.104084`

## Slide 2. Data Structure And Why It Is Hard

- Each scenario is a variable-length mixture under one shared test condition tuple
- Small dataset, strong heterogeneity, severe viscosity tails
- Need one prediction per `scenario_id`

## Slide 3. Preprocessing And Join Policy

- Deterministic property joins only
- No broad rewrite of the property merge policy late in the competition
- Strict grouped handling by `scenario_id`

## Slide 4. Why Grouped Validation Mattered

- Random leakage would overstate quality
- Near-duplicate scenarios make grouped evaluation essential
- Local decisions only counted if they held under grouped validation

## Slide 5. Baseline And Ablation Findings

- Best stable tabular subset: `conditions_structure_family`
- Conditions, structure, and family aggregates helped
- Wide weighted-property expansion hurt

## Slide 6. Stage 1.5 Metric Alignment Insight

- Moved checkpoint selection to a fixed platform-style metric
- Same core model, better alignment to the real objective
- Live improvement: `0.109256 -> 0.107282`

## Slide 7. GP/Meta Insight

- Residual correction over the validated family-condition regime
- Strong local improvement signal
- Useful insight and part of the final locked meta-lineage winner

## Slide 8. Factor Analysis And Chemistry Story

- Top factors: temperature, duration, biofuel, catalyst severity, family balance
- Family-level chemistry generalized better than exact component identity
- Literature and factor analysis told the same story

## Slide 9. Full Improvement Journey

- `0.109256 -> 0.107282 -> 0.104614 -> 0.104084`
- Show what changed, why it changed, and what survived to the final lock

## Slide 10. What We Tested And Rejected

- Overly wide weighted properties
- Overly granular component identity
- External-data sidecar for shipping
- Late blends that did not beat the winner

## Slide 11. Why The Final Solution Is Trustworthy

- Best confirmed live artifact is frozen
- Locked artifact: `outputs/submissions/neftekod_dot_submission_meta_blend_v1.zip`
- Lineage: `submission #7`, current live meta-lineage winner
- Submission package is validated
- Method is reproducible, interpretable, and competition-ready

## Slide 12. Conclusion And Impact

- Strong final score with disciplined methodology
- Chemically informed and non-tree compliant
- Clear evidence of engineering rigor, not leaderboard luck
