# Final Improvement Journey

## Executive Summary

This project improved from a valid but conservative live result of `0.109256` to a locked best score of `0.104084`. The progress did not come from random leaderboard probing. It came from a sequence of increasingly better-aligned engineering decisions:

1. build a stable non-tree baseline
2. identify the real stable signals
3. align training and selection with the platform objective
4. add residual correction only where evidence justified it
5. stop when the best confirmed live result was already in hand

## Step 1. Initial Valid Submission: `0.109256`

Artifact:

- `outputs/submissions/neftekod_dot_submission_final.zip`

What changed:

- We established the first fully valid, competition-compliant delivery path.
- The pipeline respected grouped handling by `scenario_id`, produced a correct submission schema, and used the non-tree hybrid family-only path that was already validated in the repo.

Why it was done:

- Before optimization matters, the team needs a correct, reproducible, shippable baseline.
- This anchor gave us a trusted reference point for every later decision.

Outcome:

- Live platform score: `0.109256`

What we learned:

- The solution was viable and compliant, but there was room to improve metric alignment and robustness.
- The hardest errors were still concentrated in severe viscosity cases.

## Step 2. Objective Alignment And Stage 1.5: `0.107282`

Artifacts:

- `outputs/reports/objective_alignment_report.md`
- `outputs/reports/objective_alignment_fixed_metric_report.md`
- `outputs/submissions/neftekod_dot_submission_stage15_huber_fixedmetric.zip`

What changed:

- We stopped selecting checkpoints by the older combined score alone.
- We moved to a fixed platform-style metric using target MAE normalized by the competition scales.
- We kept the strong family-only hybrid core but aligned the selection criterion with the live objective.

Why it was done:

- The model was already learning useful structure, but we suspected our training-selection loop was not optimized for the actual platform outcome.
- This was the classic engineering move from a good model to a better objective.

Outcome:

- Live platform score improved to `0.107282`

What we learned:

- Metric alignment was real, not cosmetic.
- The system did not need a new model family yet; it needed a better definition of what counted as a good checkpoint.

## Step 3. GP/Meta Family Improvement: `0.104614`

Artifacts and lineage:

- `outputs/reports/gp_ensemble_report.md`
- `outputs/reports/meta_stack_search_report.md`
- `outputs/reports/paired_bootstrap_ci_stage15_vs_best_meta.md`
- packaged GP/meta lineage:
  - `outputs/submissions/neftekod_dot_submission_gp_stage2_stack_deep_plus_gp_matern_white.zip`
  - `outputs/submissions/neftekod_dot_submission_gp_stage2_meta_blend_visc_stage15+gp_matern_white+current_stack+stack_matern_white_dot_ox_current_stack+stack_matern_white_dot.zip`

What changed:

- We added GP-based residual modeling and meta blending on top of the Stage 1.5 anchor.
- The residual correction used the validated `conditions_structure_family` regime instead of a new uncontrolled feature expansion.

Why it was done:

- Stage 1.5 had improved the core model, but local evidence suggested that smooth residual structure remained in the scenario space.
- GP/meta stacking was a targeted way to capture that structure without abandoning the main chemistry-aware design.

Outcome:

- Competition history improved further to `0.104614`

What we learned:

- The Stage 1.5 model was not saturated.
- Residual correction added value, especially when grounded in the same stable feature subset that worked in baseline ablations.
- At the same time, this phase also taught us to separate strong local evidence from final shipping decisions.

Interpretation note:

- The repo identifies the GP/meta artifact lineage and its local evidence.
- The `0.104614` figure is the corresponding competition-history milestone used in the final presentation narrative.

## Step 4. Final Best Submission: `0.104084`

Locked artifact:

- `outputs/submissions/neftekod_dot_submission_meta_blend_v1.zip`
- corresponding lineage CSV:
  - `outputs/submissions/neftekod_dot_submission_gp_stage2_meta_blend_visc_stage15+gp_matern_white+current_stack+stack_matern_white_dot_ox_current_stack+stack_matern_white_dot/predictions.csv`

What changed:

- The winning live submission came from the current live meta-lineage winner, recorded as `submission #7`.
- The corrected packaged artifact is the short upload alias `neftekod_dot_submission_meta_blend_v1.zip`.
- That ZIP is byte-identical to the packaged GP/meta blend artifact produced in the meta-stack search phase.

Why it was done:

- The team kept testing disciplined challengers, but the live platform history showed that this submission became the best confirmed result.
- Once `0.104084` was achieved, the right decision was to lock it rather than continue to churn the shipping path.

Outcome:

- Best confirmed live platform score: `0.104084`

What we learned:

- The best final outcome came from the corrected GP/meta lineage rather than from the earlier Stage 1.5 ZIP.
- Platform trustworthiness matters more than local excitement.
- The correct final engineering behavior is to freeze the winning artifact and present it clearly.

## Step 5. Later Probes That Did Not Beat The Winner

### Huber-weight tuned challenger

- Artifact:
  - `outputs/submissions/neftekod_dot_submission_huber_weight_grid_vd100_od050_joint_light.zip`
- Why it was tried:
  - local tuning evidence looked strong in `outputs/reports/huber_weight_grid_report.md`
- Outcome:
  - later live result `0.104137`
- Lesson:
  - even a promising local improvement can lose narrowly on platform

### Final prediction-space blends

- Artifacts:
  - `outputs/submissions/neftekod_dot_submission_blend_best7_best9_80_20.zip`
  - `outputs/submissions/neftekod_dot_submission_blend_best7_best9_60_40.zip`
- Why they were tried:
  - they were conservative final probes built from the two best known live submissions
- Outcome:
  - no confirmed better live result than `0.104084`
- Lesson:
  - late blending is useful only if it clears the actual platform bar

### External-data sidecar

- Evidence:
  - `outputs/reports/external_augmented_report.md`
- Outcome:
  - did not beat the local anchor strongly enough to package
- Lesson:
  - scientific support is valuable, but shipping should remain disciplined

### Chemistry-gated late candidates

- Evidence:
  - `outputs/reports/late_stage_submission_candidates.md`
- Outcome:
  - remained exploratory, not the final shipping choice
- Lesson:
  - high-variance final probes are acceptable to test, but not to freeze without winning evidence

## Why This Journey Is Strong Engineering

This is a credible case study because every stage had a clear purpose.

- The baseline established correctness.
- The ablation work identified the stable signals.
- Stage 1.5 aligned the model with the true objective.
- GP/meta work explored residual correction without abandoning discipline.
- The final lock respected confirmed live evidence rather than chasing one more idea.

## Final Case-Study Message

The project did not win through uncontrolled experimentation. It improved through targeted learning:

- from `0.109256` to `0.107282` by fixing objective alignment
- from `0.107282` to `0.104614` by adding informed residual correction
- from `0.104614` to the final locked `0.104084` by keeping only what proved itself on the platform

That is the story we should present.
