# Final Presentation Content

## Slide 1. Problem, Goal, And Final Result

Title:

`Chemically Informed Non-Tree Modeling For Daimler Oxidation Test Prediction`

Key points:

- We had to predict viscosity increase and oxidation response for each lubricant scenario.
- Each scenario combined variable-length formulation data with shared test conditions.
- Our final locked live score is `0.104084`.

Suggested visual:

- One strong score card with the two targets, the non-tree constraint, and the final score.

## Slide 2. Data Structure And Challenge

Key points:

- The unit of prediction is the scenario, not the component row.
- Every scenario contains a variable number of ingredients plus one shared severity context.
- The dataset is small, heterogeneous, and dominated by hard viscosity tails.

Suggested visual:

- A simple diagram showing multiple component rows merging into one scenario-level prediction.

## Slide 3. Preprocessing And Join Policy

Key points:

- We used deterministic property joins only.
- We did not destabilize the pipeline with a late property-merge rewrite.
- Grouping stayed strictly by `scenario_id` in every important evaluation stage.

Suggested visual:

- Pipeline strip: raw mixture rows -> prepared scenario table -> grouped validation -> prediction.

## Slide 4. Why Grouped Validation Was Non-Negotiable

Key points:

- Near-duplicate compositions exist.
- Random splits would overestimate quality.
- Grouped validation protected us from false confidence and improved platform decision quality.

Suggested visual:

- Split-screen showing "wrong random split" versus "correct grouped split".

## Slide 5. Baseline And Ablation Findings

Key points:

- Best tabular baseline: `conditions_structure_family`
- Best baseline combined score: `1.7163`
- What helped:
  - scenario conditions
  - structure statistics
  - family-level aggregation
- What hurt:
  - the full weighted-property block

Suggested visual:

- One compact bar chart of feature-set comparison.

## Slide 6. Why We Moved To A Hybrid Family-Only Model

Key points:

- Lubricant mixtures are naturally unordered sets.
- DOT outcomes also depend on global severity conditions.
- Our hybrid model combined:
  - a Deep Sets composition branch
  - an explicit condition branch
  - a compact family-aware tabular branch
- `hybrid_deep_sets_v2_family_only / raw` beat the component-granular alternative.

Suggested visual:

- Clean architecture diagram with the three branches entering one two-target head.

## Slide 7. Stage 1.5 Metric Alignment Insight

Key points:

- We improved not by changing everything, but by aligning checkpoint selection with the platform objective.
- We used a fixed platform-style metric based on target MAE scales.
- Live progression:
  - initial valid submission: `0.109256`
  - Stage 1.5 objective-aligned result: `0.107282`

Suggested visual:

- Before/after selection-metric diagram and a score arrow from `0.109256` to `0.107282`.

## Slide 8. GP/Meta Insight

Key points:

- We then tested GP and meta-stack residual correction over the validated family-condition feature regime.
- Locally, the best meta candidate improved the platform-style proxy from `0.061385` to `0.057857`.
- This phase taught us that the hybrid model captured most of the chemistry, while GP/meta captured smooth residual structure.
- Competition history then improved further to `0.104614`.

Suggested visual:

- Residual-correction illustration: base prediction plus structured correction.

## Slide 9. Factor Analysis And Chemistry Interpretation

Key points:

- Top factors were:
  - temperature
  - duration
  - biofuel fraction
  - catalyst severity
  - additive/base-oil family balance
- This matches the literature on oxidation growth, additive depletion, and viscosity increase.
- Family-level representation worked better than exact component identity because it preserved mechanism and reduced noise.

Suggested visual:

- Ranked factor chart next to a chemistry mechanism table.

## Slide 10. Final Locked Result And Why We Stopped

Key points:

- Best confirmed live score: `0.104084`
- Final locked artifact:
  - `outputs/submissions/neftekod_dot_submission_meta_blend_v1.zip`
- Locked submission lineage:
  - `submission #7`, current live meta-lineage winner
- Later candidates were tested, including Huber tuning and final blends, but none beat the confirmed winner.
- We stopped because the correct professional decision was to freeze the best verified platform submission.

Suggested visual:

- Submission ladder ending at highlighted `#7 = 0.104084`.

## Slide 11. What We Tested And Rejected

Key points:

- Wide weighted-property expansion added noise
- Granular component identity overfit
- External-data sidecar did not justify shipping
- Late blends were reasonable probes but did not produce a better confirmed score

Suggested visual:

- Simple "tested / learned / rejected" table.

## Slide 12. Why This Solution Deserves Recognition

Key points:

- It is non-tree compliant and technically disciplined.
- It is chemically informed, not purely opportunistic.
- It is reproducible, validated, and presentation-ready.
- The improvement path from `0.109256` to `0.104084` is methodical and honest.

Suggested visual:

- Four final badges:
  - disciplined validation
  - chemistry-backed interpretation
  - reproducible engineering
  - strong final live result
