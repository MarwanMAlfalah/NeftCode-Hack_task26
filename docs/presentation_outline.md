# Presentation Outline

## Slide 1. Title and Task
- Challenge: predict two DOT outputs for each lubricant scenario.
- Targets:
  - `Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %`
  - `Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm`
- Constraint: final predictor must stay non-tree-based and grouped by `scenario_id`.

## Slide 2. Data and Join Policy
- Each scenario contains a variable number of component rows plus one shared condition tuple.
- Official join policy: deterministic property join only; no broad rewrite of the property merge logic.
- Modeling split discipline: grouped CV strictly by `scenario_id`.
- Reviewer note: one prediction row is produced per test scenario.

## Slide 3. Feature Ablation Story
- Strongest tabular baseline: `conditions_structure_family`.
- Best baseline combined score: `1.7163`.
- Main ablation conclusion:
  - conditions help
  - structure stats help
  - family-level aggregation helps
  - wide weighted-property aggregation adds noise faster than signal

## Slide 4. Why We Moved to Hybrid Deep Sets
- Mixtures are variable-length unordered sets, so permutation-invariant set encoding is a natural fit.
- DOT results also depend on shared conditions that should be modeled explicitly.
- Best design choice: combine component-level set encoding with compact scenario-level engineered signals.

## Slide 5. Final Architecture
- Official shipping path:
  - `hybrid_deep_sets_v2_family_only / raw / current loss`
- Three branches:
  - set branch over component rows
  - explicit condition branch
  - compact tabular branch using `conditions_structure_family`
- Fusion head predicts both targets jointly.
- Family-only encoding beat family-plus-component encoding on grouped CV.

## Slide 6. Metrics and Benchmark Position
- Best tabular baseline: combined score `1.7163`
- Deep Sets v1: combined score `1.6813`
- Hybrid Deep Sets v2 family-only raw: combined score `1.5507`
- Gain vs best tabular baseline: `-0.1656` combined-score improvement
- Main upside: much better oxidation fit without giving up non-tree compliance

## Slide 7. Hardest Cases and Robustness
- Most instability comes from severe viscosity outliers.
- Final-selection reruns showed seed variance is still material.
- Hardest folds are driven by a small number of catastrophic viscosity misses.
- Delivery decision:
  - keep the hardened submission path unchanged
  - treat more aggressive training variants as research-only candidates

## Slide 8. Literature and Factor Analysis Takeaways
- Literature says DOT-like behavior should be dominated by temperature, time, biofuel contamination, catalyst chemistry, and additive-family balance.
- Factor analysis on the best tabular baseline confirmed:
  - top signals were temperature, duration, and biofuel fraction
  - family-level antiwear, molybdenum, detergent, dispersant, and base-oil features also ranked highly
- Interpretation:
  - condition severity matters strongly
  - family-level chemistry generalizes better than sparse component identity

## Slide 9. Full English Summary
- We framed the task as scenario-level prediction from variable-length lubricant mixtures under fixed test conditions.
- A pure tabular baseline already showed that condition variables, structure statistics, and family aggregations were the strongest stable signals.
- We then built a hybrid Deep Sets model that encodes component rows as a set, adds an explicit condition branch, and fuses a compact tabular branch with the best feature subset.
- The best validated model was `hybrid_deep_sets_v2_family_only / raw`, which improved the combined score from `1.7163` for the best tabular baseline to `1.5507`.
- Literature and factor analysis both support the same interpretation: temperature, duration, biofuel loading, catalyst chemistry, and additive-family balance are the main drivers of DOT outcomes.
- For delivery, we froze the hardened inference path and did not replace it with a riskier research variant.

## Slide 10. Final Conclusion
- Official competition path stays fixed on the hardened hybrid family-only model.
- The method is chemically plausible, competition-compliant, and reproducible.
- Main remaining weakness is robustness on the most extreme viscosity scenarios.
- Next work after submission should focus on stability and outlier handling, not on changing the shipping path right before delivery.
