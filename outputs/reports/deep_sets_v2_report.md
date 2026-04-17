# Hybrid Deep Sets v2 Report

## Hybrid Architecture
- Set branch: per-component Deep Sets encoder over family / optional component embeddings, mass fraction, property compression, masks, and coverage flags.
- Condition branch: explicit temperature, duration, biofuel, and catalyst embedding.
- Tabular branch: `77` scenario-level features from `conditions_structure_family`.
- Fusion: concatenate pooled set representation, condition representation, and tabular representation into a compact 2-target head.

## Tabular Features Used
- Feature groups: `['scenario_conditions', 'structure_and_mass', 'component_families']`
- Feature count: `77`

## Hybrid Comparison

```text
 rank_combined_score                           model_name target_strategy  combined_score__mean  target_delta_kinematic_viscosity_pct__rmse__mean  target_oxidation_eot_a_per_cm__rmse__mean  target_delta_kinematic_viscosity_pct__mae__mean  target_oxidation_eot_a_per_cm__mae__mean
                   1      hybrid_deep_sets_v2_family_only             raw              1.550652                                        122.251817                                  29.956870                                        54.181820                                 19.803472
                   2 hybrid_deep_sets_v2_family_component             raw              1.580078                                        128.521855                                  26.768027                                        62.195597                                 17.953033
                   3 hybrid_deep_sets_v2_family_component viscosity_asinh              1.711819                                        139.312044                                  28.479188                                        56.955032                                 17.533607
                   4      hybrid_deep_sets_v2_family_only viscosity_asinh              5.438440                                        494.615546                                  23.820599                                       120.454354                                 18.139275
```

## Best Hybrid Deep Sets v2
- Best configuration: `hybrid_deep_sets_v2_family_only` with target strategy `raw` and mean combined score `1.5507`
- Mean RMSEs: viscosity `122.2518`, oxidation `29.9569`

## Comparison Against References
- Best tabular baseline: `conditions_structure_family` with combined score `1.7163` and viscosity RMSE `135.8257`
- Deep Sets v1 reference: `deep_sets_v1` / `raw` with combined score `1.6813`
- Hybrid delta vs tabular: `-0.1656` (better)
- Hybrid delta vs Deep Sets v1: `-0.1306` (better)
- Hybrid viscosity RMSE delta vs Deep Sets v1: `-20.1861` (improved)

## Ablations Run
- family embedding only + raw
- family embedding only + viscosity_asinh
- family + component embedding + raw
- family + component embedding + viscosity_asinh

## Viscosity Transform Effect
- `hybrid_deep_sets_v2_family_component`: combined score delta `+0.1317`, viscosity RMSE delta `+10.7902`
- `hybrid_deep_sets_v2_family_only`: combined score delta `+3.8878`, viscosity RMSE delta `+372.3637`

## Hardest Out-of-Fold Scenarios
- `train_107`: combined normalized abs error `17.664`, viscosity true/pred `1763.34` / `192.00`, oxidation true/pred `95.47` / `75.62`
- `train_142`: combined normalized abs error `7.320`, viscosity true/pred `48.90` / `-393.87`, oxidation true/pred `93.48` / `408.83`
- `train_82`: combined normalized abs error `4.467`, viscosity true/pred `566.53` / `186.39`, oxidation true/pred `103.10` / `75.19`
- `train_109`: combined normalized abs error `4.065`, viscosity true/pred `178.68` / `571.18`, oxidation true/pred `92.78` / `111.34`
- `train_90`: combined normalized abs error `3.215`, viscosity true/pred `519.21` / `264.76`, oxidation true/pred `134.43` / `72.36`

## Risks
- The hybrid still trains on only 167 scenarios, so fold variance remains material.
- The tabular branch may memorize structure patterns that do not generalize if scenario coverage shifts.
- Component identity embeddings still risk overfitting rare additives and batches.
