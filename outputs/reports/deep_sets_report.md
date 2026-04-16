# Deep Sets CV Report

## Data Snapshot
- Train scenarios: `167`
- Max components per scenario: `20`
- Property columns per component: `77`
- Learned vocab sizes: families `11`, components `114`, catalysts `3`

## Deep Sets Comparison

```text
 rank_combined_score   model_name target_strategy  combined_score__mean  target_delta_kinematic_viscosity_pct__rmse__mean  target_oxidation_eot_a_per_cm__rmse__mean  target_delta_kinematic_viscosity_pct__mae__mean  target_oxidation_eot_a_per_cm__mae__mean
                   1 deep_sets_v1             raw              1.681277                                        142.437946                                  21.329623                                        65.982001                                 16.301573
                   2 deep_sets_v1 viscosity_asinh              5.790949                                        519.147109                                  35.613165                                       128.339837                                 19.224977
```

## Best Deep Sets v1
- Best configuration: `deep_sets_v1` with target strategy `raw` and mean combined score `1.6813`
- Mean RMSEs: viscosity `142.4379`, oxidation `21.3296`

## Comparison To Current Best Tabular Baseline
- Tabular reference: `conditions_structure_family` with mean combined score `1.7163`
- Deep Sets delta vs tabular: `-0.0350` (better)

## Input Design
- One set element per component row with family embedding, component embedding, standardized mass fraction, compressed property vector, property mask, and coverage/source flags.
- Scenario condition branch uses temperature, time, biofuel mass fraction, and catalyst category embedding.
- Pooling is permutation-invariant mean plus max before fusion into a 2-target regression head.

## Viscosity Transform Effect
- `deep_sets_v1`: combined score delta `+4.1097`, viscosity RMSE delta `+376.7092`

## Hardest Out-of-Fold Scenarios
- `train_107`: combined normalized abs error `14.773`, viscosity true/pred `1763.34` / `437.15`, oxidation true/pred `95.47` / `94.86`
- `train_142`: combined normalized abs error `8.555`, viscosity true/pred `48.90` / `-719.06`, oxidation true/pred `93.48` / `43.98`
- `train_86`: combined normalized abs error `4.661`, viscosity true/pred `225.24` / `636.70`, oxidation true/pred `61.15` / `97.01`
- `train_165`: combined normalized abs error `4.240`, viscosity true/pred `141.40` / `524.64`, oxidation true/pred `50.69` / `95.62`
- `train_110`: combined normalized abs error `4.231`, viscosity true/pred `775.51` / `369.64`, oxidation true/pred `107.79` / `92.37`

## Risks
- The dataset is small for a neural model, so fold variance remains meaningful even with the compact architecture.
- Component identity embeddings can overfit rare components; the family path is likely the more stable generalization route.
- Missing numeric properties are handled by masks and flags, but the model still sees zero-filled values behind those masks.

## Fold Volatility
- `deep_sets_v1` / `viscosity_asinh`: combined score std `8.5683`, viscosity RMSE std `783.5359`, oxidation RMSE std `30.0026`
- `deep_sets_v1` / `raw`: combined score std `0.7864`, viscosity RMSE std `66.3055`, oxidation RMSE std `3.2716`
