# Baseline Ablation Report

## Starting Point
- Current best baseline configuration from `baseline_cv_results.csv`: `pls_regression` / `raw`.
- Baseline combined score: `1.8768`.

## Feature-Group Comparison

```text
 rank_combined_score                      feature_setting  feature_count  combined_score__mean  target_delta_kinematic_viscosity_pct__rmse__mean  target_oxidation_eot_a_per_cm__rmse__mean
                   1          conditions_structure_family             77              1.716252                                        135.825674                                  34.222484
                   2 conditions_structure_family_coverage            102              1.735800                                        136.835737                                  35.369276
                   3                 conditions_structure             32              1.763088                                        139.716387                                  35.229315
                   4                      conditions_only              4              1.779233                                        142.147191                                  34.042794
                   5                     full_feature_set            410              1.876787                                        150.943157                                  34.548942
```

## Decisions
- Best feature-group configuration: `conditions_structure_family` with combined score `1.7163`.
- Improvement versus `conditions_only`: `+0.0630`.
- `conditions_structure` improved the combined score by `0.0161` versus the previous step.
- `conditions_structure_family` improved the combined score by `0.0468` versus the previous step.
- `conditions_structure_family_coverage` worsened the combined score by `0.0195` versus the previous step.
- `full_feature_set` worsened the combined score by `0.1410` versus the previous step.
- Weighted-property verdict: the full block is `not strong enough to justify all 308 weighted-property features unchanged` (delta vs previous step `+0.1410` with `410` features total).

## Worst Out-of-Fold Scenarios
- `train_107`: OOF combined normalized abs error `14.847`, viscosity true/pred `1763.34` / `435.40`, oxidation true/pred `95.47` / `88.38`.
  Drivers: extreme viscosity target at the 100.0th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 2 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 37.7% (43th pct), Противоизносная_присадка 18.3% (90th pct), Дисперсант 14.4% (85th pct).
- `train_106`: OOF combined normalized abs error `5.509`, viscosity true/pred `861.54` / `321.87`, oxidation true/pred `84.25` / `77.79`.
  Drivers: extreme viscosity target at the 99.4th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 2 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 37.7% (43th pct), Противоизносная_присадка 18.3% (90th pct), Дисперсант 14.4% (85th pct).
- `train_90`: OOF combined normalized abs error `3.568`, viscosity true/pred `519.21` / `233.35`, oxidation true/pred `134.43` / `69.93`.
  Drivers: rare condition tuple seen in only 2 scenario(s); extreme viscosity target at the 96.4th percentile; elevated oxidation target at the 99.4th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 1 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 43.9% (72th pct), Соединение_молибдена 13.6% (91th pct), Противоизносная_присадка 13.6% (64th pct).
- `train_110`: OOF combined normalized abs error `3.567`, viscosity true/pred `775.51` / `436.53`, oxidation true/pred `107.79` / `90.89`.
  Drivers: extreme viscosity target at the 98.8th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 2 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 39.4% (55th pct), Дисперсант 17.8% (93th pct), Противоизносная_присадка 16.4% (78th pct).
- `train_118`: OOF combined normalized abs error `3.447`, viscosity true/pred `754.93` / `438.90`, oxidation true/pred `103.98` / `92.45`.
  Drivers: extreme viscosity target at the 98.2th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 2 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 37.7% (43th pct), Противоизносная_присадка 18.3% (90th pct), Дисперсант 14.4% (85th pct).

## Focus Scenarios: train_106 and train_107
- `train_106`: OOF combined normalized abs error `5.509`, viscosity true/pred `861.54` / `321.87`, oxidation true/pred `84.25` / `77.79`.
  Drivers: extreme viscosity target at the 99.4th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 2 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 37.7% (43th pct), Противоизносная_присадка 18.3% (90th pct), Дисперсант 14.4% (85th pct).
- `train_107`: OOF combined normalized abs error `14.847`, viscosity true/pred `1763.34` / `435.40`, oxidation true/pred `95.47` / `88.38`.
  Drivers: extreme viscosity target at the 100.0th percentile; below-median usable numeric property coverage; above-median missing-property burden; has 2 near-duplicate composition peer(s), so small condition changes may drive large target swings
  Composition: Базовое_масло 37.7% (43th pct), Противоизносная_присадка 18.3% (90th pct), Дисперсант 14.4% (85th pct).

## Deep Sets v1 Implications
- Keep scenario conditions as explicit inputs; the hardest cases suggest strong condition-composition interactions.
- Feed per-component mass fractions and component identity/family embeddings directly rather than relying only on wide aggregated family shares.
- Include per-component numeric property vectors plus explicit coverage flags (`used_exact_batch_props`, `used_typical_fallback`, `missing_all_props`).
- Use a compressed or masked property channel in Deep Sets v1; the full weighted-property block added width faster than it added accuracy.
