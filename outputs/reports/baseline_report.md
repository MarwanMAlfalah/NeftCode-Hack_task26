# Baseline CV Report

## Data Snapshot
- Train scenarios: `167`
- Feature columns evaluated: `410`
- Feature groups: `{'scenario_conditions': 4, 'structure_and_mass': 28, 'component_families': 45, 'coverage_and_missingness': 25, 'weighted_numeric_properties': 308}`

## Baseline Comparison

```text
 rank_combined_score           model_name target_strategy  combined_score__mean  target_delta_kinematic_viscosity_pct__rmse__mean  target_oxidation_eot_a_per_cm__rmse__mean  target_delta_kinematic_viscosity_pct__mae__mean  target_oxidation_eot_a_per_cm__mae__mean
                   1       pls_regression             raw              1.876787                                        150.943157                                  34.548942                                        82.249039                                 27.764807
                   2 multitask_elasticnet viscosity_asinh              1.894828                                        161.792799                                  22.551884                                        61.145240                                 17.433556
                   3    ridge_multioutput             raw              3.367977                                        264.681961                                  68.581749                                       105.704866                                 28.273164
                   4 multitask_elasticnet             raw              4.158940                                        369.862431                                  29.916670                                       125.864000                                 18.995348
                   5       pls_regression viscosity_asinh              5.795977                                        516.603959                                  39.130695                                       137.422348                                 22.221507
                   6    ridge_multioutput viscosity_asinh              6.655697                                        517.504087                                 142.087083                                       135.448439                                 39.891768
```

## Best Current Baseline
- Best configuration: `pls_regression` with target strategy `raw` and mean combined score `1.8768`
- Mean RMSEs: viscosity `150.9432`, oxidation `34.5489`

## Viscosity Transform Effect
- `multitask_elasticnet`: combined score delta `-2.2641`, viscosity RMSE delta `-208.0696`
- `pls_regression`: combined score delta `+3.9192`, viscosity RMSE delta `+365.6608`
- `ridge_multioutput`: combined score delta `+3.2877`, viscosity RMSE delta `+252.8221`

## Error Analysis
- The viscosity target remains the main difficulty because it is highly skewed and outlier-heavy.
- Large gaps between exact-key coverage and usable numeric coverage indicate that fallback and missingness still matter materially.
- The hardest scenarios are dominated by large viscosity misses rather than oxidation misses.

Hardest out-of-fold scenarios for the best baseline:
- `train_107`: combined normalized abs error `15.733`, viscosity true/pred `1763.340` / `356.839`, oxidation true/pred `95.470` / `87.017`
- `train_142`: combined normalized abs error `6.301`, viscosity true/pred `48.900` / `-457.582`, oxidation true/pred `93.480` / `205.289`
- `train_106`: combined normalized abs error `5.536`, viscosity true/pred `861.540` / `321.718`, oxidation true/pred `84.250` / `74.755`
- `train_110`: combined normalized abs error `4.955`, viscosity true/pred `775.510` / `311.824`, oxidation true/pred `107.790` / `75.557`
- `train_118`: combined normalized abs error `4.755`, viscosity true/pred `754.930` / `328.826`, oxidation true/pred `103.980` / `75.489`

Highest fold-to-fold volatility:
- `ridge_multioutput` / `viscosity_asinh`: combined score std `10.6011`, viscosity RMSE std `792.9721`, oxidation RMSE std `262.6430`
- `pls_regression` / `viscosity_asinh`: combined score std `8.6533`, viscosity RMSE std `789.6810`, oxidation RMSE std `32.5160`
- `multitask_elasticnet` / `raw`: combined score std `5.4457`, viscosity RMSE std `497.7893`, oxidation RMSE std `18.1818`
