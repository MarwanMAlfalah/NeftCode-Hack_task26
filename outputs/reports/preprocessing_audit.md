# Preprocessing Audit

## Clean Property Table Validation
- `component_properties_clean.csv` rows: 2557
- Blank `property_name` rows: 0
- Blank `property_value` rows: 0
- Blank/invalid property rows excluded successfully: yes

## Coverage Summary

| split | rows | scenarios | exact property rows | usable numeric rows | typical fallback rows | missing_all_props rows | missing batch_id rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2230 | 167 | 1593 (71.4%) | 2030 (91.0%) | 1868 (83.8%) | 200 (9.0%) | 2 (0.1%) |
| test | 524 | 40 | 370 (70.6%) | 482 (92.0%) | 447 (85.3%) | 42 (8.0%) | 0 (0.0%) |

## Notes
- `used_exact_batch_props` tracks an exact component/batch lookup row, not guaranteed full numeric coverage.
- `has_usable_property_coverage` is the stronger numeric-coverage flag after exact-plus-typical coalescing.

## Property Names With Multiple Units
- `Индекс вязкости, ГОСТ 25371`: `-`, `–`
- `Испаряемость по NOACK, ASTM D5800`: `%`, `% масс`
- `Плотность при 15°С, ASTM D4052`: `г/см³`, `кг/м³`
