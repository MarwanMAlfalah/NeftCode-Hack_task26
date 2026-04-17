# External Source Catalog

This catalog is the manual-extraction starting queue for the external-data sprint.

## Priority Order

### 1. `nrel_b100_stability_interim_2006`
- Source type: `technical_report`
- URL: https://research-hub.nrel.gov/en/publications/stability-of-biodiesel-and-biodiesel-blends-interim-report
- Why first: strong public provenance, high relevance to oxidation/stability, and likely to contain directly extractable accelerated-test records across multiple B100 samples.
- First fields to extract: `source_id`, `source_url`, `test_family`, `temperature_c`, `duration_h`, `biofuel_pct`, `target_visc_rel_pct` when reported, `target_ox_proxy`, `notes`.

### 2. `nrel_biodiesel_stability_database_2007`
- Source type: `technical_report`
- URL: https://research-hub.nrel.gov/en/publications/empirical-study-of-the-stability-of-biodiesel-and-biodiesel-blend-2
- Why second: broad sample coverage and report-style structure make it a good backbone dataset for stable external scenarios.
- First fields to extract: scenario conditions, blend level, storage/aging setup, oxidation-stability endpoints, and any deposit/viscosity growth summaries.

### 3. `nrel_storage_stability_journal`
- Source type: `peer_reviewed_article`
- URL: https://research-hub.nrel.gov/en/publications/storage-stability-of-biodiesel-and-biodiesel-blends-2
- Why third: high-quality experimental storage results with oxidation-focused endpoints and blend-specific stability behavior.
- First fields to extract: feedstock/blend description, storage protocol, induction-time or insolubles endpoint, and any paired viscosity change values.

### 4. `nrel_long_term_storage_christensen`
- Source type: `peer_reviewed_article`
- URL: https://research-hub.nrel.gov/en/publications/long-term-storage-stability-of-biodiesel-and-biodiesel-blends-2/
- Why fourth: useful for longer-horizon degradation patterns and post-additization behavior, especially for conservative proxy rows.
- First fields to extract: aging temperature, effective duration, biodiesel fraction, antioxidant treatment, oxidation endpoint, and whether viscosity or acid-growth behavior was measured.

### 5. `mdpi_pyrogallol_derivative_2019`
- Source type: `peer_reviewed_article`
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6651424/
- Why fifth: open-access experimental paper with antioxidant composition context plus explicit viscosity and oxidation-stability discussion.
- First fields to extract: antioxidant package category, dosage, storage interval, viscosity change, oxidation stability endpoint, and any missing-condition notes.

### 6. `mdpi_high_temperature_polymerization_2018`
- Source type: `peer_reviewed_article`
- URL: https://www.mdpi.com/1996-1073/11/12/3514
- Why sixth: strong relevance for oxidation/polymerization-driven viscosity growth under accelerated conditions.
- First fields to extract: temperature, exposure time, biodiesel family, oxidation/polymerization outcome, and viscosity-growth measurements.

## Source-Type Reliability Guidance

Use these starting scores when filling `source_reliability_score`:

| source_type | starting score |
| --- | ---: |
| `peer_reviewed_article` | 0.95 |
| `technical_report` | 0.90 |
| `standard_summary` | 0.85 |
| `conference_paper` | 0.80 |
| `review_article` | 0.70 |
| `datasheet` | 0.60 |
| `other` | 0.50 |

## Manual Extraction Rules

1. Prefer primary experimental tables, appendices, and supplementary material over narrative review text.
2. Keep one row per component category within each `external_scenario_id`.
3. Repeat scenario-level targets on each component row for that scenario.
4. Use `target_ox_proxy` only when no trustworthy `target_ox_acm` value is available.
5. Favor records that are conditionally close to the repo’s regime: similar temperature, duration, biodiesel fraction, and catalyst setting.
6. Do not merge external records into the official shipping path or the core train tables directly.
