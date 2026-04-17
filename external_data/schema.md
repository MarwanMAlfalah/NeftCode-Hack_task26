# External Data Schema

Use this template for any future external-data candidate.

| Field | Description |
| --- | --- |
| `dataset_name` | Human-readable dataset or source name |
| `source_url` | Canonical public URL or internal storage location |
| `retrieval_date` | Date the file or source was retrieved |
| `license_or_terms` | License, usage terms, or review note |
| `entity_level` | Expected entity granularity: component, family, scenario, or standard |
| `key_columns` | Join keys that would be required |
| `value_columns` | Main values contributed by the dataset |
| `coverage_notes` | Known missingness, sparsity, language issues, or mapping ambiguity |
| `expected_benefit` | Why the dataset may help |
| `integration_risk` | Leakage risk, noisy join risk, or policy mismatch |
| `status` | `candidate`, `reviewing`, `approved`, or `rejected` |

## Review Checklist
- Does the source have a clear license or acceptable usage terms?
- Can the data be joined without weakening the current deterministic join policy?
- Does it add information beyond the existing family and property tables?
- Is the expected benefit large enough to justify new leakage or mapping risk?
