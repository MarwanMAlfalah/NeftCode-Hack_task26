# Literature Review

## Objective
This note summarizes literature that is most relevant to the Daimler Oxidation Test (DOT) prediction task and ties it to the shipped modeling strategy:

- official shipping path: `hybrid_deep_sets_v2_family_only / raw / current loss`
- task type: scenario-level multi-output regression from variable-length lubricant mixtures plus operating conditions
- practical question: which factors most plausibly drive viscosity increase and oxidation current in DOT-style stress tests?

## Short Takeaways
- Oxidation severity is strongly amplified by temperature, exposure duration, oxygen availability, biofuel contamination, and catalytic metal chemistry.
- Lubricant families matter because antioxidant, antiwear, detergent, dispersant, and base-oil classes operate through known mechanisms, while exact commercial component identity is often sparse, proprietary, and noisy.
- Oxidation-linked viscosity failures are nonlinear because oxidation products can both thin and thicken the fluid, with oligomerization and deposit formation often pushing viscosity upward in the hardest cases.
- A hybrid architecture is justified because scenario outcome depends on both set-structured composition and global test conditions.

## Structured Hypotheses

| Factor | Mechanism of influence on DOT results | How this factor is incorporated into our model |
| --- | --- | --- |
| Temperature | Higher temperature accelerates free-radical initiation, propagation, deposit formation, and oxidation-product growth, which can sharply increase both oxidation response and viscosity drift. | Explicit condition feature and condition branch input (`test_temperature_c`). |
| Time / exposure duration | Longer exposure increases cumulative oxidation, additive depletion, and secondary condensation/polymerization reactions. | Explicit condition feature and condition branch input (`test_duration_h`). |
| Biofuel fraction | FAME-rich contamination changes oxidation chemistry, interacts with additive packages, and can accelerate oil degradation or change the effectiveness of antioxidants and antiwear chemistry. | Explicit condition feature and condition branch input (`biofuel_mass_fraction_pct`). |
| Catalyst category / metal chemistry | Catalytic metals and catalyst-related chemistry can speed oxidation pathways and shift deposit formation behavior. | Explicit condition feature and condition branch input (`catalyst_dosage_category`). |
| Base-oil family | Base-stock saturation, aromatic content, and hydrocarbon structure affect intrinsic thermo-oxidative stability and viscosity-temperature behavior. | Family embeddings in the set branch plus family-level tabular aggregation features. |
| Antioxidant / antiwear families | ZDDP-like sulfur-phosphorus systems, phenolics, and amines interrupt oxidation chains or decompose peroxides, but can also face durability and emissions constraints. | Family embeddings and family mass-share / count features; family-only representation intentionally preserves mechanism class without depending on sparse component IDs. |
| Detergent / dispersant families | These families influence acid neutralization, contaminant handling, sludge control, and suspension of oxidation by-products. | Family embeddings and family aggregation features in the hybrid tabular branch. |
| Viscosity modifiers / depressants | These alter rheology directly and can interact with oxidation-driven thickening or shear-driven thinning. | Family embeddings and family mass-share / count features; scenario-level structure features capture composition balance. |
| Mixture structure / composition balance | Relative mass fractions, number of active families, and dominant-component concentration alter interaction effects and failure modes. | Deep Sets element encoding over component rows plus scenario-level structure statistics and mass-distribution features. |
| Missing or noisy component-property data | Exact molecular descriptors are incomplete and heterogeneous; heavy reliance on wide weighted averages can amplify noise instead of mechanism. | Compact per-component numeric compression with masks/flags in the set branch; official shipping path also uses family-level tabular signals rather than wide weighted-property blocks. |

## Evidence Summary By Theme

### 1. Oxidation and viscosity are tightly linked but nonlinear
- Reviews of lubricant oxidation consistently describe free-radical autoxidation, acid formation, condensation, oligomerization, sludge, and varnish as core pathways behind performance loss.
- The same literature also explains why viscosity can move non-monotonically: fuel dilution and some decomposition products can reduce viscosity, while oxidation-product growth and oligomerization can increase it sharply.
- This supports the project observation that severe viscosity failures are not well explained by linear weighted-average chemistry alone.

### 2. Condition severity is not a side feature; it is a primary driver
- Temperature and test duration are repeatedly identified as first-order controls on oxidation rate.
- Biodiesel or methyl-ester contamination changes lubricant degradation behavior and can alter additive effectiveness.
- Metallic exposure and catalyst-related chemistry can accelerate oxidation and corrosion-linked degradation.
- This matches our factor analysis, where `test_temperature_c` and `test_duration_h` were the top two baseline features and `biofuel_mass_fraction_pct` also ranked near the top.

### 3. Family-level additive classes are more robust than exact component identity
- The lubricant literature is written largely in terms of functional classes: base oils, antioxidants, antiwear agents, detergents, dispersants, viscosity modifiers, pour-point depressants, friction modifiers.
- Those classes have stable mechanistic meaning even when exact supplier chemistry changes.
- In our experiments, family-only hybrid Deep Sets outperformed the family-plus-component variant, which is consistent with the idea that mechanism class generalizes better than sparse product identity on a small dataset.

### 4. Base-oil structure and additive balance matter jointly
- Base oil composition influences oxidative stability and viscosity behavior.
- Additives do not act independently; antioxidant systems, detergents, dispersants, antiwear packages, and biofuel contamination can interact positively or negatively.
- That interaction pattern motivates combining a component-level set encoder with a compact scenario-level tabular branch rather than relying on a single flat representation.

### 5. Why a hybrid Deep Sets model is reasonable here
- Deep Sets gives a principled permutation-invariant way to encode a variable number of components.
- DOT outcomes depend on both the unordered composition set and scenario-wide operating conditions.
- The best engineered tabular features capture stable global summaries such as family mass shares and structure counts.
- A hybrid model therefore mirrors the chemistry problem: local component evidence plus global severity context plus compact mixture-level summary statistics.

## Sources
1. Zaheer, M. et al. "Deep Sets." NeurIPS 2017. https://papers.nips.cc/paper/6931-deep-sets
2. Sharma, B. K. et al. "Oxidative Stability of Vegetal Oil-Based Lubricants." Lubricants, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC8900678/
3. Wang, J. et al. "Engine Oil Degradation Induced by Biodiesel: Effect of Methyl Oleate on the Performance of Zinc Dialkyldithiophosphate." ACS Omega, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6777094/
4. Xia, D. et al. "Research Progress of Antioxidant Additives for Lubricating Oils." Lubricants, 2024. https://www.mdpi.com/2075-4442/12/4/115
5. Spikes, H. "Phosphate Esters, Thiophosphate Esters and Metal Thiophosphates as Lubricant Additives." Lubricants, 2013. https://www.mdpi.com/2075-4442/1/4/132
6. Cabrera, J. et al. "Impact of Lubricant Additives on the Physicochemical Properties and Activity of Three-Way Catalysts." Catalysts, 2016. https://www.mdpi.com/2073-4344/6/4/54
7. Chupka, G. et al. "An Investigation of the Effect of Temperature on the Oxidation Processes of Metallic Diesel Engine Fuel System Materials and B100 Biodiesel in Exposure Testing." Heliyon, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7486612/
8. Liu, J. et al. "Evaluation of Antioxidant Properties and Molecular Design of Lubricant Antioxidants Based on QSPR Model." Lubricants, 2024. https://www.mdpi.com/2075-4442/12/1/3
9. Riazi, M. R. et al. "An Experimental Investigation on the Oxidative Desulfurization of a Mineral Lubricant Base Oil." Scientific Reports, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8617150/
10. Api.org. "API 1509 Documents." American Petroleum Institute, accessed April 16, 2026. https://www.api.org/products-and-services/standards/important-standards-announcements/api-1509
11. Api.org. "Oil Categories." American Petroleum Institute, accessed April 16, 2026. https://www.api.org/products-and-services/engine-oil/eolcs-categories-and-classifications/oil-categories
12. Wang, H. et al. "The Effect of Water Content on Engine Oil Monitoring Based on Physical and Chemical Indicators." 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC10891499/
13. Babu, K. et al. "Thermo-Oxidative Stability Studies on Some New Generation API Group II and III Base Oils." Fuel, 2002. https://www.sciencedirect.com/science/article/pii/S0016236101002101

## Concise Conclusion
The literature supports three decisions that matter most for this project. First, a hybrid Deep Sets architecture is well matched to the problem because lubricant scenarios are naturally sets of components evaluated under global operating conditions. Second, family-level representation is more defensible than exact component identity on this dataset because the chemistry literature is organized around functional additive classes and base-oil classes, and our own ablations showed better generalization from family-only encoding. Third, condition severity matters strongly: temperature, duration, biofuel loading, and catalyst chemistry repeatedly emerge as first-order oxidation drivers in the literature and were also the strongest signals in our factor analysis. Together, these findings support keeping the current official shipping path unchanged while using the human-review materials to explain why it is chemically plausible and competition-compliant.
