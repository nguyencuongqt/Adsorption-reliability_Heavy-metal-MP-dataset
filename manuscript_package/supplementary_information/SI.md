# Supplementary Information

## S1. Supplementary Methods

### S1.1. Fresh-study boundary

The supplementary analysis belongs to the `adsorption_reliability_study` layer and does not inherit any quantitative result from the legacy benchmark package. Only the final cleaned dataset, feature inventory, and metadata registry were reused as inputs.

### S1.2. Dataset structure

The locked modeling dataset contains 1009 rows, 23 study identifiers, and 149 experiment identifiers. Median rows per study are 38, and median rows per experiment are 6. The final response variable is `qe`, and the predictor space contains 25 real variables before synthetic null-feature augmentation. Variable definitions, units, and missingness are reported in `Table S1`.

### S1.3. Validation regimes

The primary hierarchy used for hypothesis testing was `random_cv` -> `group_exp` -> `group_aut`.

### S1.4. Interpretation stress testing

Seven synthetic null features were added for `RQ2`: three random noise features, two permuted real features, and two group-aware random-effect features keyed to studies and experiments. These features were used only to audit interpretation fragility.

### Text S1. Synthetic null features

Synthetic null features were appended only for interpretation diagnostics. By construction, these variables have no causal link to `qe` and were used as negative controls to test whether feature-importance methods assign importance to predictors that should not carry adsorption mechanism information. They were not treated as physical drivers and were excluded from mechanistic interpretation.

Three null-feature classes were used. First, pure-noise variables were sampled independently for each row. Second, permuted variables were created by shuffling observed predictors, preserving their marginal distributions while breaking row-level association with `qe`. Third, group-aware nulls were keyed to `aut_id` or `exp_id` to mimic grouped structure without encoding adsorption chemistry. The seven synthetic null features are listed in `Table S2`, summary statistics are reported in `Table S3`, and `Table S4` shows the augmented dataset structure. As expected, row-wise nulls had near-zero correlation with `qe`, whereas group-aware nulls could still show incidental sample-level correlation because they track grouped identifiers rather than adsorption chemistry. These variables were used for diagnostics only.

## S2. Supplementary Results

### S2.1. Full performance table

After restoring the legacy higher-capacity training pipeline, the neural baseline was strongest under `random_cv`, `LGBM` was strongest under `group_exp`, and `EN` had the lowest mean RMSE under `group_aut`. Figure S1 shows the same validation hierarchy on the `R2` scale and supports the interpretation of main-text `Figure 1A`: `random_cv` yields the strongest apparent fit, `group_exp` weakens performance, and study-level transfer drives all models toward much poorer or negative `R2`, with an extreme collapse for `MLP`. Because the grouped summaries rely on only five folds and show substantial dispersion, these comparisons should be interpreted descriptively rather than as formal significance-tested rankings.

### S2.2. Reliability ranking of importance methods

After correcting the null-feature audit, `EN` coefficient magnitude ranked highest in the current reliability table because it combined sparsity with near-complete null suppression. SHAP remained more interpretable than gain in a mechanism sense, but not fully null-free. `LGBM` built-in gain stayed highly stable across splits, yet it also showed the largest null-importance share overall, indicating that rank stability alone can be misleading. These results still argue for method triangulation rather than single-method interpretation, especially when higher-order engineered terms are present in the feature space.

### S2.3. Mechanism-group comparison

Solution-chemistry variables were the most persistent broad signal across methods, but the weight assigned to aging, surface area, polymer identity, metal identity, and higher-order engineered terms varied meaningfully by method. Main-text mechanism visualizations therefore exclude synthetic null features, while the null benchmark is reported separately. This indicates that broad mechanism-group agreement is more reliable than highly specific rank claims for individual features.

## S3. Supplementary Figures

- Figure S1. Validation hierarchy on the `R2` scale across `random_cv`, `group_exp`, and `group_aut`.
- Figure S2. Importance stability heatmap across methods and validation regimes.

## S4. Supplementary Tables

- Table S1. Dataset variable summary, including definitions, units, and missingness.
- Table S2. Synthetic null-feature catalog, including feature name, type, and construction.
- Table S3. Summary statistics for synthetic null features, including mean, standard deviation, and Pearson correlation with `qe`.
- Table S4. Example rows from the augmented dataset showing the appended synthetic null features.
- Table S5. Feature-importance method reliability summary.
- Table S6. Mechanism-group importance summary across methods and regimes.
- Table S7. Feature-level importance summary including synthetic null features.
