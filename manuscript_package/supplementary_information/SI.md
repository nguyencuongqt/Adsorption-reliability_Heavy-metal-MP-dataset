# Supplementary Information

## S1. Supplementary Methods

### S1.1. Fresh-study boundary

The supplementary analysis belongs to the `adsorption_reliability_study` layer and does not inherit any quantitative result from the legacy benchmark package. Only the final cleaned dataset, feature inventory, and metadata registry were reused as inputs.

### S1.2. Dataset structure

The final dataset contains 1009 rows, 23 study identifiers, and 149 experiment identifiers. Median rows per study are 38, and median rows per experiment are 6. The locked feature set contains 25 real predictors before synthetic null-feature augmentation.

### S1.3. Validation regimes

The primary hierarchy used for hypothesis testing was `random_cv` -> `group_exp` -> `group_aut`. An optional `leave_one_study_out` regime was included as a harsher stress test, but its mean metrics were treated cautiously because held-out studies vary strongly in size and target variance.

### S1.4. Interpretation stress testing

Seven synthetic null features were added for `RQ2`: three random noise features, two permuted real features, and two group-aware random-effect features keyed to studies and experiments. These features were used only to audit interpretation fragility.

## S2. Supplementary Results

### S2.1. Full performance table

After restoring the legacy higher-capacity training pipeline, the neural baseline was strongest under `random_cv`, LightGBM was strongest under `group_exp`, and Elastic Net was least degraded by RMSE under `group_aut`. The auxiliary `leave_one_study_out` regime still showed severe instability in `R2`, reinforcing that study-level domain shift is the hardest regime and that fold-level variance remains high when holding out entire studies one at a time.

### S2.2. Reliability ranking of importance methods

LightGBM built-in gain and Elastic Net coefficient magnitude ranked highest in the current reliability table, but for different reasons: built-in gain was especially stable in the rerun pipeline, whereas Elastic Net coefficients remained sparse and conservative. SHAP remained more interpretable than gain in a mechanism sense, but not fully null-free. These results still argue for method triangulation rather than single-method interpretation, especially when higher-order engineered terms are present in the feature space.

### S2.3. Mechanism-group comparison

Solution-chemistry variables were the most persistent broad signal across methods, but the weight assigned to aging, surface area, polymer identity, metal identity, and higher-order engineered terms varied meaningfully by method. This indicates that broad mechanism-group agreement is more reliable than highly specific rank claims for individual features.

## S3. Supplementary Figures

- Figure S1. Importance stability heatmap across methods and validation regimes.

## S4. Supplementary Tables

- Table S1. Feature-importance method reliability summary.
- Table S2. Mechanism-group importance summary across methods and regimes.
- Table S3. Feature-level importance summary including synthetic null features.
