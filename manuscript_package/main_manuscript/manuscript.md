# Validation and Interpretation Reliability in Literature-Derived Prediction of Heavy-Metal Adsorption by Microplastics

## Abstract

Machine learning on literature-derived environmental data is often evaluated with row-level validation and interpreted with a single feature-importance method, even when the data are strongly grouped and heterogeneous. Here we studied reliability of both prediction and interpretation for a compiled dataset of heavy-metal adsorption by microplastics containing 1009 observations from 149 experiments nested within 23 studies. We framed two research questions: how grouped validation changes estimated predictive performance, and how sensitive mechanistic conclusions are to the feature-importance method. Using the locked final dataset and the legacy high-capacity training pipeline, we compared random row-level cross-validation, grouped experiment-level validation, grouped study-level validation, and an auxiliary leave-one-study-out stress test. The neural baseline (`MLPRegressor`) performed best under random cross-validation (`R2 = 0.912 +/- 0.039`, `RMSE = 0.618 +/- 0.118`), while LightGBM gave the strongest experiment-level transfer (`R2 = 0.725 +/- 0.203`, `RMSE = 0.969 +/- 0.387`). Study-level grouped validation remained difficult, with Elastic Net showing the least severe degradation by RMSE (`RMSE = 1.674 +/- 1.671`) but negative `R2`. For interpretation, we compared standardized Elastic Net coefficients, permutation importance, LightGBM built-in gain, and SHAP, while augmenting the dataset with synthetic null features. Importance rankings differed materially by method, and null-feature benchmarking showed that no interpretation method was uniformly superior across all criteria. These results show that random validation can be substantially optimistic in grouped literature-derived environmental data, and that interpretation reliability should be stress-tested rather than assumed from a single method.

**Keywords:** grouped validation; microplastics; heavy metals; adsorption; feature importance; SHAP; null features; interpretation reliability

## 1. Introduction

Literature-derived environmental datasets are attractive for machine-learning synthesis because they pool evidence across many experiments and materials. However, such datasets rarely satisfy the independent and identically distributed assumptions that are often implicitly adopted during model evaluation. Rows are commonly clustered within experiments and studies, reporting conventions differ across papers, and missingness patterns often reflect study-specific protocols rather than random omission. Under these conditions, row-level validation can inflate apparent performance by allowing study fingerprints to appear in both training and validation subsets.

Interpretation faces a related problem. Feature-importance analyses are widely used to translate predictive models into mechanistic narratives, yet the resulting rankings can depend strongly on the model class and the importance method. Moreover, many studies treat importance rankings as if they directly support causal mechanism claims, even though importance measures generally reflect predictive usefulness within a given model and dataset. The risk is especially high in heterogeneous environmental tabular data, where spurious proxies and grouped structure can distort both prediction and interpretation.

This study therefore asks two reliability-focused questions rather than a conventional best-model question. `RQ1` examines how validation design under grouped literature-derived data changes estimated predictive performance and apparent generalization. `RQ2` examines how sensitive mechanistic conclusions are to the choice of feature-importance method, and whether synthetic null features can expose interpretation methods that are especially vulnerable to spurious attribution. The study is framed as an analysis of reliability under grouped data, not as a competition to maximize a benchmark score.

## 2. Materials and Methods

### 2.1. Dataset and grouped structure

The analysis used the final cleaned modeling dataset from the project pipeline, but all results in this manuscript were re-run inside a fresh study layer. The dataset contains 1009 observations, 23 studies (`aut_id`), and 149 experiments (`exp_id`). The target is adsorption capacity (`qe`). The feature set contains 25 locked predictors derived from the final compact modeling dataset, including solution-condition variables, polymer indicators, metal indicators, aging indicators, functional-group indicators, and missingness indicators. The target distribution is strongly right-skewed, with median `qe = 0.103`, 95th percentile `qe = 5.052`, and maximum `qe = 17.779`.

### 2.2. Modeling target and preprocessing

The target was modeled as `log1p(qe)` to reduce the influence of extreme right-tail values while keeping evaluation on the original `qe` scale. All preprocessing steps that could leak information were fit inside training folds only. Specifically, continuous predictors were imputed with training-fold medians, and scaling was fit on training data only for models that required it. Validation rows were never used to fit imputers, scalers, or model parameters.

### 2.3. Validation hierarchy

We defined three main validation regimes and one optional stress-test regime. `random_cv` corresponds to repeated row-level interpolation within the observed literature domain. `group_exp` uses grouped cross-validation by `exp_id` and represents transfer to unseen experiments while some study context remains represented in training. `group_aut` uses grouped cross-validation by `aut_id` and represents transfer to unseen studies. `leave_one_study_out` was added as a harsher auxiliary stress test, but was interpreted cautiously because fold-level variance is large when held-out studies differ strongly in size and target variance.

### 2.4. Predictive models

To keep the comparison compact and scientifically interpretable, we used three representative models: Elastic Net, LightGBM, and a regularized multilayer perceptron baseline. This set spans a sparse linear model, a nonlinear tree-boosting model, and a lightweight neural baseline without turning the study into a broad benchmark contest.

### 2.5. Feature-importance methods and null-feature stress testing

For `RQ2`, we compared four main importance methods: standardized coefficient magnitude for Elastic Net, permutation importance for both Elastic Net and LightGBM, LightGBM built-in gain, and SHAP for LightGBM. We also added seven synthetic null features: pure random noise features, permuted versions of real features, and group-aware random-effect features keyed to studies and experiments but designed to remain non-informative about `qe`. We evaluated methods by null suppression, separation between real and null features, top-rank contamination by nulls, ranking stability across resampling, and robustness across validation regimes. These diagnostics were treated as evidence of interpretation reliability, not as universal proof that a method is correct.

## 3. Results and Discussion

### 3.1. Dataset structure confirms a grouped and heterogeneous prediction problem

The dataset has a pronounced hierarchical structure. The 1009 rows are distributed across 23 studies and 149 experiments, with median 38 rows per study and median 6 rows per experiment. Missingness remains substantial for key continuous variables such as `ph` (`53.3%`) and `sa` (`38.0%`). Together with the strongly skewed target, this structure makes the dataset a suitable test bed for evaluating reliability under grouped environmental data rather than for claiming universal predictive performance.

### 3.2. Random validation is optimistic relative to grouped validation

The performance hierarchy clearly supports `H1` and `H2`, although the strongest model changed once the old higher-capacity training pipeline was restored. Under `random_cv`, the neural baseline achieved the strongest performance (`R2 = 0.912 +/- 0.039`, `RMSE = 0.618 +/- 0.118`, `MAE = 0.223 +/- 0.038`). Under `group_exp`, LightGBM gave the strongest transfer result (`R2 = 0.725 +/- 0.203`, `RMSE = 0.969 +/- 0.387`, `MAE = 0.383 +/- 0.164`), whereas the neural baseline weakened but remained competitive. Under `group_aut`, performance degraded sharply for all models; Elastic Net had the lowest RMSE (`1.674 +/- 1.671`), LightGBM had a similar RMSE but worse `R2`, and the neural baseline became highly unstable. These results indicate that random validation still overstates reliability even when a stronger legacy training pipeline is used.

These results suggest that row-level validation materially overstates predictive reliability for grouped literature-derived data. The key implication is not that the models are useless, but that they should be interpreted according to the generalization regime. Row-level interpolation inside the observed domain is feasible, experiment-level transfer is harder but partially workable, and study-level transfer remains the dominant challenge.

### 3.3. Different model classes respond differently to grouped transfer

The models did not fail in the same way, which supports `H3`. The neural baseline was strongest in `random_cv`, indicating that the legacy training pipeline can exploit rich in-domain structure when train and validation remain close. LightGBM was strongest in `group_exp`, suggesting that boosted trees remain the most reliable nonlinear option when some study context still transfers. By contrast, Elastic Net was least degraded at the study-grouped level, which suggests that a simpler model may be more conservative under severe cross-study shift. The neural baseline deteriorated most strongly in grouped study transfer despite its high random-split performance, which is exactly the kind of regime-sensitive behavior this study was designed to expose.

### 3.4. Mechanistic conclusions are method-sensitive

The interpretation results support `H4`. Across methods, the broad importance pattern remained method-sensitive rather than uniform. Solution-chemistry variables were consistently prominent, but the relative weight assigned to aging, surface area, polymer identity, metal identity, and higher-order engineered terms changed meaningfully with the importance method. Elastic Net coefficients emphasized solution chemistry while remaining highly sparse. LightGBM SHAP preserved a chemically plausible ranking but still assigned non-trivial importance to null features. LightGBM built-in gain was stable in the rerun pipeline, but because it is model-specific and can favor particular split structures, it should still be interpreted cautiously. LightGBM permutation importance remained the least stable family overall.

This disagreement is scientifically important. If a paper used only one importance method, it could tell a much more confident mechanism story than the data justify. Here, the cross-method comparison shows that some mechanism-level conclusions are relatively stable, while others remain method-dependent.

### 3.5. Null features reveal interpretation fragility

The null-feature results support `H5` and `H6`. Under the restored legacy training pipeline, LightGBM built-in gain and Elastic Net coefficient magnitude both showed near-complete null suppression in the current audit table, but for different reasons: built-in gain was more stable across splits, whereas Elastic Net remained more conservative and sparse. LightGBM SHAP remained relatively stable but still assigned non-trivial importance to null features, especially under `group_aut`, whereas permutation importance stayed the least stable family overall. These differences show that stronger predictive performance does not automatically translate into a single universally trustworthy interpretation method; interpretation still depends on how null suppression, stability, and method-specific bias are balanced.

These findings do not imply that Elastic Net reveals the true mechanism or that SHAP is invalid. Instead, they show that interpretation methods differ in their vulnerability to spurious attribution under grouped environmental data. Null benchmarking therefore serves as a practical audit tool: it helps identify which methods are more conservative and which methods require more caution when being translated into mechanistic discussion.

## 4. Conclusion

This study reframed literature-derived adsorption modeling as a reliability problem rather than a best-model contest. Random row-level validation produced substantially stronger performance estimates than grouped experiment-level and grouped study-level validation, confirming that validation design strongly affects apparent predictive success. At the same time, mechanistic interpretation was shown to be method-sensitive, and null-feature stress testing exposed clear differences in how aggressively importance methods promote spurious signal. The broader implication is that reliable environmental ML requires stress-testing both prediction and interpretation. For literature-derived grouped data, importance should be reported as predictive relevance under explicit validation assumptions, not as direct causal evidence.

## References

References should be inserted from the author-managed bibliography. No citation list is auto-generated in this package.
