# Figure Captions

**Figure 1.** Performance drop across the main validation hierarchy. Panel A compares `random_cv`, `group_exp`, and `group_aut` using mean `RMSE +/- SD` on the original `qe` scale, with the extreme `MLP` study-level result labeled directly to preserve readability. Panel B rescales grouped-regime `RMSE` to each model's `random_cv` baseline, highlighting how predictive performance deteriorates as validation moves from row-level interpolation toward study-level transfer.

**Figure 2.** Mechanism-group importance across methods. The heatmap compares the relative importance of solution chemistry, surface area and porosity, metal identity, surface functionality, polymer identity, aging state, and other engineered terms across EN coefficient, EN permutation, LGBM permutation, LGBM gain, and LGBM SHAP. Synthetic null features are excluded from this visualization and audited separately in Figure 3.

**Figure 3.** Null-feature benchmarking across importance methods. Because the corrected audit yielded near-zero mean null-importance share and top-10 null intrusion across methods, the figure emphasizes the remaining informative diagnostics: mean real-vs-null importance separation and mean rank stability. Together, these panels show the tradeoff between conservative null suppression and ranking stability across methods.
