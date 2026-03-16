# Discussion Notes

- Random CV can be optimistic in grouped literature-derived environmental data because row-level splits allow study and experiment fingerprints to leak into both train and validation.
- Study-level transfer is the hardest and often the most realistic test for future literature-derived generalization because an unseen study may carry new reporting patterns, missingness, and hidden protocol choices.
- The optional `leave_one_study_out` regime is a harsher stress test, but its fold means can be noisy because held-out studies differ strongly in size and target variance; it should therefore be read as qualitative stress evidence rather than a cleaner headline metric than grouped five-fold study CV.
- Feature-importance conclusions depend strongly on method choice, model class, and validation regime; agreement across methods is informative, but disagreement is itself a result.
- Null-feature benchmarking helps detect interpretive fragility by showing whether an importance method assigns meaningful weight to features that should not carry true signal.
- Importance should be read as predictive relevance under a given model and split design, not as evidence of causal adsorption mechanisms.
