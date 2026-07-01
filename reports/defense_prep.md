# Defense Preparation — Top 100 Trainer Questions

Evidence base: OOF calibrated AUC 0.795, PR-AUC N/A, 10 features, 5-fold stratified CV.

## 1. Why LightGBM?

**Answer:** LightGBM achieved the highest standalone OOF ROC-AUC on our tabular feature matrix. Its histogram-based leaf-wise boosting handles heterogeneous credit features efficiently after one-hot expansion and target encoding.

## 2. Why CatBoost in the ensemble?

**Answer:** CatBoost provides model diversity for stacking even when linear blend weight goes to zero. Different inductive bias reduces correlated ranking errors across folds.

## 3. Why XGBoost was optional?

**Answer:** XGBoost adds marginal lift on some splits but increases training time. It is enabled via config when GPU/CPU budget allows; default off to keep CI fast.

## 4. Why SHAP?

**Answer:** SHAP provides consistent, additive feature attributions required for adverse-action reason codes and model risk documentation in regulated lending.

## 5. Why StratifiedKFold?

**Answer:** Default rate is ~8%. Stratification ensures each fold has stable positive counts for reliable AUC and threshold estimation.

## 6. Why calibration?

**Answer:** Ranking quality (AUC) does not guarantee reliable probabilities. Isotonic calibration improves Brier score for expected-loss pricing and tiered limits.

## 7. Why threshold optimization?

**Answer:** Underwriting is a decision problem, not pure ranking. F1-optimal threshold on OOF data balances precision and recall for triage policies.

## 8. Why ROC-AUC instead of accuracy?

**Answer:** Accuracy is misleading on imbalanced credit data (~92% non-default). AUC measures rank quality independent of class prevalence.

## 9. What is leakage?

**Answer:** Using information not available at decision time — e.g., future payment outcomes or target statistics computed on validation data.

## 10. Why this feature engineering strategy?

**Answer:** Credit default is driven by capacity (income vs obligation), external scores, and behavioral history. We aggregate subsidiary tables point-in-time and engineer ratios, trends, and domain interactions.

## 11. Why temporal windows (3m/6m/12m)?

**Answer:** Recent behavior is more predictive than distant history. Rolling windows capture payment deterioration before it appears in long-run aggregates.

## 12. Why payment_mean_3m/6m/12m?

**Answer:** Installment payment ratio over 90/180/365 days measures repayment discipline at multiple horizons.

## 13. Why overdue_count features?

**Answer:** Delinquency frequency in bureau and installment windows is a direct precursor to default.

## 14. Why utilization_3m/6m/12m?

**Answer:** High revolving utilization signals liquidity stress and correlates with default on new installment loans.

## 15. Why trend features?

**Answer:** A borrower worsening from 12m to 3m behavior is riskier than one with stable mediocre history.

## 16. Why debt_growth_rate?

**Answer:** Rapid debt accumulation relative to total exposure indicates over-leveraging.

## 17. Why interaction features?

**Answer:** Domain-informed interactions (e.g., low EXT_SOURCE × high overdue) capture compounding risk not visible in marginals.

## 18. Why target encoding?

**Answer:** High-cardinality categoricals (occupation, organization) would explode with one-hot. Target encoding compresses default-rate signal.

## 19. How is target encoding kept leak-free?

**Answer:** Each CV fold fits encoding maps on training split only; validation receives encoded values from train statistics.

## 20. Why isotonic over Platt?

**Answer:** Tree model scores often have non-logistic distortion. Isotonic fits monotonic step functions with lower Brier on our OOF data.

## 21. Why stacking over simple blend?

**Answer:** Stacking learns conditional reweighting of base learners across the score distribution, yielding modest AUC lift over best single model.

## 22. Why per-fold preprocessors?

**Answer:** Target encoding and median imputation must not see validation rows during fit — per-fold preprocessors enforce this.

## 23. Why PR-AUC in addition to ROC-AUC?

**Answer:** PR-AUC focuses on the rare positive class (defaulters) and is more informative when approval rates are low.

## 24. Why Brier score?

**Answer:** Brier measures probability calibration quality — critical when scores drive pricing and reserves.

## 25. Why scale_pos_weight?

**Answer:** Class imbalance (~1:11) would bias trees toward majority class without reweighting.

## 26. Why early stopping?

**Answer:** Prevents overfitting on training folds; uses validation AUC with patience of 120 rounds.

## 27. Why 5 folds?

**Answer:** Balance between variance reduction and compute cost on ~307K rows.

## 28. Why feature pruning?

**Answer:** Bottom 6% by ensemble importance removes noise features that add variance without signal.

## 29. Why missing indicators?

**Answer:** Missingness in EXT_SOURCE and occupation is informative — thin-file applicants behave differently.

## 30. Why drop columns with >70% missing?

**Answer:** Near-empty columns add dimensionality without stable signal and increase overfitting risk.

## 31. Why recency-weighted aggregates?

**Answer:** Exponential decay on MONTHS_BALANCE emphasizes recent bureau/CC/POS behavior over stale months.

## 32. Why bureau_balance windows?

**Answer:** Monthly status progression captures delinquency streaks invisible in bureau snapshot alone.

## 33. Why previous_application features?

**Answer:** Prior refusals and approval patterns reveal underwriting history and credit hunger.

## 34. Why installment payment diff?

**Answer:** Days between scheduled and actual payment directly measures lateness severity.

## 35. Why POS and credit card tables?

**Answer:** Revolving and POS products expose complementary behavioral signals to installment loans.

## 36. Why EXT_SOURCE composite features?

**Answer:** Three bureau scores are noisy individually; mean/min/max/std composite stabilizes signal.

## 37. Why ANNUITY_CREDIT_RATIO?

**Answer:** Near-term repayment burden relative to principal indicates affordability stress.

## 38. Why CREDIT_INCOME_RATIO?

**Answer:** Classic leverage measure — high credit relative to income increases default probability.

## 39. Why DELINQ_COMPOSITE_INDEX?

**Answer:** Combines POS, CC, and installment delinquency into single behavioral stress indicator.

## 40. Why CPU fallback for LightGBM?

**Answer:** CUDA LightGBM builds can produce degenerate predictions (AUC≈0.5). Auto-fallback to CPU preserves model integrity.

## 41. Why training sanity gate?

**Answer:** Fail fast if any base learner AUC < 0.60 — prevents silent deployment of broken models.

## 42. Why automated leakage audit?

**Answer:** 723+ features require systematic classification (SAFE / REVIEW / HIGH RISK) for governance and viva defense.

## 43. Why error analysis report?

**Answer:** FP/FN patterns guide policy overrides and feature improvements beyond aggregate AUC.

## 44. Why FastAPI for inference?

**Answer:** Production-grade async API with schema validation, batching, and calibration metadata in responses.

## 45. Why fold-averaged test predictions?

**Answer:** Each fold model scores test data; averaging reduces variance vs single model.

## 46. Why not use test labels?

**Answer:** Kaggle test has no public labels; production never has labels at scoring time.

## 47. Why separate train/test preprocessing?

**Answer:** Test applicants must not influence imputation, encoding, or aggregation statistics.

## 48. Why pickle for model bundle?

**Answer:** Preserves fold preprocessors, boosters, stacker, and calibrator atomically for inference.

## 49. Why 90% test coverage?

**Answer:** Ensures pipeline regressions are caught before trainer review or deployment.

## 50. Why Docker?

**Answer:** Reproducible inference environment with pinned dependencies for staging/production parity.

## 51. Why Optuna optional?

**Answer:** Hyperparameter search improves AUC but multiplies training time; enabled via config for final runs.

## 52. Why logistic meta-learner for stacking?

**Answer:** Simple, interpretable, low-variance combiner for two-three correlated base predictions.

## 53. Why not neural networks?

**Answer:** Tabular credit data with mixed types favors GBDT ensembles; NNs need more data and tuning for marginal gain.

## 54. Why not random feature interactions?

**Answer:** Random crosses explode dimensionality and leak noise; domain interactions target known credit-risk mechanisms.

## 55. How do you handle class imbalance at threshold?

**Answer:** OOF F1 grid search selects threshold; business can override for cost-sensitive decline/approve tradeoff.

## 56. What is a false positive in this context?

**Answer:** Predicting default for a customer who repays — causes lost revenue from unnecessary decline.

## 57. What is a false negative?

**Answer:** Missing a defaulter — causes direct credit loss.

## 58. Which error is costlier?

**Answer:** False negatives (missed defaults) typically have higher financial cost; threshold tuning reflects this.

## 59. Why monitor SHAP drift?

**Answer:** Shift in feature importance indicates population or data pipeline change requiring retraining.

## 60. Why segment by risk tier?

**Answer:** Low/medium/high bands map to auto-approve, manual review, and decline workflows.

## 61. What is low risk segment?

**Answer:** Calibrated probability below ~5% with strong EXT_SOURCE and clean payment history.

## 62. What is medium risk segment?

**Answer:** Probability 5–15% or mixed signals — route to enhanced verification.

## 63. What is high risk segment?

**Answer:** Probability above ~15% or rising overdue/utilization trends — decline or reduced limits.

## 64. Why external scores dominate?

**Answer:** Third-party bureau indices summarize longitudinal credit behavior beyond application form.

## 65. Why behavioral features matter?

**Answer:** Stated income can be incomplete; actual payment behavior is revealed preference.

## 66. Why previous refusal rate?

**Answer:** Repeated refusals may indicate undisclosed risk or adverse selection.

## 67. Why credit velocity?

**Answer:** Rapid new credit origination may signal financial stress.

## 68. Why employment tenure features?

**Answer:** Job stability correlates with repayment capacity.

## 69. Why family size ratios?

**Answer:** Income per dependent affects discretionary capacity for loan repayment.

## 70. Why skew in aggregations?

**Answer:** Distribution shape captures outliers — one severe delinquency vs consistent mild lates.

## 71. Why max delinquency streak?

**Answer:** Longest consecutive delinquent months indicates persistent payment problems.

## 72. Why severe delinquency flags?

**Answer:** STATUS >= 3 in bureau_balance marks serious deterioration beyond minor lates.

## 73. Why high utilization frequency?

**Answer:** Repeated months above 90% utilization indicate chronic revolving stress.

## 74. Why payment acceleration?

**Answer:** Increasing late-payment rate in 90d vs 365d windows signals deterioration.

## 75. Why PREV_APPROVED_AMONG_DECIDED?

**Answer:** Approval rate among decided applications measures historical creditworthiness perception.

## 76. Why fold-safe feature selection?

**Answer:** Importance computed on fold-0 probe only; minor optimistic bias acknowledged in leakage audit.

## 77. Why mutual information in feature ranking?

**Answer:** Captures non-linear univariate signal complementary to tree gain.

## 78. Why composite feature ranking?

**Answer:** Blends gain, MI, and SHAP for robust importance beyond single method bias.

## 79. Why not use all Kaggle tables?

**Answer:** Some tables (e.g., application_test) lack labels; we use all labeled subsidiary tables with SK_ID_CURR join.

## 80. Why DAYS_EMPLOYED sentinel?

**Answer:** 365243 indicates unemployed; replaced with NaN to avoid distorting tenure features.

## 81. Why float32 downcasting?

**Answer:** Reduces memory for 300K × 700+ feature matrix during aggregation and training.

## 82. Why batch merge in aggregation?

**Answer:** Joining 700+ columns in batches prevents memory spikes on consumer hardware.

## 83. Why incremental table study?

**Answer:** Quantifies marginal AUC lift from each subsidiary table for ablation narrative.

## 84. Why holdout evaluate_models?

**Answer:** Quick sanity check for feature stages; production uses full CV ensemble.

## 85. How often retrain?

**Answer:** Quarterly or when PSI > 0.2 on top features or realized default rate deviates >10% from expected.

## 86. What triggers model rollback?

**Answer:** OOF AUC drop >0.01 on champion/challenger, or FP rate spike in production monitoring.

## 87. Why fairness considerations?

**Answer:** Proxy features (region, gender OHE) require disparate impact monitoring even if not used directly.

## 88. How provide adverse action reasons?

**Answer:** Top SHAP drivers per applicant translated to plain-language factors (e.g., recent late payments).

## 89. Why local SHAP examples?

**Answer:** Global importance does not explain individual decisions; local examples support case review.

## 90. Why confidence cutoff in error analysis?

**Answer:** High-confidence wrong predictions are most harmful and deserve priority review.

## 91. Why business_strategy report?

**Answer:** Translates model scores into operational lending actions by risk segment.

## 92. Why model_governance report?

**Answer:** Documents assumptions, limitations, monitoring, and retraining for model risk management.

## 93. Why executive_summary?

**Answer:** Non-technical stakeholders need problem, results, and actions without reading code.

## 94. What is OOF prediction?

**Answer:** Out-of-fold prediction — each row scored by model that did not train on it.

## 95. Why not train on full data for final model?

**Answer:** CV ensemble uses all data across folds; each applicant scored by held-out fold model.

## 96. Why schema_version in API?

**Answer:** Clients can detect feature schema changes and handle backward compatibility.

## 97. Why calibration_method in API response?

**Answer:** Downstream systems know whether probabilities are isotonic-calibrated.

## 98. Why threshold in API response?

**Answer:** Documents default decision cutoff used during training evaluation.

## 99. Why sparse API payloads warn?

**Answer:** Full 723-feature vector required for accurate scores; sparse input uses imputation fallback.

## 100. Why thread-safe CatBoost predict?

**Answer:** FastAPI serves concurrent requests; CatBoost predict is not thread-safe without locking.
