# Final Report

## Objective
Predict borrower default risk for the Home Credit dataset and generate competition-ready probability scores.

## Pipeline Overview
- Raw data loading from `data/raw/`
- Table-level aggregation to `SK_ID_CURR` grain
- Application-level feature engineering
- Leakage-safe fold-fitted preprocessing and target encoding
- LightGBM + CatBoost blended K-fold ensemble
- Submission generation from processed test set

## Evaluation Metric
- Primary metric: ROC-AUC
- Validation includes leakage-safe holdout baseline checks
- Primary training/evaluation uses stratified K-fold OOF ROC-AUC/PR artifacts

## Artifacts
- Processed datasets: `data/processed/merged_train.pkl`, `data/processed/merged_test.pkl`
- Model bundle: `models/ensemble_model.pkl`
- Figures: `reports/figures/roc_curve.png`, `reports/figures/pr_curve.png`, `reports/figures/feature_importance.png`
- Metrics: `outputs/metrics/training_metrics.json`, `outputs/metrics/fold_metrics.csv`
- Submission: `outputs/submissions/submission.csv`

