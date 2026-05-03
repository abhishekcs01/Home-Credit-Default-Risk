# Final Report

## Objective
Predict borrower default risk for the Home Credit dataset and generate competition-ready probability scores.

## Pipeline Overview
- Raw data loading from `data/raw/`
- Table-level aggregation to `SK_ID_CURR` grain
- Application-level feature engineering
- Reproducible design-matrix preprocessing
- LightGBM holdout validation + K-fold ensemble
- Submission generation from processed test set

## Evaluation Metric
- Primary metric: ROC-AUC
- Validation includes logistic baseline and LightGBM holdout AUC
- Final training uses stratified K-fold ensembling and reports OOF ROC-AUC

## Artifacts
- Processed datasets: `data/processed/merged_train.pkl`, `data/processed/merged_test.pkl`
- Model bundle: `models/lightgbm_model.pkl`
- Figures: `reports/figures/roc_curve.png`, `reports/figures/feature_importance.png`
- Submission: `submission.csv`

