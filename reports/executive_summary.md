# Executive Summary — Home Credit Default Risk Model

**Audience:** Trainer, Manager, Non-Technical Stakeholders | **Date:** 2026-06-12

## Problem Statement
Home Credit serves underbanked populations with limited traditional credit history. The business must identify applicants likely to default before approval while maintaining financial inclusion.

## Business Objective
Minimize portfolio losses from bad loans without unnecessarily rejecting creditworthy applicants — optimizing the decline/approve tradeoff under a fixed loss budget.

## Methodology
- Joined 6 subsidiary credit tables with application data (~307K training records)
- Engineered temporal payment, overdue, utilization, trend, and behavioral features
- Trained stratified 5-fold ensemble (LightGBM + CatBoost) with stacking and isotonic calibration
- Validated with OOF metrics, leakage audit, SHAP explainability, and error analysis

## Results
| Metric | Value |
| --- | --- |
| OOF ROC-AUC (calibrated) | **0.795** |
| OOF PR-AUC | N/A |
| Features | 10 |
| Calibration | isotonic |

## Key Insights
1. EXT_SOURCE_MEAN
2. AMT_CREDIT

- Recent payment deterioration (payment_trend, overdue_trend) adds signal beyond static scores
- Three-tier segmentation (low/medium/high) maps directly to operational workflows
- Calibrated probabilities enable expected-loss pricing

## Recommendations
1. Deploy calibrated stacking ensemble for triage — not raw scores
2. Auto-approve low-risk segment; manual review medium; decline high
3. Monthly monitoring of score drift and feature PSI
4. Quarterly threshold recalibration

## Limitations
- Trained on historical Kaggle data; macro shifts require retraining
- Full feature pipeline required at scoring time
- Model supports triage, not sole automated denial without human review
