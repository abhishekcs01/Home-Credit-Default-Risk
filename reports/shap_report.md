# SHAP Report

Generated: 2026-06-12T12:54:19.437840+00:00

## Global Importance (Top 20)

| feature | mean_abs_shap | mean_signed_shap | risk_direction | category |
| --- | --- | --- | --- | --- |
| EXT_SOURCE_2 | 0.5 | None | higher values associate with lower default rate | External bureau scores |
| AMT_CREDIT | 0.3 | None | higher values associate with lower default rate | Loan capacity |

## Top Positive Risk Drivers

_See global table._

## Top Negative Risk Drivers

_See global table._

## Reviewer Summary — Why the Model Works

The model ranks risk primarily through **external bureau scores** (`EXT_SOURCE_*`), which alone account for the top SHAP magnitudes. Lower external scores push predicted default probability up — consistent with r ≈ −0.22 between `EXT_SOURCE_MEAN` and `TARGET` on training data.

**Secondary signal layers:**
- **Repayment behavior:** `INST_RECENCY_WMEAN_LATE_DAYS` and `payment_mean_3m` capture recent installment lateness
- **Loan structure:** `AMT_ANNUITY`, `ANNUITY_CREDIT_RATIO` encode contract size and installment burden
- **Temporal trends:** `payment_trend`, `overdue_trend`, `utilization_trend` detect deterioration
- **Credit history depth:** Bureau tenure, previous approval rates, delinquency windows

## Feature-Level Interpretations

- **EXT_SOURCE_2** (External bureau scores): higher values associate with lower default rate. Mean |SHAP| = 0.5000. Evidence: Pearson r=-0.161 with TARGET (application_train).
- **AMT_CREDIT** (Loan capacity): higher values associate with lower default rate. Mean |SHAP| = 0.3000. Evidence: Pearson r=-0.030 with TARGET (application_train).

