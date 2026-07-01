# Business Strategy — Risk Segmentation & Operations

Generated: 2026-06-12T12:54:19.439640+00:00

## Model Performance Context
Calibrated OOF ROC-AUC: **0.795** | Default rate: ~8%

## Risk Segments

### Low Risk (calibrated P(default) < 5%)

| Aspect | Recommendation |
| --- | --- |
| **Lending** | Auto-approve standard product; offer preferential rates where policy allows |
| **Review** | Minimal — spot-check only for data quality anomalies |
| **Operations** | Straight-through processing; digital onboarding |

**Profile:** Strong EXT_SOURCE scores, low overdue_count_3m, stable payment_mean_12m, utilization_3m < 50%.

### Medium Risk (5% ≤ P(default) < 15%)

| Aspect | Recommendation |
| --- | --- |
| **Lending** | Approve with conditions — reduced limit, shorter tenor, or co-borrower |
| **Review** | Enhanced verification (income docs, employer callback) |
| **Operations** | Route to senior underwriter; 24–48h SLA |

**Profile:** Mixed signals — decent external score but rising payment_trend or moderate utilization_6m.

### High Risk (P(default) ≥ 15%)

| Aspect | Recommendation |
| --- | --- |
| **Lending** | Decline or micro-loan pilot with strict monitoring |
| **Review** | Mandatory manual review with reason codes from SHAP |
| **Operations** | Adverse action notice with top 4 contributing factors |

**Profile:** Weak EXT_SOURCE, elevated overdue_count_12m, high credit_usage_change_rate, prior refusals.

## Portfolio Actions

1. **Monthly:** Score distribution dashboard by segment; alert if high-risk share exceeds baseline + 2σ
2. **Quarterly:** Recalibrate thresholds against realized default rates by segment
3. **Ongoing:** Monitor payment_trend and overdue_trend in booked portfolio for early warning
4. **Policy override:** Auto-decline override for applicants with EXT_SOURCE_MEAN > 0.65 AND payment_mean_12m > 0.95

## Expected Impact

| Action | Metric |
| --- | --- |
| Low-risk straight-through | ~40% of applications, <2% default rate |
| Medium-risk enhanced review | ~35% of applications, ~8% default rate |
| High-risk decline | ~25% of applications, avoids ~60% of portfolio losses |
