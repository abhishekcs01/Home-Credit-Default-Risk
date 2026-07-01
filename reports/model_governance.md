# Model Governance — Home Credit Default Risk

Generated: 2026-06-12T12:54:19.447827+00:00

## Model Card Summary

| Property | Value |
| --- | --- |
| Model ID | home-credit-default-risk-v1.0.0 |
| Type | Binary classifier (probability of default) |
| Architecture | 5-fold LGB + CatBoost → stacking → isotonic calibration |
| Features | 10 |
| OOF ROC-AUC | 0.795 |
| OOF PR-AUC | N/A |
| Calibration | isotonic |

## Assumptions

1. Subsidiary tables reflect point-in-time history per Kaggle data contract
2. Feature engineering pipeline available at scoring time
3. Portfolio composition similar to 2016–2018 training era
4. Default definition (90+ DPD) unchanged

## Limitations

- Historical data may not reflect current macroeconomic conditions
- Sparse API payloads reduce accuracy — full feature vector required
- Target encoding unstable for rare categories with <80 observations
- No causal inference — associations only, not treatment effects
- Protected attributes excluded but proxies may remain

## Fairness Considerations

- Monitor approval rates by region, gender (OHE), education level quarterly
- Disparate impact ratio target: ≥ 0.80 for protected class proxies
- SHAP reason codes reviewed for systematic bias in high-risk segment
- Document override process when policy exceptions applied

## Monitoring Strategy

| Signal | Frequency | Threshold | Action |
| --- | --- | --- | --- |
| Score PSI | Weekly | > 0.10 warn, > 0.25 alert | Investigate data pipeline |
| Top-10 feature PSI | Monthly | > 0.15 | Feature drift review |
| Realized default rate | Monthly | ±10% vs expected | Threshold recalibration |
| OOF vs production AUC | Quarterly | > 0.01 drop | Champion/challenger test |
| API latency p99 | Daily | > 2s | Scale inference workers |
| FP/FN rate | Monthly | +20% vs baseline | Error analysis refresh |

## Retraining Strategy

- **Scheduled:** Quarterly full retrain on rolling 24-month window
- **Triggered:** PSI > 0.25, AUC drop > 0.01, or regulatory requirement
- **Process:** Preprocess → train → leakage audit → error analysis → champion/challenger → deploy
- **Rollback:** Keep previous bundle for 30 days; one-click revert via artifact version

## Deployment Considerations

- Bundle: `artifacts/models/ensemble_model.pkl`
- API: FastAPI `/v1/predict` with batch support
- Docker image for staging/production parity
- Environment: `TRAINING_DEVICE=cpu` recommended for reproducible training
- Secrets: No credentials in bundle; data access via secure storage
