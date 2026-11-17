# Monitoring plan

Thresholds are conventions to be agreed with the model-risk owner.

| What to track | Frequency | Alert threshold | Owner | Action / retrain trigger |
|---|---|---|---|---|
| Score PSI (vs. training) | Monthly | <0.10 ok; 0.10-0.25 investigate; >0.25 escalate | Model owner | PSI > 0.25 two months running -> retrain review |
| Feature PSI (top 10 by SHAP) | Monthly | Same thresholds | Model owner | Check upstream data/pipeline first, then population change |
| Approval rate | Weekly | +/- 3 pts vs. 3-month average | Credit risk team | Check cut-off, channel mix, data feeds |
| Bad rate by origination cohort | Monthly (labels mature in 6-12 months) | Observed > predicted by 20% relative | Risk analytics | Re-calibrate; consider retrain |
| Calibration (predicted vs observed PD by decile) | Quarterly | ECE > 0.02 or Brier up > 5% relative | Model owner | Refit calibrator on recent matured cohorts |
| Ranking power (AUC / KS on matured cohorts) | Quarterly | AUC drop > 0.02 vs. development | Model validation | Retrain challenger, champion-challenger test |
| Fairness (approval & error rates by group) | Quarterly | Disparate-impact ratio < 0.8 or gap widening | Compliance + risk | Review features/proxies, document decision |
| Data quality (missing rates, sentinel counts) | Daily | Missing rate change > 5 pts | Data engineering | Block scoring or fall back to manual review |
