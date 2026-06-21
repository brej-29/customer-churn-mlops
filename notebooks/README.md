# Notebooks

## `churn_modeling_narrative.ipynb`

The end-to-end modeling narrative for the Tier 1 rebuild on the IBM Telco Customer Churn dataset.

### What it covers

| Section | Mode |
|---|---|
| 1. Problem & dataset framing | narrative |
| 2. EDA — churn rates, TotalCharges cleaning | live compute |
| 3. Preprocessing & feature engineering demo | live compute |
| 4. Model selection leaderboard | loads `reports/leaderboard.csv` |
| 5. Class imbalance experiment | loads `reports/imbalance_experiment.csv` |
| 6. Hyperparameter tuning (60-trial Optuna) | loads `reports/best_xgb_params.json` + plots |
| 7. Calibration & threshold selection | loads `reports/threshold.json` + plots |
| 8. Final test-set evaluation | loads `reports/final_test_metrics.json` |
| 9. SHAP explainability | loads `reports/shap_importance.csv` + plots |
| 10. Fairness analysis | loads `reports/fairness_*.csv` |
| 11. Conclusion, limitations & next steps | narrative |

### Running it

From the project root:

```bash
uv run jupyter notebook notebooks/churn_modeling_narrative.ipynb
```

Or re-execute it top-to-bottom (typically takes ~30 seconds):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    notebooks/churn_modeling_narrative.ipynb
```

### Notes

- **No heavy recomputation**: the notebook loads all expensive results from `reports/`.
  The only live computations are EDA aggregations and a preprocessing demo on a 50-row sample.
- The notebook must be run **after** `build_final_model()` has been called at least once
  (i.e., all files under `reports/` must exist).
- `churn_modeling_narrative.ipynb` is committed **with outputs** so reviewers see results
  without needing to execute.
