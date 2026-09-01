# Retail VoC Churn Intelligence — PoC

Self-contained Streamlit proof of concept for a multi-channel retail Voice-of-Customer
predictive analytics system:

- ~1,500-row synthetic multi-channel retail CX dataset (deterministic)
- Gradient-boosted 90-day churn classifier (XGBoost, sklearn fallback)
- SHAP (`TreeExplainer`) global + local explainability
- Key-driver analysis: Shapley attribution vs. standardised logistic-regression coefficients
- Intervention Priority Index (IPI) for service-recovery targeting
- Executive dashboard: KPI cards + three analytical tabs

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deployable as-is to [Streamlit Community Cloud](https://share.streamlit.io):
point it at this repo, set the main file to `app.py`.
