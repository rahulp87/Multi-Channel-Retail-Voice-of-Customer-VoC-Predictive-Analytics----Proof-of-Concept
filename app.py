pip install pandas numpy scikit-learn xgboost shap streamlit plotly matplotlib
"""
Multi-Channel Retail Voice-of-Customer (VoC) Predictive Analytics -- Proof of Concept
====================================================================================

Single-file Streamlit application. Everything (data, model, explainability, UI) is
self-contained and regenerated deterministically on first run.

    * Synthetic multi-channel retail CX dataset (~1,500 customers)
    * Gradient-boosted 90-day churn classifier (XGBoost, sklearn fallback)
    * SHAP (TreeExplainer) global + local explainability
    * Key-driver analysis: Shapley attribution triangulated against
      standardised logistic-regression coefficients
    * Intervention Priority Index (IPI) for service-recovery targeting
    * Executive dashboard: KPI cards + three analytical tabs

Dependencies (Python 3.10+)
---------------------------
    pandas>=2.0
    numpy>=1.24
    scikit-learn>=1.3
    xgboost>=1.7          # optional; auto-falls back to sklearn GradientBoostingClassifier
    shap>=0.44
    streamlit>=1.30
    plotly>=5.18
    matplotlib>=3.7       # transitively required by shap; used for the SHAP beeswarm

Install & run
-------------
    pip install pandas numpy scikit-learn xgboost shap streamlit plotly matplotlib
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - environment dependent
    from sklearn.ensemble import GradientBoostingClassifier
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_CUSTOMERS = 1500
CHANNELS = ["In-Store", "Online Grocery", "Online Clothing", "Contact Centre"]

FEATURES_NUMERIC = [
    "csat",
    "nps",
    "sentiment_score",
    "recency_days",
    "frequency_90d",
    "aov",
    "delivery_delay_flag",
    "fcr_flag",
    "return_flag",
]
CHANNEL_COLS = [f"Channel_{c}" for c in CHANNELS]
MODEL_FEATURES = FEATURES_NUMERIC + CHANNEL_COLS

FEATURE_LABELS = {
    "csat": "CSAT Score",
    "nps": "NPS / Advocacy",
    "sentiment_score": "Verbatim Sentiment",
    "recency_days": "Recency (days)",
    "frequency_90d": "Frequency (90d)",
    "aov": "Avg Order Value",
    "delivery_delay_flag": "Delivery Delay",
    "fcr_flag": "First-Contact Resolution",
    "return_flag": "Product Return",
    "Channel_In-Store": "Channel: In-Store",
    "Channel_Online Grocery": "Channel: Online Grocery",
    "Channel_Online Clothing": "Channel: Online Clothing",
    "Channel_Contact Centre": "Channel: Contact Centre",
}

# Drivers where a higher raw value means a WORSE customer experience.
PERF_INVERTED = {"recency_days", "delivery_delay_flag", "return_flag"}


# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(n: int = N_CUSTOMERS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Deterministic synthetic multi-channel retail CX panel with a logit-driven target."""
    rng = np.random.default_rng(seed)

    channel = rng.choice(CHANNELS, size=n, p=[0.34, 0.24, 0.24, 0.18])

    # A single latent satisfaction factor drives CSAT, NPS and verbatim sentiment jointly,
    # so the three survey signals are correlated the way they are in real VoC data.
    latent_sat = rng.normal(0.0, 1.0, n)
    csat = np.clip(np.round(6.6 + 1.7 * latent_sat + rng.normal(0, 0.9, n)), 1, 10).astype(int)
    nps = np.clip(np.round(6.2 + 1.9 * latent_sat + rng.normal(0, 1.1, n)), 1, 10).astype(int)
    sentiment = np.clip(0.15 + 0.42 * latent_sat + rng.normal(0, 0.25, n), -1.0, 1.0)

    # RFM block.
    recency = np.clip(rng.exponential(38.0, n) + rng.normal(0, 6, n), 1, 365).astype(int)
    frequency = np.clip(
        rng.poisson(4.2, n) + (channel == "Online Grocery") * rng.poisson(2.0, n), 0, 40
    ).astype(int)
    aov_base = {"In-Store": 58.0, "Online Grocery": 82.0, "Online Clothing": 96.0, "Contact Centre": 64.0}
    aov = np.array([rng.lognormal(np.log(aov_base[c]), 0.45) for c in channel])
    aov = np.clip(aov, 8.0, 600.0).round(2)

    # Operational friction signals.
    is_online = np.isin(channel, ["Online Grocery", "Online Clothing"]).astype(float)
    delivery_delay = rng.binomial(1, np.clip(0.08 + 0.17 * is_online + 0.10 * (recency > 60), 0, 1))

    contacted = (channel == "Contact Centre") | (rng.random(n) < 0.22)
    fcr = np.where(contacted, rng.binomial(1, 0.68, n), 1)  # 1 == resolved on first contact

    return_flag = rng.binomial(
        1, np.clip(0.06 + 0.14 * (channel == "Online Clothing") + 0.05 * (sentiment < -0.1), 0, 1)
    )

    def z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-9)

    # Deterministic logit with controlled irreducible noise -> real learnable signal.
    logit = (
        -1.05
        + 0.95 * z(recency)
        - 0.70 * z(frequency)
        - 1.05 * z(csat)
        - 1.10 * sentiment
        - 0.30 * z(nps)
        - 0.25 * z(aov)
        + 1.10 * delivery_delay
        - 0.80 * fcr
        + 0.75 * return_flag
        + 0.45 * (channel == "Contact Centre").astype(float)
        + rng.normal(0.0, 0.55, n)
    )
    churn_prob_true = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob_true)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "channel": channel,
            "csat": csat,
            "nps": nps,
            "sentiment_score": sentiment.round(3),
            "recency_days": recency,
            "frequency_90d": frequency,
            "aov": aov,
            "delivery_delay_flag": delivery_delay.astype(int),
            "fcr_flag": fcr.astype(int),
            "return_flag": return_flag.astype(int),
            "churn_90d": churn.astype(int),
        }
    )

    # Customer Lifetime Value = annualised spend x gross margin x planning horizon,
    # floored at a single order so zero-frequency customers keep a positive CLV.
    gross_margin, horizon_years = 0.30, 3.0
    clv = df["aov"] * df["frequency_90d"] * 4.0 * gross_margin * horizon_years
    df["clv"] = np.maximum(clv, df["aov"]).round(2)
    return df


# ---------------------------------------------------------------------------
# 2. Model + explainability build (single cached pipeline)
# ---------------------------------------------------------------------------
def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[FEATURES_NUMERIC + ["channel"]], columns=["channel"], prefix="Channel")
    return X.reindex(columns=MODEL_FEATURES, fill_value=0).astype(float)


def _extract_shap(explainer, X: pd.DataFrame):
    """Return (shap_matrix, base_value) as positive-class log-odds contributions."""
    try:
        expl = explainer(X, check_additivity=False)
        vals = np.asarray(expl.values, dtype=float)
        base = np.asarray(expl.base_values, dtype=float)
        if vals.ndim == 3:  # (rows, features, classes)
            vals = vals[:, :, -1]
            base = base[:, -1] if base.ndim == 2 else base
    except Exception:
        raw = explainer.shap_values(X)
        vals = np.asarray(raw[-1] if isinstance(raw, list) else raw, dtype=float)
        ev = explainer.expected_value
        base = np.asarray(ev[-1] if np.ndim(ev) > 0 else ev, dtype=float)
    return vals, float(np.mean(base))


@st.cache_resource(show_spinner="Generating data, training model, computing SHAP...")
def build_pipeline():
    df = generate_dataset()
    X = _design_matrix(df)
    y = df["churn_90d"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    if _HAS_XGB:
        model = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_STATE,
        )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    proba_all = model.predict_proba(X)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "algo": "XGBoost" if _HAS_XGB else "GradientBoosting (sklearn)",
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, pred_test, zero_division=0)),
        "base_rate": float(np.mean(y)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    curves = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}

    # Standardised logistic regression for key-driver coefficient triangulation.
    scaler = StandardScaler()
    logit = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logit.fit(scaler.fit_transform(X), y)
    coef = pd.Series(logit.coef_[0], index=X.columns, name="logit_coef")

    # SHAP (TreeExplainer) over the full population.
    explainer = shap.TreeExplainer(model)
    shap_vals, shap_base = _extract_shap(explainer, X)
    shap_df = pd.DataFrame(shap_vals, columns=X.columns, index=X.index)

    return {
        "df": df,
        "X": X,
        "y": y,
        "model": model,
        "proba_all": proba_all,
        "metrics": metrics,
        "curves": curves,
        "coef": coef,
        "shap_df": shap_df,
        "shap_base": shap_base,
    }


# ---------------------------------------------------------------------------
# 3. Scoring, prioritisation, recommendations
# ---------------------------------------------------------------------------
def add_scores(df: pd.DataFrame, proba: np.ndarray, high_thr: float) -> pd.DataFrame:
    out = df.copy()
    out["churn_prob"] = np.round(proba, 4)
    # Intervention Priority Index.
    out["ipi"] = (
        out["churn_prob"] * (out["clv"] / out["aov"]) * (1.0 - out["csat"] / 10.0)
    ).round(3)
    out["risk_tier"] = np.where(
        out["churn_prob"] >= high_thr,
        "High",
        np.where(out["churn_prob"] >= high_thr / 2.0, "Medium", "Low"),
    )
    out["friction_signals"] = (
        out["delivery_delay_flag"] + out["return_flag"] + (1 - out["fcr_flag"])
    ).astype(int)
    out["preventable_churn"] = (
        (out["churn_prob"] >= high_thr) & (out["friction_signals"] >= 1)
    ).astype(int)
    out["expected_clv_loss"] = (out["churn_prob"] * out["clv"]).round(0)
    return out


def primary_push_driver(shap_row: pd.Series) -> str:
    """Feature contributing the largest positive (churn-increasing) SHAP value for a customer."""
    feat = shap_row.sort_values(ascending=False).index[0]
    return FEATURE_LABELS.get(feat, feat)


def recommend_action(row: pd.Series) -> str:
    if row["delivery_delay_flag"] == 1:
        return "Expedite logistics remediation + proactive delivery credit"
    if row["fcr_flag"] == 0:
        return "Escalate to senior resolution specialist; close the loop <24h"
    if row["return_flag"] == 1:
        return "Product-fit outreach + curated replacement offer"
    if row["sentiment_score"] < -0.15:
        return "Assign relationship manager; VoC deep-dive call"
    if row["csat"] <= 5:
        return "Service-recovery call + goodwill gesture"
    return "Targeted loyalty save-offer aligned to channel"


def performance_score(series: pd.Series, col: str) -> float:
    """0-100 score for how favourably the base performs on a driver (higher = better CX)."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return 50.0
    norm = (s.mean() - lo) / (hi - lo) * 100.0
    return 100.0 - norm if col in PERF_INVERTED else norm


# ---------------------------------------------------------------------------
# 4. Streamlit application
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Retail VoC Churn Intelligence", layout="wide", page_icon="📊")

P = build_pipeline()
df_raw, X = P["df"], P["X"]
model, proba_all = P["model"], P["proba_all"]
metrics, curves, coef = P["metrics"], P["curves"], P["coef"]
shap_df, shap_base = P["shap_df"], P["shap_base"]

st.sidebar.header("Controls")
high_thr = st.sidebar.slider("High-risk probability threshold", 0.30, 0.90, 0.55, 0.05)
st.sidebar.markdown(
    f"**Model:** {metrics['algo']}  \n"
    f"**Train / Test:** {metrics['n_train']} / {metrics['n_test']}  \n"
    f"**Observed churn base rate:** {metrics['base_rate']:.1%}  \n"
    f"**ROC-AUC:** {metrics['roc_auc']:.3f} · **PR-AUC:** {metrics['pr_auc']:.3f}"
)

scored = add_scores(df_raw, proba_all, high_thr)

mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
impact_index = (100.0 * mean_abs_shap / mean_abs_shap.max()).round(1)
mean_signed_shap = shap_df.mean()
top_driver_feat = mean_abs_shap.index[0]
top_driver_label = FEATURE_LABELS.get(top_driver_feat, top_driver_feat)

high_risk = scored[scored["risk_tier"] == "High"]
revenue_at_stake = float((high_risk["churn_prob"] * high_risk["clv"]).sum())

# ---- Header + Executive KPI cards ----------------------------------------
st.title("Multi-Channel Retail — Voice-of-Customer Churn Intelligence")
st.caption(
    "Proof of Concept · synthetic data · gradient-boosted 90-day churn model with SHAP explainability "
    "and an Intervention Priority Index for service recovery."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "High-Risk Revenue at Stake",
    f"${revenue_at_stake:,.0f}",
    help="Σ (churn probability × Customer Lifetime Value) across High-risk customers.",
)
k2.metric(
    "Top Driver of Churn",
    top_driver_label,
    f"{impact_index.iloc[0]:.0f}/100 SHAP impact",
    delta_color="off",
)
k3.metric(
    "Model Performance (ROC-AUC)",
    f"{metrics['roc_auc']:.3f}",
    f"PR-AUC {metrics['pr_auc']:.3f} · Recall {metrics['recall']:.2f}",
    delta_color="off",
)
k4.metric(
    "High-Risk Customers",
    f"{len(high_risk):,}",
    f"{len(high_risk) / len(scored):.1%} of base",
    delta_color="off",
)

tab1, tab2, tab3 = st.tabs(
    [
        "1 · Risk Segmentation & Prediction",
        "2 · Explainable Driver Analysis",
        "3 · Service-Recovery Priority Queue",
    ]
)

# =======================================================================
# TAB 1 — Risk Segmentation & Churn Prediction
# =======================================================================
with tab1:
    st.subheader("Risk Segmentation & Churn Prediction")

    f1, f2, f3 = st.columns([2, 2, 1.6])
    ch_sel = f1.multiselect("Channel", CHANNELS, default=CHANNELS)
    tier_sel = f2.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    min_p = f3.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)

    view = scored[
        scored["channel"].isin(ch_sel)
        & scored["risk_tier"].isin(tier_sel)
        & (scored["churn_prob"] >= min_p)
    ].copy()

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Customers in view", f"{len(view):,}")
    v2.metric("Mean churn probability", f"{view['churn_prob'].mean():.1%}" if len(view) else "—")
    v3.metric("CLV in view", f"${view['clv'].sum():,.0f}")
    v4.metric("Expected CLV loss", f"${view['expected_clv_loss'].sum():,.0f}")

    table_cols = [
        "customer_id", "channel", "risk_tier", "churn_prob", "ipi",
        "csat", "nps", "sentiment_score", "recency_days", "frequency_90d",
        "aov", "clv", "delivery_delay_flag", "fcr_flag", "return_flag", "churn_90d",
    ]
    st.dataframe(
        view[table_cols].sort_values("churn_prob", ascending=False),
        use_container_width=True,
        height=380,
        hide_index=True,
        column_config={
            "churn_prob": st.column_config.ProgressColumn(
                "Churn P", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "ipi": st.column_config.NumberColumn("IPI", format="%.2f"),
            "clv": st.column_config.NumberColumn("CLV", format="$%d"),
            "aov": st.column_config.NumberColumn("AOV", format="$%.0f"),
            "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.2f"),
        },
    )

    st.markdown("#### Individual explanation — SHAP contribution waterfall")
    pool = view["customer_id"].tolist() or scored["customer_id"].tolist()
    cust = st.selectbox("Select a customer", pool)
    pos = scored.index[scored["customer_id"] == cust][0]

    row_shap = shap_df.loc[pos].sort_values(key=np.abs, ascending=False)
    labels = [FEATURE_LABELS.get(f, f) for f in row_shap.index]
    total_logodds = float(shap_base + row_shap.sum())

    wf = go.Figure(
        go.Waterfall(
            orientation="h",
            y=["Population base"] + labels + ["This customer"],
            x=[shap_base] + list(row_shap.values) + [total_logodds],
            measure=["absolute"] + ["relative"] * len(row_shap) + ["total"],
            connector={"line": {"color": "#9aa0a6"}},
            increasing={"marker": {"color": "#c62828"}},   # pushes churn risk up
            decreasing={"marker": {"color": "#2e7d32"}},   # pulls churn risk down
            totals={"marker": {"color": "#1f4e79"}},
        )
    )
    wf.update_layout(
        title=(
            f"{cust} — base log-odds {shap_base:.2f} → model log-odds {total_logodds:.2f} "
            f"(predicted churn {scored.loc[pos, 'churn_prob']:.1%})"
        ),
        xaxis_title="SHAP value (contribution to churn log-odds)",
        height=460,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    st.plotly_chart(wf, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.metric("Predicted churn probability", f"{scored.loc[pos, 'churn_prob']:.1%}")
    d2.metric("Intervention Priority Index", f"{scored.loc[pos, 'ipi']:.2f}")
    d3.metric("Actual 90-day churn (label)", "Yes" if scored.loc[pos, "churn_90d"] == 1 else "No")

# =======================================================================
# TAB 2 — Explainable Driver Analysis
# =======================================================================
with tab2:
    st.subheader("Explainable Driver Analysis")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Global SHAP summary (beeswarm)")
        disp = X.rename(columns=FEATURE_LABELS)
        plt.figure(figsize=(8, 5.6))
        shap.summary_plot(shap_df.values, disp, show=False, max_display=13)
        st.pyplot(plt.gcf())
        plt.close("all")
    with right:
        st.markdown("#### Mean |SHAP| — global feature importance")
        imp_tbl = mean_abs_shap.rename(index=FEATURE_LABELS).reset_index()
        imp_tbl.columns = ["Driver", "Mean |SHAP|"]
        fig_imp = px.bar(imp_tbl.iloc[::-1], x="Mean |SHAP|", y="Driver", orientation="h")
        fig_imp.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Key-Driver Impact vs. Performance matrix")
    kd_feats = [f for f in X.columns if not f.startswith("Channel_")]
    matrix = pd.DataFrame(
        {
            "Driver": [FEATURE_LABELS.get(f, f) for f in kd_feats],
            "feature": kd_feats,
            "Impact": [float(impact_index.get(f, 0.0)) for f in kd_feats],
            "Performance": [round(performance_score(X[f], f), 1) for f in kd_feats],
            "Logit coef": [round(float(coef.get(f, 0.0)), 3) for f in kd_feats],
        }
    )
    matrix["coef_mag"] = matrix["Logit coef"].abs() + 0.05
    imp_mid = float(matrix["Impact"].median())
    perf_mid = float(matrix["Performance"].median())

    def _quadrant(r: pd.Series) -> str:
        hi_imp, hi_perf = r["Impact"] >= imp_mid, r["Performance"] >= perf_mid
        if hi_imp and not hi_perf:
            return "Fix & Prioritise"
        if hi_imp and hi_perf:
            return "Maintain & Promote"
        if not hi_imp and not hi_perf:
            return "Monitor"
        return "Low-leverage strength"

    matrix["Quadrant"] = matrix.apply(_quadrant, axis=1)

    fig_m = px.scatter(
        matrix,
        x="Performance",
        y="Impact",
        text="Driver",
        color="Quadrant",
        size="coef_mag",
        size_max=26,
        hover_data={"Logit coef": True, "coef_mag": False, "Driver": False},
        color_discrete_map={
            "Fix & Prioritise": "#c62828",
            "Maintain & Promote": "#2e7d32",
            "Monitor": "#607d8b",
            "Low-leverage strength": "#f9a825",
        },
    )
    fig_m.add_vline(x=perf_mid, line_dash="dot", line_color="#9aa0a6")
    fig_m.add_hline(y=imp_mid, line_dash="dot", line_color="#9aa0a6")
    fig_m.update_traces(textposition="top center")
    fig_m.update_layout(
        height=540,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Performance score (higher = better current CX)",
        yaxis_title="Impact on churn (mean |SHAP|, indexed 0-100)",
    )
    st.plotly_chart(fig_m, use_container_width=True)
    st.caption(
        "Upper-left **Fix & Prioritise** drivers combine high churn impact with weak current "
        "performance — the highest-return targets for CX investment."
    )

    st.markdown("#### Shapley attribution vs. standardised regression coefficients")
    driver_tbl = pd.DataFrame(
        {
            "Driver": [FEATURE_LABELS.get(f, f) for f in X.columns],
            "Impact index (0-100)": [float(impact_index.get(f, 0.0)) for f in X.columns],
            "Mean signed SHAP": mean_signed_shap.reindex(X.columns).round(3).to_numpy(),
            "Logit coefficient (std.)": coef.reindex(X.columns).round(3).to_numpy(),
        }
    )
    driver_tbl["Effect on churn"] = np.where(
        driver_tbl["Mean signed SHAP"] >= 0, "▲ increases risk", "▼ decreases risk"
    )
    driver_tbl = driver_tbl.sort_values("Impact index (0-100)", ascending=False)
    st.dataframe(driver_tbl, use_container_width=True, hide_index=True)

    with st.expander("Model diagnostics — ROC & Precision-Recall curves"):
        g1, g2 = st.columns(2)
        roc_fig = go.Figure()
        roc_fig.add_scatter(x=curves["fpr"], y=curves["tpr"], mode="lines", name="Model")
        roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dot"))
        roc_fig.update_layout(
            title=f"ROC curve — AUC {metrics['roc_auc']:.3f}",
            xaxis_title="False positive rate", yaxis_title="True positive rate",
            height=360, margin=dict(l=10, r=10, t=45, b=10),
        )
        g1.plotly_chart(roc_fig, use_container_width=True)

        pr_fig = go.Figure()
        pr_fig.add_scatter(x=curves["recall"], y=curves["precision"], mode="lines", name="Model")
        pr_fig.add_hline(y=metrics["base_rate"], line_dash="dot", line_color="#9aa0a6")
        pr_fig.update_layout(
            title=f"Precision-Recall — AP {metrics['pr_auc']:.3f}",
            xaxis_title="Recall", yaxis_title="Precision",
            height=360, margin=dict(l=10, r=10, t=45, b=10),
        )
        g2.plotly_chart(pr_fig, use_container_width=True)

# =======================================================================
# TAB 3 — Actionable Service Recovery / Priority Queue
# =======================================================================
with tab3:
    st.subheader("Actionable Service Recovery — Intervention Priority Queue")
    st.caption(
        "IPI = churn probability × (CLV ÷ AOV) × (1 − CSAT/10).  "
        "Queue = High-risk customers carrying ≥ 1 operational friction signal (delivery delay, "
        "failed first-contact resolution, or product return) — i.e. preventable churn."
    )

    q = scored[scored["preventable_churn"] == 1].copy()
    if q.empty:
        q = scored[scored["risk_tier"] == "High"].copy()

    q["primary_driver"] = [primary_push_driver(shap_df.loc[i]) for i in q.index]
    q["recommended_action"] = q.apply(recommend_action, axis=1)
    queue = q.sort_values("ipi", ascending=False).head(50)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers in queue", f"{len(queue):,}")
    m2.metric("CLV protected if fully saved", f"${queue['clv'].sum():,.0f}")
    m3.metric("Expected CLV loss (no action)", f"${queue['expected_clv_loss'].sum():,.0f}")
    m4.metric("Mean churn probability", f"{queue['churn_prob'].mean():.1%}")

    queue_cols = [
        "customer_id", "channel", "ipi", "churn_prob", "clv", "aov", "csat",
        "primary_driver", "friction_signals", "recommended_action",
    ]
    st.dataframe(
        queue[queue_cols],
        use_container_width=True,
        height=430,
        hide_index=True,
        column_config={
            "ipi": st.column_config.NumberColumn("IPI", format="%.2f"),
            "churn_prob": st.column_config.ProgressColumn(
                "Churn P", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "clv": st.column_config.NumberColumn("CLV", format="$%d"),
            "aov": st.column_config.NumberColumn("AOV", format="$%.0f"),
            "friction_signals": st.column_config.NumberColumn("Friction #"),
        },
    )

    fig_q = px.bar(
        queue.head(15).iloc[::-1],
        x="ipi",
        y="customer_id",
        orientation="h",
        color="channel",
        hover_data=["recommended_action", "primary_driver", "clv", "churn_prob"],
    )
    fig_q.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Intervention Priority Index",
        yaxis_title="",
    )
    st.plotly_chart(fig_q, use_container_width=True)

    st.download_button(
        "Download priority queue (CSV)",
        queue[queue_cols + ["recency_days", "frequency_90d", "expected_clv_loss", "churn_90d"]]
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="voc_intervention_priority_queue.csv",
        mime="text/csv",
    )

st.divider()
with st.expander("Methodology & assumptions"):
    st.markdown(
        """
- **Data**: 1,500 synthetic customers across four retail journeys. CSAT, NPS and verbatim
  sentiment share a latent satisfaction factor; RFM and friction flags are channel-conditioned.
- **Target**: 90-day churn drawn from a deterministic logit of the features plus calibrated
  Gaussian noise, so the model recovers genuine, bounded signal (no leakage, no separability).
- **Model**: `XGBClassifier` (falls back to sklearn `GradientBoostingClassifier`), 25% stratified
  hold-out for ROC-AUC / PR-AUC / precision / recall.
- **Explainability**: `shap.TreeExplainer` over the full population; contributions are in
  positive-class log-odds units and are additive to the model margin (base + Σ SHAP).
- **Key-driver triangulation**: mean |SHAP| (impact) is cross-checked against standardised
  logistic-regression coefficients; the Impact-vs-Performance matrix flags where weak current
  CX performance coincides with high churn leverage.
- **Prioritisation**: `IPI = P(churn) × (CLV / AOV) × (1 − CSAT/10)`. CLV = annualised spend
  × 30% gross margin × 3-year horizon. The recovery queue keeps the top 50 preventable-churn
  customers (High risk **and** ≥ 1 operational friction signal) ranked by IPI.
        """
    )
