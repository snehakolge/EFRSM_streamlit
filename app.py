import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Enterprise Fraud Risk Monitoring System",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main { background-color: #f5f7fa; }

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #d9d9d9;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL FILES
# =========================================================

model = joblib.load("efrms_xgboost_model.pkl")
features = joblib.load("features.pkl")
threshold = joblib.load("threshold.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("🏦 Enterprise Fraud Risk Monitoring System (EFRMS)")

st.markdown("""
### AI-Powered Real-Time Fraud Surveillance Dashboard
Real-Time Fraud Detection • RBI EWS Signals • Alert Monitoring • Fraud Analytics
""")

# =========================================================
# FEATURE GROUPS
# =========================================================

time_features = ["hour", "weekday", "is_weekend", "is_night", "dayofyear"]

velocity_features = ["transaction_velocity_7d", "seconds_since_last_txn"]

amount_features = ["avg_amount_30d", "amount_deviation_ratio"]

network_features = ["shared_device_count", "customer_merchant_txn_count", "merchant_ring_id"]

default_values = {
    "hour": 12,
    "weekday": 2,
    "is_weekend": 0,
    "is_night": 0,
    "dayofyear": 100,
    "transaction_velocity_7d": 10,
    "seconds_since_last_txn": 100000,
    "avg_amount_30d": 500,
    "amount_deviation_ratio": 1.0,
    "shared_device_count": 0,
    "customer_merchant_txn_count": 2,
    "merchant_ring_id": 0
}

# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.header("💳 Real-Time Transaction Input")

input_dict = {}

st.sidebar.subheader("⏰ Time Features")
for f in time_features:
    if f in features:
        input_dict[f] = st.sidebar.number_input(f, value=float(default_values[f]))

st.sidebar.subheader("⚡ Velocity Features")
for f in velocity_features:
    if f in features:
        input_dict[f] = st.sidebar.number_input(f, value=float(default_values[f]))

st.sidebar.subheader("💰 Amount Features")
for f in amount_features:
    if f in features:
        input_dict[f] = st.sidebar.number_input(f, value=float(default_values[f]))

st.sidebar.subheader("🕸️ Network Features")
for f in network_features:
    if f in features:
        input_dict[f] = st.sidebar.number_input(f, value=float(default_values[f]))

# =========================================================
# INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame([input_dict])

# align with training features
input_data = input_data.reindex(columns=features, fill_value=0)

# =========================================================
# PREDICTION
# =========================================================

try:
    fraud_probability = model.predict_proba(input_data)[0][1]
    prediction = int(fraud_probability >= threshold)
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# =========================================================
# RISK LEVEL
# =========================================================

if fraud_probability >= 0.80:
    risk_level = "HIGH RISK"
elif fraud_probability >= 0.50:
    risk_level = "MEDIUM RISK"
else:
    risk_level = "LOW RISK"

# =========================================================
# RBI EWS ALERTS
# =========================================================

ews_alerts = []

if input_dict.get("transaction_velocity_7d", 0) > 20:
    ews_alerts.append("⚠️ High Transaction Velocity")

if input_dict.get("amount_deviation_ratio", 0) > 2:
    ews_alerts.append("⚠️ Abnormal Transaction Amount")

if input_dict.get("shared_device_count", 0) > 2:
    ews_alerts.append("⚠️ Shared Device Risk")

if input_dict.get("seconds_since_last_txn", 999999) < 60:
    ews_alerts.append("⚠️ Rapid Sequential Transactions")

if input_dict.get("is_night", 0) == 1:
    ews_alerts.append("⚠️ Suspicious Night Transaction")

# =========================================================
# KPI DASHBOARD
# =========================================================

st.subheader("📊 Executive Dashboard")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Fraud Probability", f"{fraud_probability:.2%}")
c2.metric("Risk Level", risk_level)
c3.metric("EWS Alerts", len(ews_alerts))
c4.metric("Threshold", f"{threshold:.2f}")

# =========================================================
# ALERT ENGINE
# =========================================================

st.subheader("🚨 Fraud Alert Engine")

if prediction == 1:
    st.error(f"⚠️ FRAUD ALERT | {risk_level}")
else:
    st.success("✅ Transaction Legitimate")

# =========================================================
# EWS SECTION
# =========================================================

st.subheader("🏦 RBI Early Warning Signals")

if ews_alerts:
    for a in ews_alerts:
        st.warning(a)
else:
    st.success("No EWS alerts triggered")

# =========================================================
# FRAUD GAUGE
# =========================================================

st.subheader("🎯 Fraud Risk Gauge")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=fraud_probability * 100,
    title={"text": "Fraud Risk Score"},
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 40], "color": "green"},
            {"range": [40, 70], "color": "orange"},
            {"range": [70, 100], "color": "red"}
        ],
        "bar": {"color": "darkred"}
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)

# =========================================================
# SHAP EXPLAINABILITY (IMPORTANT PART)
# =========================================================

st.subheader("🧠 Why this transaction was flagged (SHAP Explainability)")

try:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Impact": shap_values[0]
    }).sort_values("Impact")

    st.write("### Feature Contribution")

    fig = go.Figure(go.Bar(
        x=shap_df["Impact"],
        y=shap_df["Feature"],
        orientation='h'
    ))

    st.plotly_chart(fig, use_container_width=True)

    top_feature = shap_df.iloc[-1]

    st.info(
        f"Most influential feature: **{top_feature['Feature']}** "
        f"with impact score **{top_feature['Impact']:.4f}**"
    )

except Exception as e:
    st.warning("SHAP explanation not available")
    st.write(e)

# =========================================================
# TRANSACTION SUMMARY
# =========================================================

st.subheader("📋 Transaction Summary")
st.dataframe(input_data, use_container_width=True)

# =========================================================
# STATIC VISUALS (YOUR EXISTING DASHBOARD)
# =========================================================

st.subheader("📊 Fraud Distribution")

fig_pie = px.pie(
    names=["Legit", "Fraud"],
    values=[95, 5],
    title="Fraud vs Legitimate"
)

st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("⏰ Fraud Trend Simulation")

hour_df = pd.DataFrame({
    "Hour": list(range(24)),
    "Fraud_Count": np.random.randint(1, 20, 24)
})

fig_line = px.line(hour_df, x="Hour", y="Fraud_Count", markers=True)
st.plotly_chart(fig_line, use_container_width=True)

# =========================================================
# CASE MANAGEMENT
# =========================================================

st.subheader("🗂️ Case Management")

case_df = pd.DataFrame({
    "Case_ID": ["CASE1001", "CASE1002", "CASE1003"],
    "Risk_Level": ["HIGH", "MEDIUM", "HIGH"],
    "Status": ["OPEN", "UNDER REVIEW", "ESCALATED"]
})

st.dataframe(case_df, use_container_width=True)

# =========================================================
# LIVE MONITOR
# =========================================================

st.subheader("🛰️ Live Monitor")

monitor_df = pd.DataFrame({
    "Timestamp": [datetime.now()],
    "Fraud_Probability": [fraud_probability],
    "Risk_Level": [risk_level]
})

st.dataframe(monitor_df, use_container_width=True)

# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📂 Batch Monitoring")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head(), use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("🏦 EFRMS | AI Fraud Detection | RBI Compliant Risk Analytics")
