import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
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
# LOAD MODEL FILES
# =========================================================

model = joblib.load("efrms_xgboost_model.pkl")
features = joblib.load("features.pkl")
threshold = joblib.load("threshold.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("🏦 Enterprise Fraud Risk Monitoring System (EFRMS)")
st.markdown("### AI-Powered Real-Time Fraud Surveillance Dashboard")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("💳 Real-Time Transaction Input")

input_dict = {}

# ---------------------------------------------------------
# SMART DEFAULT VALUES
# ---------------------------------------------------------

default_values = {
    "hour": 12,
    "day": 15,
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
# CREATE INPUTS DYNAMICALLY
# =========================================================

for feature in features:

    default_val = default_values.get(feature, 0.0)

    input_dict[feature] = st.sidebar.number_input(
        label=feature,
        value=float(default_val)
    )

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame([input_dict])

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
# RISK CLASSIFICATION
# =========================================================

if fraud_probability >= 0.80:
    risk_level = "HIGH RISK"
    risk_color = "red"

elif fraud_probability >= 0.50:
    risk_level = "MEDIUM RISK"
    risk_color = "orange"

else:
    risk_level = "LOW RISK"
    risk_color = "green"

# =========================================================
# RBI EWS RULES
# =========================================================

ews_alerts = []

if input_dict.get("transaction_velocity_7d", 0) > 20:
    ews_alerts.append("⚠️ High Transaction Velocity")

if input_dict.get("amount_deviation_ratio", 0) > 2:
    ews_alerts.append("⚠️ Abnormal Transaction Amount")

if input_dict.get("shared_device_count", 0) > 2:
    ews_alerts.append("⚠️ Shared Device Fraud Risk")

if input_dict.get("seconds_since_last_txn", 999999) < 60:
    ews_alerts.append("⚠️ Rapid Sequential Transactions")

if input_dict.get("is_night", 0) == 1:
    ews_alerts.append("⚠️ Night-Time Suspicious Activity")

if input_dict.get("customer_merchant_txn_count", 0) > 10:
    ews_alerts.append("⚠️ Repeated Merchant Transactions")

# =========================================================
# MAIN DASHBOARD
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

with col2:
    st.metric(
        "Risk Level",
        risk_level
    )

with col3:
    st.metric(
        "Model Threshold",
        f"{threshold:.2f}"
    )

# =========================================================
# FRAUD ALERT
# =========================================================

st.subheader("🚨 Real-Time Fraud Alert Engine")

if prediction == 1:

    st.error(
        f"⚠️ FRAUD ALERT GENERATED | Risk Level: {risk_level}"
    )

else:

    st.success(
        "✅ Transaction appears legitimate"
    )

# =========================================================
# RBI EWS ALERTS
# =========================================================

st.subheader("🏦 RBI Early Warning Signals (EWS)")

if len(ews_alerts) > 0:

    for alert in ews_alerts:
        st.warning(alert)

else:
    st.success("✅ No RBI EWS alerts triggered")

# =========================================================
# TRANSACTION SUMMARY
# =========================================================

st.subheader("📋 Transaction Summary")

summary_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Value": input_data.iloc[0].values
})

st.dataframe(summary_df, width='stretch')

# =========================================================
# RISK VISUALIZATION
# =========================================================

st.subheader("📊 Fraud Risk Visualization")

risk_df = pd.DataFrame({
    "Category": ["Legitimate Probability", "Fraud Probability"],
    "Value": [1 - fraud_probability, fraud_probability]
})

fig = px.bar(
    risk_df,
    x="Category",
    y="Value",
    title="Fraud Risk Distribution"
)

st.plotly_chart(fig, width='stretch')

# =========================================================
# LIVE ALERT MONITOR
# =========================================================

st.subheader("🛰️ Live Alert Monitor")

alert_data = {
    "Timestamp": [datetime.now()],
    "Fraud Probability": [round(fraud_probability, 4)],
    "Risk Level": [risk_level],
    "Alert Status": [
        "ALERT GENERATED" if prediction == 1 else "NORMAL"
    ]
}

alert_df = pd.DataFrame(alert_data)

st.dataframe(alert_df, width='stretch')

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "🏦 Enterprise Fraud Risk Monitoring System | "
    "AI + RBI EWS + Real-Time Fraud Analytics"
)
