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

.main {
    background-color: #f5f7fa;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e6e6e6;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD FILES
# =========================================================

model = joblib.load("efrms_xgboost_model.pkl")

features = joblib.load("features.pkl")

threshold = joblib.load("threshold.pkl")

# =========================================================
# TITLE
# =========================================================

st.title("🏦 Enterprise Fraud Risk Monitoring System (EFRMS)")

st.markdown(
    "### AI-Powered Real-Time Fraud Surveillance Dashboard"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("💳 Real-Time Transaction Input")

# =========================================================
# FEATURE GROUPS
# =========================================================

time_features = [
    "hour",
    "weekday",
    "is_weekend",
    "is_night",
    "dayofyear"
]

velocity_features = [
    "transaction_velocity_7d",
    "seconds_since_last_txn"
]

amount_features = [
    "avg_amount_30d",
    "amount_deviation_ratio"
]

network_features = [
    "shared_device_count",
    "customer_merchant_txn_count",
    "merchant_ring_id"
]

# =========================================================
# DEFAULT VALUES
# =========================================================

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
# INPUT DICTIONARY
# =========================================================

input_dict = {}

# =========================================================
# TIME FEATURES
# =========================================================

st.sidebar.subheader("⏰ Time Features")

for feature in time_features:

    if feature in features:

        input_dict[feature] = st.sidebar.number_input(
            feature,
            value=float(default_values.get(feature, 0))
        )

# =========================================================
# VELOCITY FEATURES
# =========================================================

st.sidebar.subheader("⚡ Velocity Features")

for feature in velocity_features:

    if feature in features:

        input_dict[feature] = st.sidebar.number_input(
            feature,
            value=float(default_values.get(feature, 0))
        )

# =========================================================
# AMOUNT FEATURES
# =========================================================

st.sidebar.subheader("💰 Amount Features")

for feature in amount_features:

    if feature in features:

        input_dict[feature] = st.sidebar.number_input(
            feature,
            value=float(default_values.get(feature, 0))
        )

# =========================================================
# NETWORK FEATURES
# =========================================================

st.sidebar.subheader("🕸️ Network Features")

for feature in network_features:

    if feature in features:

        input_dict[feature] = st.sidebar.number_input(
            feature,
            value=float(default_values.get(feature, 0))
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

elif fraud_probability >= 0.50:

    risk_level = "MEDIUM RISK"

else:

    risk_level = "LOW RISK"

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

if input_dict.get("merchant_ring_id", 0) > 0:

    ews_alerts.append("⚠️ Merchant Ring Risk Detected")

# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

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
        "EWS Alerts",
        len(ews_alerts)
    )

with col4:

    st.metric(
        "Model Threshold",
        f"{threshold:.2f}"
    )

# =========================================================
# FRAUD ALERT ENGINE
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

    st.success(
        "✅ No RBI EWS alerts triggered"
    )

# =========================================================
# FRAUD RISK GAUGE
# =========================================================

st.subheader("🎯 Fraud Risk Gauge")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=fraud_probability * 100,
    title={'text': "Fraud Risk Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 40], 'color': "green"},
            {'range': [40, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}
        ],
        'bar': {'color': "darkred"}
    }
))

st.plotly_chart(fig_gauge, width='stretch')

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
# FRAUD DISTRIBUTION
# =========================================================

st.subheader("📊 Fraud Distribution")

fraud_dist = pd.DataFrame({
    "Category": ["Legitimate", "Fraud"],
    "Count": [95, 5]
})

fig_pie = px.pie(
    fraud_dist,
    names="Category",
    values="Count",
    title="Fraud vs Legitimate Transactions"
)

st.plotly_chart(fig_pie, width='stretch')

# =========================================================
# FRAUD ACTIVITY BY HOUR
# =========================================================

st.subheader("⏰ Fraud Activity by Hour")

hour_df = pd.DataFrame({
    "Hour": list(range(24)),
    "Fraud_Count": np.random.randint(1, 20, 24)
})

fig_hour = px.line(
    hour_df,
    x="Hour",
    y="Fraud_Count",
    markers=True,
    title="Hourly Fraud Trend"
)

st.plotly_chart(fig_hour, width='stretch')

# =========================================================
# RBI EWS DISTRIBUTION
# =========================================================

st.subheader("🏦 RBI EWS Trigger Distribution")

ews_df = pd.DataFrame({
    "EWS_Type": [
        "High Velocity",
        "Shared Device",
        "Night Transaction",
        "Rapid Transactions",
        "Merchant Ring"
    ],
    "Count": [12, 5, 8, 6, 4]
})

fig_ews = px.bar(
    ews_df,
    x="EWS_Type",
    y="Count",
    title="RBI EWS Alerts"
)

st.plotly_chart(fig_ews, width='stretch')

# =========================================================
# LIVE ALERT TABLE
# =========================================================

st.subheader("🚨 Live Fraud Alerts")

alerts_df = pd.DataFrame({
    "Alert_ID": ["ALT1001", "ALT1002", "ALT1003"],
    "Risk_Level": ["HIGH", "MEDIUM", "HIGH"],
    "Fraud_Probability": [0.92, 0.71, 0.88],
    "Status": ["OPEN", "UNDER REVIEW", "ESCALATED"]
})

st.dataframe(alerts_df, width='stretch')

# =========================================================
# LIVE MONITOR
# =========================================================

st.subheader("🛰️ Live Alert Monitor")

monitor_df = pd.DataFrame({
    "Timestamp": [datetime.now()],
    "Fraud Probability": [round(fraud_probability, 4)],
    "Risk Level": [risk_level],
    "Alert Status": [
        "ALERT GENERATED" if prediction == 1 else "NORMAL"
    ]
})

st.dataframe(monitor_df, width='stretch')

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "🏦 Enterprise Fraud Risk Monitoring System | "
    "AI + RBI EWS + Real-Time Fraud Analytics"
)
