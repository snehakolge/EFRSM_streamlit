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
# SIDEBAR INPUTS
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
# INPUT COLLECTION
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
# PREDICTION ENGINE
# =========================================================

try:

    fraud_probability = model.predict_proba(input_data)[0][1]

    prediction = int(fraud_probability >= threshold)

except Exception as e:

    st.error(f"Prediction Error: {e}")

    st.stop()

# =========================================================
# RISK LEVELS
# =========================================================

if fraud_probability >= 0.80:

    risk_level = "HIGH RISK"

elif fraud_probability >= 0.50:

    risk_level = "MEDIUM RISK"

else:

    risk_level = "LOW RISK"

# =========================================================
# RBI EARLY WARNING SIGNALS
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

if input_dict.get("customer_merchant_txn_count", 0) > 10:

    ews_alerts.append("⚠️ Repeated Merchant Activity")

if input_dict.get("merchant_ring_id", 0) > 0:

    ews_alerts.append("⚠️ Suspicious Merchant Network")

# =========================================================
# EXECUTIVE KPI DASHBOARD
# =========================================================

st.subheader("📊 Executive Fraud Dashboard")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

with kpi2:

    st.metric(
        "Risk Level",
        risk_level
    )

with kpi3:

    st.metric(
        "EWS Alerts",
        len(ews_alerts)
    )

with kpi4:

    st.metric(
        "Threshold",
        f"{threshold:.2f}"
    )

# =========================================================
# ALERT ENGINE
# =========================================================

st.subheader("🚨 Real-Time Fraud Alert Engine")

if prediction == 1:

    st.error(
        f"⚠️ FRAUD ALERT GENERATED | {risk_level}"
    )

else:

    st.success(
        "✅ Transaction appears legitimate"
    )

# =========================================================
# RBI EWS ALERTS
# =========================================================

st.subheader("🏦 RBI Early Warning Signals")

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
# FRAUD TREND
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
# EWS DISTRIBUTION
# =========================================================

st.subheader("🏦 RBI EWS Trigger Distribution")

ews_df = pd.DataFrame({
    "EWS_Type": [
        "High Velocity",
        "Shared Device",
        "Night Transaction",
        "Rapid Transactions",
        "Merchant Network"
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
# CASE MANAGEMENT DASHBOARD
# =========================================================

st.subheader("🗂️ Fraud Case Management")

case_df = pd.DataFrame({
    "Case_ID": ["CASE1001", "CASE1002", "CASE1003"],
    "Risk_Level": ["HIGH", "MEDIUM", "HIGH"],
    "Status": ["OPEN", "UNDER REVIEW", "ESCALATED"],
    "Priority": ["CRITICAL", "MEDIUM", "HIGH"],
    "Assigned_To": [
        "Fraud Analyst",
        "Risk Team",
        "Investigation Unit"
    ]
})

st.dataframe(case_df, width='stretch')

# =========================================================
# LIVE FRAUD ALERTS
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
    "Fraud_Probability": [round(fraud_probability, 4)],
    "Risk_Level": [risk_level],
    "Alert_Status": [
        "ALERT GENERATED" if prediction == 1 else "NORMAL"
    ]
})

st.dataframe(monitor_df, width='stretch')

# =========================================================
# FILE UPLOAD SECTION
# =========================================================

st.subheader("📂 Batch Transaction Fraud Monitoring")

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Transactions")

    st.dataframe(batch_df.head(), width='stretch')

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
🏦 Enterprise Fraud Risk Monitoring System (EFRMS)

AI-Powered Fraud Detection • RBI Early Warning Signals •
Real-Time Alert Surveillance • Enterprise Fraud Analytics
""")
