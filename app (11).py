
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Enterprise Fraud Risk Monitoring System",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🏦 Enterprise Fraud Risk Monitoring System (EFRMS)")

st.markdown("""
AI-Powered Real-Time Fraud Surveillance Dashboard  
RBI-Aligned Early Warning Signal Monitoring
""")

# =====================================================
# LOAD MODEL FILES
# =====================================================

model = joblib.load("efrms_xgboost_model.pkl")

features = joblib.load("features.pkl")

threshold = joblib.load("threshold.pkl")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("💳 Real-Time Transaction Input")

hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)

weekday = st.sidebar.slider("Weekday", 0, 6, 2)

is_weekend = st.sidebar.selectbox(
    "Weekend Transaction",
    [0,1]
)

is_night = st.sidebar.selectbox(
    "Night Transaction",
    [0,1]
)

dayofyear = st.sidebar.slider(
    "Day of Year",
    1,
    365,
    120
)

transaction_velocity_7d = st.sidebar.slider(
    "Transaction Velocity 7D",
    1,
    50,
    10
)

seconds_since_last_txn = st.sidebar.number_input(
    "Seconds Since Last Transaction",
    value=1000
)

avg_amount_30d = st.sidebar.number_input(
    "Average Amount 30D",
    value=500.0
)

amount_deviation_ratio = st.sidebar.slider(
    "Amount Deviation Ratio",
    0.0,
    5.0,
    1.0
)

shared_device_count = st.sidebar.slider(
    "Shared Device Count",
    0,
    10,
    0
)

customer_merchant_txn_count = st.sidebar.slider(
    "Customer Merchant Transaction Count",
    0,
    20,
    2
)

# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({
    "hour":[hour],
    "weekday":[weekday],
    "is_weekend":[is_weekend],
    "is_night":[is_night],
    "dayofyear":[dayofyear],
    "transaction_velocity_7d":[transaction_velocity_7d],
    "seconds_since_last_txn":[seconds_since_last_txn],
    "avg_amount_30d":[avg_amount_30d],
    "amount_deviation_ratio":[amount_deviation_ratio],
    "shared_device_count":[shared_device_count],
    "customer_merchant_txn_count":[customer_merchant_txn_count]
})

# =====================================================
# FRAUD PREDICTION
# =====================================================

fraud_probability = model.predict_proba(input_data)[0][1]

# =====================================================
# RBI EWS RULE ENGINE
# =====================================================

alerts = []

ews_score = 0

# HIGH VELOCITY

if transaction_velocity_7d > 25:
    alerts.append("🚨 High Transaction Velocity")
    ews_score += 25

# RAPID TRANSACTIONS

if seconds_since_last_txn < 60:
    alerts.append("🚨 Rapid Sequential Transactions")
    ews_score += 20

# ABNORMAL AMOUNT

if amount_deviation_ratio > 2:
    alerts.append("🚨 Abnormal Transaction Amount")
    ews_score += 20

# SHARED DEVICE

if shared_device_count > 2:
    alerts.append("🚨 Shared Device Risk")
    ews_score += 20

# NIGHT TRANSACTION

if is_night == 1:
    alerts.append("🚨 Night-Time Transaction")
    ews_score += 10

# REPEATED MERCHANT ACTIVITY

if customer_merchant_txn_count > 10:
    alerts.append("🚨 Repeated Customer-Merchant Activity")
    ews_score += 15

# =====================================================
# HYBRID RISK SCORE
# =====================================================

final_risk_score = (
    fraud_probability * 0.7 +
    (ews_score / 100) * 0.3
)

# =====================================================
# RISK CLASSIFICATION
# =====================================================

if final_risk_score > 0.8:
    risk_level = "🔴 HIGH RISK"

elif final_risk_score > 0.5:
    risk_level = "🟡 MEDIUM RISK"

else:
    risk_level = "🟢 LOW RISK"

# =====================================================
# KPI DASHBOARD
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Fraud Probability",
    f"{fraud_probability:.2%}"
)

col2.metric(
    "EWS Score",
    ews_score
)

col3.metric(
    "Risk Level",
    risk_level
)

# =====================================================
# REAL-TIME ALERT PANEL
# =====================================================

st.subheader("🚨 Real-Time RBI EWS Alerts")

if alerts:

    for alert in alerts:
        st.error(alert)

else:
    st.success("✅ No suspicious activity detected")

# =====================================================
# FRAUD RISK VISUALIZATION
# =====================================================

fig = px.pie(
    names=["Fraud Risk","Safe"],
    values=[fraud_probability,1-fraud_probability],
    title="Fraud Risk Distribution"
)

st.plotly_chart(fig, width='stretch')

# =====================================================
# INVESTIGATION SUMMARY
# =====================================================

st.subheader("🕵 Investigation Summary")

st.write(f"""
Transaction evaluated at: {datetime.now()}

Final Hybrid Risk Score: {final_risk_score:.2f}

Triggered RBI Early Warning Signals:
""")

for alert in alerts:
    st.write(alert)

# =====================================================
# CASE MANAGEMENT
# =====================================================

case_data = pd.DataFrame({
    "Case ID":["EWS001"],
    "Risk Level":[risk_level],
    "Fraud Probability":[fraud_probability],
    "EWS Score":[ews_score],
    "Status":["OPEN"]
})

st.subheader("📂 Fraud Case Management")

st.dataframe(case_data)
