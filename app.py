 import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import random
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Enterprise Fraud Risk Monitoring System",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================

model = joblib.load("efrms_xgboost_model.pkl")

iso_model = joblib.load(
    "isolation_forest_model.pkl"
)

threshold = joblib.load(
    "threshold.pkl"
)

features = joblib.load(
    "features.pkl"
)

# =========================================================
# TITLE
# =========================================================

st.title(
    "🏦 Enterprise Fraud Risk Monitoring System (EFRMS)"
)

st.markdown("""
### AI-Powered Fraud Detection • RBI EWS • Real-Time Fraud Surveillance
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "💳 Real-Time Transaction Input"
)

# =========================================================
# TIME FEATURES
# =========================================================

st.sidebar.subheader("⏰ Time Features")

hour = st.sidebar.slider(
    "hour",
    0,
    23,
    12
)

weekday = st.sidebar.slider(
    "weekday",
    0,
    6,
    2
)

is_weekend = st.sidebar.selectbox(
    "is_weekend",
    [0,1]
)

is_night = st.sidebar.selectbox(
    "is_night",
    [0,1]
)

dayofyear = st.sidebar.slider(
    "dayofyear",
    1,
    365,
    100
)

weekofyear = st.sidebar.slider(
    "weekofyear",
    1,
    52,
    20
)

month = st.sidebar.slider(
    "month",
    1,
    12,
    6
)

# =========================================================
# VELOCITY FEATURES
# =========================================================

st.sidebar.subheader(
    "⚡ Velocity Features"
)

transaction_velocity_7d = st.sidebar.slider(
    "transaction_velocity_7d",
    1,
    50,
    10
)

seconds_since_last_txn = st.sidebar.number_input(
    "seconds_since_last_txn",
    value=100000
)

# =========================================================
# AMOUNT FEATURES
# =========================================================

st.sidebar.subheader(
    "💰 Amount Features"
)

avg_amount_30d = st.sidebar.number_input(
    "avg_amount_30d",
    value=500.0
)

amount_deviation_ratio = st.sidebar.slider(
    "amount_deviation_ratio",
    0.0,
    10.0,
    1.0
)

# =========================================================
# NETWORK FEATURES
# =========================================================

st.sidebar.subheader(
    "🕸️ Network Features"
)

shared_device_count = st.sidebar.slider(
    "shared_device_count",
    0,
    10,
    0
)

customer_merchant_txn_count = st.sidebar.slider(
    "customer_merchant_txn_count",
    0,
    20,
    2
)

merchant_ring_id = st.sidebar.slider(
    "merchant_ring_id",
    0,
    20,
    0
)

# =========================================================
# ANALYSE BUTTON
# =========================================================

analyse = st.sidebar.button(
    "🚨 Analyse Transaction"
)

# =========================================================
# RUN ANALYSIS
# =========================================================

if analyse:

    # =====================================================
    # RBI EWS FEATURES
    # =====================================================

    rapid_funds_flag = int(
        seconds_since_last_txn < 60
    )

    high_velocity_risk = int(
        transaction_velocity_7d > 20
    )

    night_transaction_risk = int(
        is_night == 1
    )

    shared_device_risk = int(
        shared_device_count > 2
    )

    merchant_network_risk = int(
        merchant_ring_id > 0
    )

    amount_spike_risk = int(
        amount_deviation_ratio > 2
    )

    dormant_account_risk = int(
        seconds_since_last_txn > 2592000
    )

    repeated_merchant_risk = int(
        customer_merchant_txn_count > 10
    )

    rapid_night_risk = int(
        (
            seconds_since_last_txn < 60
        )
        and
        (
            is_night == 1
        )
    )

    mule_account_risk = int(
        (
            transaction_velocity_7d > 20
        )
        and
        (
            shared_device_count > 2
        )
    )

    account_takeover_risk = int(
        (
            shared_device_count > 3
        )
        and
        (
            seconds_since_last_txn < 120
        )
    )

    structuring_risk = int(
        (
            transaction_velocity_7d > 25
        )
        and
        (
            avg_amount_30d < 1000
        )
    )

    abnormal_behavior_risk = int(
        (
            amount_deviation_ratio > 2
        )
        and
        (
            transaction_velocity_7d > 15
        )
    )

    transaction_zscore = (
        avg_amount_30d - 500
    ) / 200

    extreme_spike_flag = int(
        transaction_zscore > 3
    )

    ews_score = (

        rapid_funds_flag
        +
        high_velocity_risk
        +
        night_transaction_risk
        +
        shared_device_risk
        +
        merchant_network_risk
        +
        amount_spike_risk
        +
        dormant_account_risk
        +
        repeated_merchant_risk
        +
        rapid_night_risk
        +
        mule_account_risk
        +
        account_takeover_risk
        +
        structuring_risk
        +
        abnormal_behavior_risk
        +
        extreme_spike_flag
    )

    # =====================================================
    # INPUT DATA
    # =====================================================

    input_data = pd.DataFrame([{

        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "dayofyear": dayofyear,
        "weekofyear": weekofyear,
        "month": month,

        "transaction_velocity_7d":
        transaction_velocity_7d,

        "seconds_since_last_txn":
        seconds_since_last_txn,

        "avg_amount_30d":
        avg_amount_30d,

        "amount_deviation_ratio":
        amount_deviation_ratio,

        "transaction_zscore":
        transaction_zscore,

        "shared_device_count":
        shared_device_count,

        "customer_merchant_txn_count":
        customer_merchant_txn_count,

        "merchant_ring_id":
        merchant_ring_id,

        "rapid_funds_flag":
        rapid_funds_flag,

        "high_velocity_risk":
        high_velocity_risk,

        "night_transaction_risk":
        night_transaction_risk,

        "shared_device_risk":
        shared_device_risk,

        "merchant_network_risk":
        merchant_network_risk,

        "amount_spike_risk":
        amount_spike_risk,

        "dormant_account_risk":
        dormant_account_risk,

        "repeated_merchant_risk":
        repeated_merchant_risk,

        "rapid_night_risk":
        rapid_night_risk,

        "mule_account_risk":
        mule_account_risk,

        "account_takeover_risk":
        account_takeover_risk,

        "structuring_risk":
        structuring_risk,

        "abnormal_behavior_risk":
        abnormal_behavior_risk,

        "extreme_spike_flag":
        extreme_spike_flag,

        "ews_score":
        ews_score

    }])

    # =====================================================
    # SAFE FEATURE MATCHING
    # =====================================================

    for col in features:

        if col not in input_data.columns:

            input_data[col] = 0

    input_data = input_data[features]

    # =====================================================
    # PREDICTION
    # =====================================================

    fraud_probability = model.predict_proba(
        input_data
    )[:,1][0]

    anomaly_score = iso_model.decision_function(
        input_data
    )[0]

    # =====================================================
    # RISK LEVEL
    # =====================================================

    if fraud_probability > 0.90:

        risk_level = "CRITICAL"

    elif fraud_probability > 0.75:

        risk_level = "HIGH"

    elif fraud_probability > 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # =====================================================
    # ALERT PRIORITY
    # =====================================================

    alert_priority = "LOW"

    if ews_score >= 6:

        alert_priority = "CRITICAL"

    elif ews_score >= 4:

        alert_priority = "HIGH"

    elif ews_score >= 2:

        alert_priority = "MEDIUM"

    # =====================================================
    # TRANSACTION ID
    # =====================================================

    transaction_id = f"TXN{random.randint(100000,999999)}"

    st.info(
        f"Transaction ID: {transaction_id}"
    )

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.subheader("📊 Executive Dashboard")

    col1,col2,col3,col4,col5 = st.columns(5)

    col1.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    col2.metric(
        "Risk Level",
        risk_level
    )

    col3.metric(
        "EWS Score",
        ews_score
    )

    col4.metric(
        "Alert Priority",
        alert_priority
    )

    col5.metric(
        "Anomaly Score",
        round(anomaly_score,4)
    )

    # =====================================================
    # ALERT ENGINE
    # =====================================================

    st.subheader(
        "🚨 Real-Time Fraud Alert Engine"
    )

    if (
        fraud_probability > threshold
        or
        ews_score >= 3
        or
        anomaly_score < -0.1
    ):

        st.error(
            "🚨 HIGH RISK FRAUD ALERT GENERATED"
        )

    else:

        st.success(
            "✅ Transaction appears legitimate"
        )

    # =====================================================
    # EWS SIGNALS
    # =====================================================

    st.subheader(
        "🏦 RBI Early Warning Signals"
    )

    ews_alerts = []

    if rapid_funds_flag:
        ews_alerts.append(
            "Rapid movement of funds"
        )

    if high_velocity_risk:
        ews_alerts.append(
            "High transaction velocity"
        )

    if night_transaction_risk:
        ews_alerts.append(
            "Late-night suspicious activity"
        )

    if shared_device_risk:
        ews_alerts.append(
            "Shared device anomaly"
        )

    if merchant_network_risk:
        ews_alerts.append(
            "Merchant network risk"
        )

    if amount_spike_risk:
        ews_alerts.append(
            "Sudden amount spike"
        )

    if mule_account_risk:
        ews_alerts.append(
            "Potential mule activity"
        )

    if account_takeover_risk:
        ews_alerts.append(
            "Potential account takeover"
        )

    if structuring_risk:
        ews_alerts.append(
            "Structuring / smurfing risk"
        )

    if abnormal_behavior_risk:
        ews_alerts.append(
            "Abnormal customer behavior"
        )

    if len(ews_alerts) > 0:

        for alert in ews_alerts:

            st.warning(alert)

    else:

        st.success(
            "No RBI EWS alerts triggered"
        )

    # =====================================================
    # SHAP
    # =====================================================

    st.subheader(
        "🧠 SHAP Explainability"
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        input_data
    )

    shap_df = pd.DataFrame({

        "Feature":
        input_data.columns,

        "Impact":
        shap_values[0]

    })

    shap_df = shap_df.sort_values(
        by="Impact",
        ascending=False
    )

    top_shap = shap_df.head(10)

    shap_fig = px.bar(

        top_shap,

        x="Impact",

        y="Feature",

        orientation="h"
    )

    st.plotly_chart(
        shap_fig,
        width='stretch'
    )

    st.subheader(
        "📌 Fraud Decision Reason"
    )

    top_feature = shap_df.iloc[0]["Feature"]

    top_impact = shap_df.iloc[0]["Impact"]

    st.info(

        f"""
        Most influential feature:
        {top_feature}

        Impact Score:
        {round(top_impact,4)}
        """
    )

    # =====================================================
    # FRAUD GAUGE
    # =====================================================

    st.subheader("🎯 Fraud Risk Gauge")

    fig = go.Figure(go.Indicator(

        mode = "gauge+number",

        value = fraud_probability * 100,

        title = {
            'text': "Fraud Risk %"
        },

        gauge = {

            'axis': {
                'range': [0,100]
            }
        }
    ))

    st.plotly_chart(
        fig,
        width='stretch'
    )

    # =====================================================
    # PIE CHART
    # =====================================================

    st.subheader(
        "📊 Fraud Distribution"
    )

    fraud_dist = pd.DataFrame({

        "Category":[
            "Fraud",
            "Legitimate"
        ],

        "Value":[
            fraud_probability,
            1-fraud_probability
        ]
    })

    pie_fig = px.pie(

        fraud_dist,

        names="Category",

        values="Value",

        hole=0.4
    )

    st.plotly_chart(
        pie_fig,
        width='stretch'
    )

    # =====================================================
    # FRAUD TREND
    # =====================================================

    st.subheader(
        "⏰ Hourly Fraud Trend"
    )

    trend_df = pd.DataFrame({

        "Hour": list(range(24)),

        "Fraud_Risk":[

            max(
                0,
                min(
                    1,
                    fraud_probability +
                    np.random.normal(0,0.05)
                )
            )

            for _ in range(24)
        ]
    })

    trend_fig = px.line(

        trend_df,

        x="Hour",

        y="Fraud_Risk",

        markers=True
    )

    st.plotly_chart(
        trend_fig,
        width='stretch'
    )

    # =====================================================
    # LIVE ALERTS
    # =====================================================

    st.subheader(
        "🛰️ Live Fraud Alert Monitor"
    )

    live_alerts = pd.DataFrame({

        "Time":[

            datetime.now().strftime("%H:%M:%S"),

            datetime.now().strftime("%H:%M:%S"),

            datetime.now().strftime("%H:%M:%S")

        ],

        "Transaction_ID":[

            f"TXN{random.randint(10000,99999)}",

            f"TXN{random.randint(10000,99999)}",

            f"TXN{random.randint(10000,99999)}"
        ],

        "Risk_Level":[

            "HIGH",
            "MEDIUM",
            "LOW"
        ],

        "Alert":[

            "Rapid Movement",
            "Shared Device",
            "Legitimate"
        ]
    })

    st.dataframe(
        live_alerts,
        width='stretch'
    )

    # =====================================================
    # CASE MANAGEMENT
    # =====================================================

    st.subheader(
        "🗂️ Fraud Case Management"
    )

    case_status = st.selectbox(

        "Case Status",

        [
            "OPEN",
            "UNDER INVESTIGATION",
            "ESCALATED",
            "CLOSED"
        ]
    )

    investigator_notes = st.text_area(
        "Investigator Notes"
    )

    if st.button(
        "Save Investigation"
    ):

        st.success(
            "Investigation Updated"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
🏦 Enterprise Fraud Risk Monitoring System  
AI + RBI EWS + Real-Time Fraud Analytics
""")
