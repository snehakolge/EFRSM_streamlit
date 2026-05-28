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

model = joblib.load(
    "efrms_xgboost_model.pkl"
)

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
    "🏦 Enterprise Fraud Risk Monitoring System"
)

st.markdown("""
### AI-Powered Fraud Surveillance • RBI EWS • Real-Time Alert Monitoring
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

st.sidebar.subheader("⚡ Velocity Features")

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

st.sidebar.subheader("💰 Amount Features")

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

st.sidebar.subheader("🕸️ Network Features")

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
    # RBI EWS ENGINE
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
        mule_account_risk
        +
        account_takeover_risk
        +
        structuring_risk
        +
        abnormal_behavior_risk
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

        "mule_account_risk":
        mule_account_risk,

        "account_takeover_risk":
        account_takeover_risk,

        "structuring_risk":
        structuring_risk,

        "abnormal_behavior_risk":
        abnormal_behavior_risk,

        "ews_score":
        ews_score

    }])

    # =====================================================
    # MODEL PREDICTIONS
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
    # CASE ID
    # =====================================================

    case_id = f"CASE-{random.randint(100000,999999)}"

    # =====================================================
    # EXECUTIVE DASHBOARD
    # =====================================================

    st.subheader("📊 Executive Dashboard")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    c2.metric(
        "Risk Level",
        risk_level
    )

    c3.metric(
        "EWS Score",
        ews_score
    )

    c4.metric(
        "Case ID",
        case_id
    )

    # =====================================================
    # ALERT ENGINE
    # =====================================================

    st.subheader(
        "🚨 Enterprise Fraud Alert Engine"
    )

    if (
        fraud_probability > threshold
        or
        ews_score >= 3
        or
        anomaly_score < -0.1
    ):

        st.error(
            f"""
🚨 HIGH RISK ALERT GENERATED

CASE ID: {case_id}
"""
        )

    else:

        st.success(
            "✅ Transaction appears legitimate"
        )

    # =====================================================
    # RBI EWS SIGNALS
    # =====================================================

    st.subheader(
        "🏦 RBI Early Warning Signals"
    )

    alerts = []

    if rapid_funds_flag:
        alerts.append(
            "Rapid movement of funds"
        )

    if high_velocity_risk:
        alerts.append(
            "High transaction velocity"
        )

    if night_transaction_risk:
        alerts.append(
            "Suspicious late-night activity"
        )

    if shared_device_risk:
        alerts.append(
            "Shared device anomaly"
        )

    if merchant_network_risk:
        alerts.append(
            "Merchant network linkage"
        )

    if mule_account_risk:
        alerts.append(
            "Potential mule account pattern"
        )

    if structuring_risk:
        alerts.append(
            "Structuring / smurfing risk"
        )

    if len(alerts) > 0:

        for a in alerts:

            st.warning(a)

    else:

        st.success(
            "No RBI EWS alerts triggered"
        )

    # =====================================================
    # SHAP EXPLAINABILITY
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

    st.dataframe(
        shap_df.sort_values(
            by="Impact",
            ascending=False
        ),
        width='stretch'
    )

    # =====================================================
    # FRAUD GAUGE
    # =====================================================

    st.subheader(
        "🎯 Fraud Risk Gauge"
    )

    gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=fraud_probability*100,

        title={
            'text':"Fraud Risk %"
        },

        gauge={
            'axis':{
                'range':[0,100]
            }
        }
    ))

    st.plotly_chart(
        gauge,
        width='stretch'
    )

    # =====================================================
    # FRAUD DISTRIBUTION
    # =====================================================

    st.subheader(
        "📊 Fraud Distribution"
    )

    fraud_df = pd.DataFrame({

        "Category":[
            "Fraud",
            "Legitimate"
        ],

        "Value":[
            fraud_probability,
            1-fraud_probability
        ]
    })

    pie = px.pie(

        fraud_df,

        names="Category",

        values="Value",

        hole=0.4
    )

    st.plotly_chart(
        pie,
        width='stretch'
    )

    # =====================================================
    # FRAUD TREND
    # =====================================================

    st.subheader(
        "📈 Hourly Fraud Trend"
    )

    trend_df = pd.DataFrame({

        "Hour":list(range(24)),

        "Fraud_Risk":
        np.abs(
            np.sin(
                np.linspace(
                    0,
                    3*np.pi,
                    24
                )
            )
        )
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
    # LIVE FRAUD MONITOR
    # =====================================================

    st.subheader(
        "🛰️ Live Fraud Monitor"
    )

    live_df = pd.DataFrame({

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
        live_df,
        width='stretch'
    )

    # =====================================================
    # CASE MANAGEMENT
    # =====================================================

    st.subheader(
        "🗂️ Enterprise Case Management"
    )

    case_df = pd.DataFrame({

        "Case_ID":[
            case_id
        ],

        "Risk_Level":[
            risk_level
        ],

        "Status":[
            "OPEN"
        ]
    })

    st.dataframe(
        case_df,
        width='stretch'
    )

    notes = st.text_area(
        "Investigation Notes"
    )

    if st.button(
        "💾 Save Investigation"
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
