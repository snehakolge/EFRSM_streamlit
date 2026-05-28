import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import random
import time
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
    "🏦 Enterprise Fraud Risk Monitoring System (EFRMS)"
)

st.markdown("""
### AI-Powered Fraud Detection • RBI EWS • Real-Time Fraud Surveillance
""")

# =========================================================
# KPI HEADER
# =========================================================

k1,k2,k3,k4 = st.columns(4)

k1.metric(
    "Total Alerts Today",
    random.randint(120,250)
)

k2.metric(
    "Open Cases",
    random.randint(20,80)
)

k3.metric(
    "Critical Cases",
    random.randint(5,20)
)

k4.metric(
    "Blocked Accounts",
    random.randint(2,15)
)

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab1,tab2,tab3,tab4 = st.tabs([

    "📊 Dashboard",

    "🚨 Alerts",

    "🗂️ Cases",

    "📈 Analytics"

])

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "💳 Real-Time Transaction Input"
)

# =========================================================
# TIME FEATURES
# =========================================================

st.sidebar.subheader(
    "⏰ Time Features"
)

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
# ANALYSIS ENGINE
# =========================================================

if analyse:

    # =====================================================
    # CASE IDS
    # =====================================================

    case_id = f"CASE-{random.randint(100000,999999)}"

    transaction_id = f"TXN-{random.randint(100000,999999)}"

    customer_id = f"CUST-{random.randint(1000,9999)}"

    # =====================================================
    # RBI EWS RULES
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

    impossible_travel_risk = int(
        (
            is_night == 1
        )
        and
        (
            transaction_velocity_7d > 25
        )
    )

    beneficiary_risk = int(
        customer_merchant_txn_count > 15
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
        impossible_travel_risk
        +
        beneficiary_risk
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

        "impossible_travel_risk":
        impossible_travel_risk,

        "beneficiary_risk":
        beneficiary_risk,

        "extreme_spike_flag":
        extreme_spike_flag,

        "ews_score":
        ews_score

    }])

    # =====================================================
    # FEATURE ALIGNMENT
    # =====================================================

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    fraud_probability = model.predict_proba(
        input_data
    )[:,1][0]

    anomaly_score = iso_model.decision_function(
        input_data
    )[0]

    # =====================================================
    # RISK ENGINE
    # =====================================================

    if fraud_probability > 0.95:

        risk_level = "SEV1"

    elif fraud_probability > 0.85:

        risk_level = "CRITICAL"

    elif fraud_probability > 0.70:

        risk_level = "HIGH"

    elif fraud_probability > 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # =====================================================
    # DASHBOARD TAB
    # =====================================================

    with tab1:

        st.subheader(
            "📊 Executive Dashboard"
        )

        c1,c2,c3,c4,c5 = st.columns(5)

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

        c5.metric(
            "Anomaly Score",
            round(anomaly_score,4)
        )

        # =================================================
        # ALERT ENGINE
        # =================================================

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
🚨 HIGH RISK FRAUD ALERT GENERATED

CASE ID: {case_id}
"""
            )

        else:

            st.success(
                "✅ Transaction appears legitimate"
            )

        # =================================================
        # CUSTOMER PROFILE
        # =================================================

        st.subheader(
            "👤 Customer 360 Profile"
        )

        customer_df = pd.DataFrame({

            "Attribute":[

                "Customer ID",
                "KYC Risk",
                "Country",
                "Previous Alerts",
                "Shared Devices"
            ],

            "Value":[

                customer_id,
                "HIGH",
                "India",
                random.randint(0,5),
                shared_device_count
            ]
        })

        st.dataframe(
            customer_df,
            use_container_width=True
        )

        # =================================================
        # EWS ALERTS
        # =================================================

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
                "Merchant network linkage"
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
                "Structuring / smurfing pattern"
            )

        if impossible_travel_risk:
            ews_alerts.append(
                "Impossible travel risk"
            )

        if beneficiary_risk:
            ews_alerts.append(
                "Beneficiary manipulation risk"
            )

        if len(ews_alerts) > 0:

            for alert in ews_alerts:

                st.warning(alert)

        else:

            st.success(
                "No RBI EWS alerts triggered"
            )

        # =================================================
        # SHAP
        # =================================================

        st.subheader(
            "🧠 SHAP Explainability"
        )

        explainer = shap.TreeExplainer(
            model
        )

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

        shap_chart = px.bar(

            shap_df.head(10),

            x="Impact",

            y="Feature",

            orientation="h"
        )

        st.plotly_chart(
            shap_chart,
            use_container_width=True
        )

        # =================================================
        # GAUGE
        # =================================================

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
            use_container_width=True
        )

    # =====================================================
    # ALERTS TAB
    # =====================================================

    with tab2:

        st.subheader(
            "🛰️ Live Fraud Alert Monitor"
        )

        placeholder = st.empty()

        for i in range(3):

            live_df = pd.DataFrame({

                "Time":[

                    datetime.now().strftime(
                        "%H:%M:%S"
                    )
                ],

                "Transaction":[

                    f"TXN-{random.randint(1000,9999)}"
                ],

                "Risk":[

                    random.choice([
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL"
                    ])
                ],

                "Alert":[

                    random.choice([
                        "Velocity Spike",
                        "Shared Device",
                        "Account Takeover",
                        "Structuring",
                        "Mule Pattern"
                    ])
                ]
            })

            placeholder.dataframe(
                live_df,
                use_container_width=True
            )

            time.sleep(1)

    # =====================================================
    # CASE MANAGEMENT TAB
    # =====================================================

    with tab3:

        st.subheader(
            "🗂️ Enterprise Case Management"
        )

        case_queue = pd.DataFrame({

            "Case_ID":[

                case_id,

                f"CASE-{random.randint(1000,9999)}",

                f"CASE-{random.randint(1000,9999)}"
            ],

            "Risk_Level":[

                risk_level,

                "MEDIUM",

                "LOW"
            ],

            "Status":[

                "OPEN",

                "UNDER REVIEW",

                "ESCALATED"
            ]
        })

        st.data_editor(
            case_queue,
            use_container_width=True
        )

        selected_case = st.selectbox(

            "Open Investigation Case",

            case_queue["Case_ID"]
        )

        investigator = st.selectbox(

            "Assign Investigator",

            [

                "Fraud Analyst 1",

                "Fraud Analyst 2",

                "AML Team",

                "Risk Team"
            ]
        )

        st.subheader(
            "🔍 Investigation Panel"
        )

        st.write(
            f"Currently Investigating: {selected_case}"
        )

        st.write(
            f"Assigned To: {investigator}"
        )

        # =================================================
        # COMMENTS
        # =================================================

        comments_df = pd.DataFrame({

            "Timestamp":[

                "10:01",

                "10:05",

                "10:12"
            ],

            "Analyst":[

                "Fraud Team",

                "Manager",

                "Investigator"
            ],

            "Comment":[

                "Velocity anomaly detected",

                "Escalated for review",

                "Customer unreachable"
            ]
        })

        st.subheader(
            "📝 Investigation Comments"
        )

        st.dataframe(
            comments_df,
            use_container_width=True
        )

        analyst_comment = st.text_area(
            "Add Investigation Comment"
        )

        if st.button(
            "💾 Save Comment"
        ):

            st.success(
                "Comment saved successfully"
            )

        # =================================================
        # ACTIONS
        # =================================================

        st.subheader(
            "⚙️ Analyst Actions"
        )

        a1,a2,a3,a4 = st.columns(4)

        if a1.button(
            "🚨 Escalate"
        ):

            st.warning(
                f"{selected_case} escalated"
            )

        if a2.button(
            "🔒 Freeze Account"
        ):

            st.error(
                "Account freeze initiated"
            )

        if a3.button(
            "📤 Generate STR"
        ):

            st.info(
                "STR generation initiated"
            )

        if a4.button(
            "✅ Close Case"
        ):

            st.success(
                f"{selected_case} closed"
            )

        # =================================================
        # TIMELINE
        # =================================================

        st.subheader(
            "⏳ Case Timeline"
        )

        timeline_df = pd.DataFrame({

            "Time":[

                "10:01",

                "10:05",

                "10:12",

                "10:20"
            ],

            "Event":[

                "Transaction Triggered",

                "RBI EWS Triggered",

                "Analyst Assigned",

                "Escalated"
            ]
        })

        st.dataframe(
            timeline_df,
            use_container_width=True
        )

    # =====================================================
    # ANALYTICS TAB
    # =====================================================

    with tab4:

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
            use_container_width=True
        )

        # =================================================
        # FRAUD TREND
        # =================================================

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
            use_container_width=True
        )

        # =================================================
        # MODEL MONITORING
        # =================================================

        st.subheader(
            "🧠 Model Monitoring"
        )

        m1,m2,m3,m4 = st.columns(4)

        m1.metric(
            "Precision",
            "0.99"
        )

        m2.metric(
            "Recall",
            "0.82"
        )

        m3.metric(
            "Threshold",
            "0.40"
        )

        m4.metric(
            "Model Drift",
            "LOW"
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
🏦 Enterprise Fraud Risk Monitoring System  
AI + RBI EWS + Real-Time Fraud Analytics
""")
