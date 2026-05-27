# 🏦 Enterprise Fraud Risk Monitoring System (EFRMS)

AI-Powered Real-Time Fraud Detection Platform with RBI Early Warning Signals (EWS), Explainable AI, and Real-Time Alert Monitoring.

---

# 📌 Project Overview

This project simulates an enterprise-grade fraud risk monitoring system similar to those used in global banks for:

- Real-time fraud detection
- Early Warning Signal (EWS) monitoring
- Transaction anomaly detection
- Fraud risk scoring
- Live fraud alert generation
- Explainable AI (SHAP)
- Fraud case management

The system combines:

- Machine Learning
- RBI-inspired fraud scenarios
- Behavioral analytics
- Real-time alerting
- Anomaly detection

to detect suspicious financial transactions.

---

# 🚀 Key Features

## ✅ Real-Time Fraud Detection

- XGBoost fraud prediction engine
- Real-time fraud probability scoring
- Risk categorization:
  - LOW
  - MEDIUM
  - HIGH
  - CRITICAL

---

## ✅ RBI Early Warning Signals (EWS)

The system includes multiple fraud monitoring rules inspired by enterprise EFRMS systems.

### Supported EWS Scenarios

- Rapid movement of funds
- High transaction velocity
- Suspicious night transactions
- Shared device anomaly
- Merchant network risk
- Sudden amount spike
- Dormant account activation
- Structuring / smurfing risk
- Mule account pattern
- Potential account takeover
- Repeated merchant activity
- Abnormal customer behavior

---

## ✅ Explainable AI (SHAP)

The dashboard provides:

- Feature importance
- Transaction explainability
- SHAP impact visualization

to help fraud investigators understand why a transaction was flagged.

---

## ✅ Real-Time Fraud Alert Engine

The system generates live fraud alerts using:

- ML fraud probability
- RBI EWS rules
- Isolation Forest anomaly detection

---

## ✅ Executive Dashboard

The dashboard includes:

- Fraud probability
- Risk level
- EWS score
- Fraud risk gauge
- Fraud distribution pie chart
- Hourly fraud trend
- Live alert monitoring

---

## ✅ Fraud Case Management

Fraud analysts can:

- Review suspicious alerts
- Update investigation status
- Add investigator notes
- Escalate fraud cases

---

# 🧠 Machine Learning Models Used

## XGBoost Classifier

Used for:
- Fraud probability prediction
- Risk scoring

---

## Isolation Forest

Used for:
- Anomaly detection
- Unusual behavioral pattern detection

---

# 📊 Features Used

## Time Features

- hour
- weekday
- is_weekend
- is_night
- dayofyear
- weekofyear
- month

---

## Velocity Features

- transaction_velocity_7d
- seconds_since_last_txn

---

## Amount Features

- avg_amount_30d
- amount_deviation_ratio
- transaction_zscore

---

## Network Features

- shared_device_count
- customer_merchant_txn_count
- merchant_ring_id

---

## RBI EWS Features

- rapid_funds_flag
- high_velocity_risk
- night_transaction_risk
- shared_device_risk
- merchant_network_risk
- amount_spike_risk
- dormant_account_risk
- repeated_merchant_risk
- rapid_night_risk
- mule_account_risk
- account_takeover_risk
- structuring_risk
- abnormal_behavior_risk
- extreme_spike_flag
- ews_score

---

# 📈 Model Performance

## XGBoost Results

### Confusion Matrix

```text
[[48064   545]
 [  314  1077]]
```
