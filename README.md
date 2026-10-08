# Vendor Invoice Intelligence System

> AI-powered system that predicts freight costs and detects high-risk vendor invoices in real time.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://vendor-invoice-intelligence-systemgit.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)

---

## Live Demo

🔗 **[vendor-invoice-intelligence-systemgit.streamlit.app](https://vendor-invoice-intelligence-systemgit.streamlit.app/)**

---

## Problem Statement

Organizations process thousands of vendor invoices where freight costs may be inflated and anomalous invoices go undetected. Manual auditing is slow, error-prone, and expensive.

This system automates two core finance operations:

- **Predict** the expected freight cost for any vendor invoice
- **Flag** high-risk invoices before they reach the finance team

---

## Model Performance

| Model                   | Task           | Algorithm     | R² Score   | MAE   |
| ----------------------- | -------------- | ------------- | ---------- | ----- |
| Freight Cost Predictor  | Regression     | Random Forest | **96.59%** | 27.64 |
| Invoice Risk Classifier | Classification | Random Forest | **89.88%** | 0.05  |

---

## System Architecture

```
SQLite Database
      │
      ▼
Feature Engineering (SQL Aggregations)
      │
      ▼
Data Preprocessing + Scaling (StandardScaler)
      │
      ├──▶ Random Forest Regressor  ──▶ Predicted Freight Cost
      │
      └──▶ Random Forest Classifier ──▶ Safe / Manual Approval Flag
      │
      ▼
Streamlit Dashboard (Real-time Inference)
```

---

## Key Features

- **Freight Cost Prediction** — Regression model predicts freight from invoice features with 96.59% R²
- **Invoice Risk Detection** — Classifier flags anomalous invoices based on cost ratios and payment patterns
- **SQL Feature Engineering** — Aggregated features built directly from relational database tables
- **Scaler Pipeline** — StandardScaler saved and loaded at inference to prevent data leakage
- **Interactive Dashboard** — Dark-themed Streamlit UI with real-time predictions and risk chips

---

## Project Structure

```
Vendor-Invoice-Intelligence-System/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt
│
├── data/
│   └── inventory.db                # SQLite database
│
├── models/
│   ├── predict_freight_cost.pkl    # Trained regression model
│   ├── freight_scaler.pkl          # Scaler for freight features
│   ├── predict_flag_invoice.pkl    # Trained classification model
│   └── scaler.pkl                  # Scaler for flag features
│
├── invoice_flagging/
│   ├── train.py                    # Train invoice risk classifier
│   ├── train_freight.py            # Train freight cost regressor
│   ├── data_preprocessing.py       # SQL data loading + feature engineering
│   └── modeling_evaluation.py      # Model training + evaluation functions
│
└── inference/
    ├── predict_freight.py          # Freight cost inference
    └── predict_invoice_flag.py     # Invoice flag inference
```

---

## Feature Engineering

Features are built from a SQLite database using SQL aggregations across 4 tables: `vendor_invoice`, `purchases`, `purchase_prices`, `inventory_tables`.

| Feature               | Description                            |
| --------------------- | -------------------------------------- |
| `invoice_dollars`     | Total invoice value                    |
| `Freight`             | Freight cost on the invoice            |
| `days_po_to_invoice`  | Days between PO creation and invoice   |
| `days_to_pay`         | Days between invoice and payment       |
| `total_brands`        | Number of distinct brands on PO        |
| `total_item_quantity` | Total units on PO                      |
| `total_item_dollars`  | Total item value on PO                 |
| `avg_receiving_delay` | Average delay between PO and receiving |

---

## Invoice Risk Labeling Logic

Invoices are flagged as high-risk if any of these conditions are met:

```python
# Dollar discrepancy > 20% between invoice and item values
if abs(invoice_dollars - total_item_dollars) / total_item_dollars > 0.20

# Freight cost > 20% of invoice value (suspicious markup)
if freight / invoice_dollars > 0.20

# Unusually long payment terms
if days_to_pay > 90

# Long receiving delay
if avg_receiving_delay > 30
```

---

## Installation

```bash
git clone https://github.com/OfficialTanishGupta/Vendor-Invoice-Intelligence-System.git
cd Vendor-Invoice-Intelligence-System
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

## Retraining Models

```bash
# Train invoice risk classifier
cd invoice_flagging
python train.py

# Train freight cost regressor
python train_freight.py
```

---

## Tech Stack

| Layer      | Tools                                       |
| ---------- | ------------------------------------------- |
| Data       | SQLite, Pandas, SQL                         |
| ML         | Scikit-Learn, Random Forest, StandardScaler |
| Backend    | Python, Joblib                              |
| Frontend   | Streamlit                                   |
| Deployment | Streamlit Community Cloud                   |

---

## Business Impact

- Reduces manual invoice review workload
- Detects freight cost anomalies before payment
- Flags high-risk invoices with explainable risk signals
- Provides real-time predictions via web dashboard

---

## Future Improvements

- Add SHAP explainability to show why an invoice was flagged
- Implement XGBoost and LightGBM for comparison
- Add batch CSV upload for bulk invoice scoring
- Deploy on AWS with Docker containerization
- Add real-time database integration

---

## Author

**Tanish Gupta** — AI / Machine Learning Engineer

Focused on building production-ready ML systems and data-driven applications.

[![GitHub](https://img.shields.io/badge/GitHub-OfficialTanishGupta-181717?style=flat&logo=github)](https://github.com/OfficialTanishGupta)

---

⭐ If you found this useful, give it a star on GitHub!
