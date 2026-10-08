# 📊 Vendor Invoice Intelligence System

An AI-powered analytics system that predicts freight costs and identifies risky vendor invoices using machine learning.

This project helps businesses detect invoice anomalies, reduce cost leakage, and improve audit efficiency by combining data engineering, statistical analysis, and ML models.

## 🚀 Project Overview

Many organizations deal with thousands of vendor invoices where freight costs may be incorrect or risky. Manual auditing is slow and inefficient.

This system automates the process by:

Predicting expected freight cost

Flagging high-risk invoices

Providing data insights through visualization

Delivering results via an interactive Streamlit dashboard

## 🧠 Key Features

✅ Freight Cost Prediction using Regression Models
✅ Invoice Risk Detection using Classification Models
✅ SQL-based Feature Engineering
✅ Exploratory Data Analysis (EDA)
✅ Statistical Testing (T-Tests)
✅ ML Model Evaluation
✅ Real-time insights through Streamlit App

## 🏗 System Architecture

![Project Architecture](Project-images/project-architecture.png)

The pipeline follows a complete machine learning lifecycle:

1️⃣ Business Problem Definition
2️⃣ Data Collection from SQL Database
3️⃣ Feature Engineering using SQL
4️⃣ Exploratory Data Analysis
5️⃣ Machine Learning Modeling
6️⃣ Model Evaluation
7️⃣ Model Deployment
8️⃣ Streamlit Dashboard for Business Users

## 📂 Project Structure

```text
Vendor-Invoice-Intelligence-System/
│
├── data/
│   └── inventory.db                # SQLite database
│
├── notebooks/
│   └── analysis.ipynb              # EDA + experimentation
│
├── src/
│   ├── data_loader.py              # Load data from SQLite
│   ├── feature_engineering.py      # SQL aggregations
│   ├── eda_analysis.py             # Cost & risk analysis
│   ├── train_regression.py         # Freight cost prediction
│   ├── train_classification.py     # Risk classification
│   ├── evaluate_model.py           # Model evaluation
│   └── utils.py
│
├── models/
│   ├── freight_cost_model.pkl
│   └── risk_classifier.pkl
│
├── app/
│   └── streamlit_app.py            # Streamlit dashboard
│
├── requirements.txt
└── README.md
```

## 📊 Dataset

The project uses a SQLite database containing vendor and inventory information.

Tables

vendor_invoice

purchases

purchase_prices

inventory_tables

These tables are used to generate invoice-level aggregated features for modeling.

## 🔎 Exploratory Data Analysis

EDA helps understand patterns in:

Freight cost distribution

Vendor pricing behavior

Invoice risk patterns

Techniques used:

Data visualization

Correlation analysis

Statistical testing (T-Tests)

## 🤖 Machine Learning Models

### 1️⃣ Freight Cost Prediction

Predicts the expected freight cost using regression models.

Possible models:

Linear Regression

Random Forest Regressor

Gradient Boosting

Evaluation metrics:

MAE

RMSE

R² Score

### 2️⃣ Invoice Risk Flagging

Classifies invoices as Safe or Risky.

Possible models:

Logistic Regression

Random Forest

XGBoost

Evaluation metrics:

Precision

Recall

F1 Score

### 📈 Model Evaluation

Model performance is evaluated using:

Regression Metrics

Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

R² Score

Classification Metrics

Precision

Recall

F1 Score

## 💾 Model Deployment

Trained models are exported as:

.pkl files

These models are then used in the Streamlit application for real-time predictions.

## 🖥 Streamlit Dashboard

The Streamlit app allows business users to:

Upload or input invoice data

Predict freight cost

Detect risky invoices

Visualize cost insights

Run the app with:

streamlit run app/streamlit_app.py

## ⚙️ Installation

### Clone the repository:

git clone https://github.com/yourusername/vendor-invoice-intelligence-system.git

cd vendor-invoice-intelligence-system

### Install dependencies:

pip install -r requirements.txt

## ▶️ Running the Project

Run the notebook for experimentation:

jupyter notebook notebooks/analysis.ipynb

Run Streamlit dashboard:

streamlit run app/streamlit_app.py

## 🛠 Tech Stack

Python

Pandas

NumPy

Scikit-Learn

SQLite

Matplotlib / Seaborn

Streamlit

## 🎯 Business Impact

This system helps companies:

Reduce invoice fraud risk

Detect freight cost anomalies

Improve audit efficiency

Gain real-time insights

## 📌 Future Improvements

Add XGBoost and LightGBM models

Implement SHAP explainability

Deploy on AWS / Docker

Add real-time database integration

## 👨‍💻 Author

Tanish Gupta

AI / Machine Learning Enthusiast
Focused on building real-world AI systems and data-driven applications.

## ⭐ If you like this project

Give it a ⭐ on GitHub!
