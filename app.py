import streamlit as st
import pandas as pd
import numpy as np
 
 
import joblib
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FREIGHT_MODEL_PATH  = os.path.join(BASE_DIR, "models", "predict_freight_cost.pkl")
FREIGHT_SCALER_PATH = os.path.join(BASE_DIR, "models", "freight_scaler.pkl")
FLAG_MODEL_PATH     = os.path.join(BASE_DIR, "models", "predict_flag_invoice.pkl")
FLAG_SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")

FREIGHT_FEATURES = [
    "invoice_dollars", "days_po_to_invoice", "days_to_pay",
    "total_brands", "total_item_quantity", "total_item_dollars"
]

FLAG_FEATURES = [
    "invoice_dollars", "Freight", "total_item_quantity",
    "total_item_dollars", "days_po_to_invoice", "days_to_pay"
]

def predict_freight_cost(input_data):
    model  = joblib.load(FREIGHT_MODEL_PATH)
    scaler = joblib.load(FREIGHT_SCALER_PATH)
    df = pd.DataFrame(input_data)
    for col in FREIGHT_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X_scaled = scaler.transform(df[FREIGHT_FEATURES])
    df['Predicted_Freight_Amt'] = model.predict(X_scaled).round(2)
    return df

def predict_invoice_flag(input_data):
    model  = joblib.load(FLAG_MODEL_PATH)
    scaler = joblib.load(FLAG_SCALER_PATH)
    df = pd.DataFrame(input_data)
    for col in FLAG_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X        = df[FLAG_FEATURES].values
    X_scaled = scaler.transform(X)
    df['Predicted_Flag'] = model.predict(X_scaled)
    return df
 
# --------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------
 
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)
 
# --------------------------------------------------------------
# Custom CSS — Dark Financial Dashboard Aesthetic
# --------------------------------------------------------------
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
 
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
}
 
/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(120deg, #0f172a 0%, #1e293b 40%, #0f2744 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 20px;
    padding: 48px 56px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}
.hero-title span { color: #38bdf8; }
.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #64748b;
    margin: 0 0 32px 0;
    line-height: 1.7;
}
.hero-stats { display: flex; gap: 40px; flex-wrap: wrap; }
.stat-item { display: flex; flex-direction: column; gap: 4px; }
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #38bdf8;
}
.stat-label {
    font-size: 11px;
    color: #475569;
    letter-spacing: 1px;
    text-transform: uppercase;
}
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0b1120 !important;
    border-right: 1px solid rgba(56, 189, 248, 0.08);
}
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #38bdf8;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.sidebar-tagline {
    font-size: 10px;
    color: #334155;
    letter-spacing: 2px;
    margin-bottom: 28px;
    font-family: 'DM Mono', monospace;
}
.impact-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(56,189,248,0.05);
    font-size: 12px;
    color: #64748b;
    font-family: 'DM Mono', monospace;
}
.impact-dot {
    width: 5px; height: 5px;
    background: #38bdf8;
    border-radius: 50%;
    flex-shrink: 0;
}
 
/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.1);
}
.section-icon {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(99,102,241,0.12));
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}
.section-desc {
    font-size: 11px;
    color: #334155;
    margin: 3px 0 0 0;
    letter-spacing: 1px;
    font-family: 'DM Mono', monospace;
}
 
/* ── Input Labels ── */
.input-label {
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 12px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    opacity: 0.8;
}
 
/* ── Number Inputs ── */
.stNumberInput input {
    background: rgba(10, 14, 26, 0.9) !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
    transition: border-color 0.2s;
}
.stNumberInput input:focus {
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.07) !important;
}
.stNumberInput label {
    color: #475569 !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
}
 
/* ── Submit Button ── */
.stFormSubmitButton button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    margin-top: 20px !important;
    transition: all 0.3s ease !important;
}
.stFormSubmitButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25) !important;
}
 
/* ── Result Cards ── */
.result-card-freight {
    background: linear-gradient(135deg, rgba(56,189,248,0.06), rgba(99,102,241,0.04));
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-left: 4px solid #38bdf8;
    border-radius: 16px;
    padding: 32px 36px;
    margin-top: 24px;
    animation: fadeIn 0.4s ease;
}
.result-card-success {
    background: linear-gradient(135deg, rgba(16,185,129,0.07), rgba(5,150,105,0.04));
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-left: 4px solid #10b981;
    border-radius: 16px;
    padding: 32px 36px;
    margin-top: 24px;
    animation: fadeIn 0.4s ease;
}
.result-card-danger {
    background: linear-gradient(135deg, rgba(239,68,68,0.07), rgba(220,38,38,0.04));
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-left: 4px solid #ef4444;
    border-radius: 16px;
    padding: 32px 36px;
    margin-top: 24px;
    animation: fadeIn 0.4s ease;
}
 
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
 
.result-label {
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 10px;
    font-family: 'DM Mono', monospace;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #38bdf8;
    line-height: 1;
    margin-bottom: 10px;
}
.result-flag-safe {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #10b981;
    margin-bottom: 6px;
}
.result-flag-danger {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #ef4444;
    margin-bottom: 6px;
}
.result-sublabel {
    font-size: 12px;
    color: #475569;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.chip {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.12);
    color: #64748b;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
}
 
/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
hr { border-color: rgba(56,189,248,0.07) !important; }
 
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)
 
# --------------------------------------------------------------
# Hero Banner
# --------------------------------------------------------------
 
st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">AI-POWERED · FINANCE OPS · ML SYSTEM</div>
    <h1 class="hero-title">Vendor Invoice<br><span>Intelligence</span> Portal</h1>
    <p class="hero-subtitle">
        Machine learning models that predict freight costs and detect high-risk<br>
        invoices before they reach your finance team — in real time.
    </p>
    <div class="hero-stats">
        <div class="stat-item">
            <span class="stat-value">96.6%</span>
            <span class="stat-label">Freight R²</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">89.9%</span>
            <span class="stat-label">Classifier R²</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">2</span>
            <span class="stat-label">AI Models</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">Live</span>
            <span class="stat-label">Inference</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
 
# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
 
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ InvoiceIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">VENDOR INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)
 
    selected_model = st.radio(
        "SELECT MODULE",
        ["Freight Cost Prediction", "Invoice Manual Approval Flag"]
    )
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;color:#334155;font-family:\'DM Mono\',monospace;margin-bottom:8px;">BUSINESS IMPACT</div>',
        unsafe_allow_html=True
    )
    for item in [
        "Accurate freight forecasting",
        "Reduced invoice fraud risk",
        "Faster finance operations",
        "Real-time anomaly detection"
    ]:
        st.markdown(f'<div class="impact-item"><div class="impact-dot"></div>{item}</div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#1e293b;font-family:'DM Mono',monospace;line-height:2;">
        TECH STACK<br>
        ├─ Python · Scikit-learn<br>
        ├─ SQLite · Pandas<br>
        ├─ Random Forest Models<br>
        └─ Streamlit Dashboard
    </div>
    """, unsafe_allow_html=True)
 
# --------------------------------------------------------------
# Freight Cost Prediction
# --------------------------------------------------------------
 
if selected_model == "Freight Cost Prediction":
 
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🚚</div>
        <div>
            <div class="section-title">Freight Cost Prediction</div>
            <div class="section-desc">REGRESSION · RANDOM FOREST · R² = 96.59%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    with st.form("freight_form"):
        col1, col2 = st.columns(2, gap="large")
 
        with col1:
            st.markdown('<div class="input-label">Invoice Details</div>', unsafe_allow_html=True)
            invoice_dollars    = st.number_input("Invoice Dollars ($)",    min_value=1.0,  value=25000.0, step=100.0)
            days_po_to_invoice = st.number_input("Days — PO to Invoice",   min_value=0,    value=5)
            days_to_pay        = st.number_input("Days to Pay",            min_value=0,    value=30)
 
        with col2:
            st.markdown('<div class="input-label">Item & Vendor Details</div>', unsafe_allow_html=True)
            total_brands        = st.number_input("Total Brands",          min_value=1,    value=5)
            total_item_quantity = st.number_input("Total Item Quantity",   min_value=1,    value=50)
            total_item_dollars  = st.number_input("Total Item Dollars ($)", min_value=1.0, value=24000.0, step=100.0)
 
        submit_freight = st.form_submit_button("⚡  Run Freight Prediction")
 
    if submit_freight:
        input_data = {
            "invoice_dollars":    [invoice_dollars],
            "days_po_to_invoice": [days_po_to_invoice],
            "days_to_pay":        [days_to_pay],
            "total_brands":       [total_brands],
            "total_item_quantity":[total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }
 
        with st.spinner("Running model inference..."):
            prediction_result = predict_freight_cost(input_data)
            prediction_value  = prediction_result['Predicted_Freight_Amt'].iloc[0]
 
        freight_pct = (float(prediction_value) / invoice_dollars) * 100
 
        st.markdown(f"""
        <div class="result-card-freight">
            <div class="result-label">Estimated Freight Cost</div>
            <div class="result-value">${float(prediction_value):,.2f}</div>
            <div class="result-sublabel">
                Approximately {freight_pct:.1f}% of total invoice value<br>
                Predicted by Random Forest Regressor · R² = 96.59%
            </div>
            <div class="chip-row">
                <span class="chip">📦 Invoice: ${invoice_dollars:,.0f}</span>
                <span class="chip">🏷 Items: ${total_item_dollars:,.0f}</span>
                <span class="chip">📊 Qty: {total_item_quantity}</span>
                <span class="chip">🗓 Pay in {days_to_pay}d</span>
                <span class="chip">🏢 Brands: {total_brands}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
# --------------------------------------------------------------
# Invoice Flag Prediction
# --------------------------------------------------------------
 
else:
 
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔍</div>
        <div>
            <div class="section-title">Invoice Risk Detection</div>
            <div class="section-desc">CLASSIFICATION · RANDOM FOREST · R² = 89.88%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3, gap="large")
 
        with col1:
            st.markdown('<div class="input-label">Cost Signals</div>', unsafe_allow_html=True)
            invoice_dollars = st.number_input("Invoice Dollars ($)",    min_value=1.0, value=352.95)
            freight_input   = st.number_input("Freight Cost ($)",       min_value=0.0, value=1.73)
 
        with col2:
            st.markdown('<div class="input-label">Item Details</div>', unsafe_allow_html=True)
            total_item_quantity = st.number_input("Total Item Quantity",    min_value=1,   value=162)
            total_item_dollars  = st.number_input("Total Item Dollars ($)", min_value=1.0, value=2476.0)
 
        with col3:
            st.markdown('<div class="input-label">Timeline</div>', unsafe_allow_html=True)
            days_po_to_invoice = st.number_input("Days — PO to Invoice", min_value=0, value=5)
            days_to_pay        = st.number_input("Days to Pay",          min_value=0, value=30)
 
        submit_flag = st.form_submit_button("🔎  Evaluate Invoice Risk")
 
    if submit_flag:
        input_data = {
            "invoice_dollars":    [invoice_dollars],
            "Freight":            [freight_input],
            "total_item_quantity":[total_item_quantity],
            "total_item_dollars": [total_item_dollars],
            "days_po_to_invoice": [days_po_to_invoice],
            "days_to_pay":        [days_to_pay]
        }
 
        with st.spinner("Analysing invoice risk signals..."):
            flag_result     = predict_invoice_flag(input_data)
            flag_prediction = flag_result['Predicted_Flag']
            is_flagged      = bool(flag_prediction.iloc[0])
 
        dollar_diff_pct = abs(invoice_dollars - total_item_dollars) / total_item_dollars * 100 if total_item_dollars > 0 else 0
        freight_pct     = freight_input / invoice_dollars * 100 if invoice_dollars > 0 else 0
 
        if is_flagged:
            st.markdown(f"""
            <div class="result-card-danger">
                <div class="result-label">Risk Assessment</div>
                <div class="result-flag-danger">⚠️ MANUAL APPROVAL REQUIRED</div>
                <div class="result-sublabel">
                    This invoice has been flagged as high-risk by the classifier.<br>
                    Please route to a senior finance reviewer before processing.
                </div>
                <div class="chip-row">
                    <span class="chip">💰 Invoice vs Items: {dollar_diff_pct:.1f}% diff</span>
                    <span class="chip">🚚 Freight: {freight_pct:.1f}% of invoice</span>
                    <span class="chip">🗓 Pay in {days_to_pay}d</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-success">
                <div class="result-label">Risk Assessment</div>
                <div class="result-flag-safe">✅ SAFE FOR AUTO-APPROVAL</div>
                <div class="result-sublabel">
                    No anomalies detected. Invoice passes all risk thresholds.<br>
                    Cleared for automated processing.
                </div>
                <div class="chip-row">
                    <span class="chip">💰 Invoice vs Items: {dollar_diff_pct:.1f}% diff</span>
                    <span class="chip">🚚 Freight: {freight_pct:.1f}% of invoice</span>
                    <span class="chip">🗓 Pay in {days_to_pay}d</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
 