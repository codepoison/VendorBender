import joblib
import pandas as pd
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "predict_flag_invoice.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Must match EXACTLY what train.py used
FEATURE_COLS = [
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "days_po_to_invoice",
    "days_to_pay"
]

def predict_invoice_flag(input_data):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Always build DataFrame directly — no wrapping needed
    input_df = pd.DataFrame(input_data)

    # Fill any missing columns with 0
    for col in FEATURE_COLS:
        if col not in input_df.columns:
            input_df[col] = 0

    X_scaled = scaler.transform(input_df[FEATURE_COLS])
    input_df['Predicted_Flag'] = (model.predict(X_scaled) >= 0.5).astype(int)
    return input_df

if __name__ == "__main__":
    # Test with two very different invoices
    sample_data = {
        "invoice_dollars":    [25000, 50],
        "Freight":            [150.0, 1.5],
        "total_item_quantity":[500,   2],
        "total_item_dollars": [24000, 45],
        "days_po_to_invoice": [30,    1],
        "days_to_pay":        [60,    5]
    }

    results = predict_invoice_flag(sample_data)
    print("\n--- Invoice Flag Predictions ---")
    print(results[['invoice_dollars', 'Freight', 'Predicted_Flag']])