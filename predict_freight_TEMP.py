import joblib
import pandas as pd
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "predict_freight_cost.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "freight_scaler.pkl")

FEATURE_COLS = [
    "invoice_dollars",
    "days_po_to_invoice",
    "days_to_pay",
    "total_brands",
    "total_item_quantity",
    "total_item_dollars"
]

def predict_freight_cost(input_data: dict) -> pd.DataFrame:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.DataFrame(input_data)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X_scaled = scaler.transform(df[FEATURE_COLS])
    df['Predicted_Freight_Amt'] = model.predict(X_scaled).round(2)

    return df

if __name__ == "__main__":
    sample_data = {
        "invoice_dollars":     [25000, 500],
        "days_po_to_invoice":  [5,     2],
        "days_to_pay":         [30,    10],
        "total_brands":        [5,     1],
        "total_item_quantity": [50,    1],
        "total_item_dollars":  [24000, 450]
    }

    results = predict_freight_cost(sample_data)
    print("\n--- Freight Cost Predictions ---")
    print(results[['invoice_dollars', 'total_item_dollars', 'Predicted_Freight_Amt']])