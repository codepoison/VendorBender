import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"

def load_model(model_path: str = MODEL_PATH):
    """Load trained classifier model."""
    # joblib handles the file stream internally
    return joblib.load(model_path)

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    input_data: dict or list of dicts
    """
    model = load_model()
    
    # Ensures a single dict is treated as one row
    if isinstance(input_data, dict):
        input_data = [input_data]
        
    input_df = pd.DataFrame(input_data)
    
    # Predict and add column
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
  
