import joblib
from pathlib import Path
# Import existing functions from your local modules
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier

# Features identified from your previous database column check
FEATURES = [
    "invoice_dollars",
    "Freight",
    "total_item_quantity", 
    "total_item_dollars",
    "days_po_to_invoice",   
    "days_to_pay"           
]

TARGET = "flag_invoice"

def main():
    # Define the path to the SHARED models folder (one level up from invoice_flagging)
    # This ensures both train.py and predict_freight.py look at the same place.
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)

    # load data
    df = load_invoice_data() 
    print("Columns found in database:", df.columns.tolist())
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    
    # Updated path to save scaler in the main models folder
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, str(output_dir / 'scaler.pkl')
    )

    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Updated path to save model in the main models folder
    model_save_path = output_dir / 'predict_flag_invoice.pkl'
    joblib.dump(grid_search.best_estimator_, model_save_path)
    print(f"Success! Model saved to: {model_save_path}")
    
if __name__ == '__main__':
    main()
