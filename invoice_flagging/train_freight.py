import joblib
from pathlib import Path
from data_preprocessing import load_invoice_data, split_data, scale_features
from modeling_evaluation import train_random_forest, evaluate_classifier
FEATURES = [
    "invoice_dollars",
    "days_po_to_invoice",
    "days_to_pay",
    "total_brands",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "Freight"  # target only, NOT in FEATURES anymore

def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)

    df = load_invoice_data()
    print("Columns found:", df.columns.tolist())
    df = df.dropna(subset=[TARGET])

    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, str(output_dir / 'freight_scaler.pkl')  # separate scaler!
    )

    # ↓ use actual function name
    grid_search = train_random_forest(X_train_scaled, y_train)

    # ↓ use actual function name
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Regressor"
    )

    joblib.dump(grid_search.best_estimator_, output_dir / 'predict_freight_cost.pkl')
    print(f"Model saved to: {output_dir / 'predict_freight_cost.pkl'}")

if __name__ == '__main__':
    main()