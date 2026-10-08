import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_invoice_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH  = os.path.join(BASE_DIR, "data", "inventory.db")
    conn = sqlite3.connect(DB_PATH)
    query = """
    WITH purchase_agg AS (
        SELECT
             p.PONumber,
             COUNT(DISTINCT p.Brand) AS total_brands,
             SUM(p.Quantity) AS total_item_quantity,
             SUM(p.Dollars) AS total_item_dollars,
             AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    # Flag if invoice dollars differ from item dollars by more than 20%
    if row["total_item_dollars"] > 0:
        dollar_diff_pct = abs(row["invoice_dollars"] - row["total_item_dollars"]) / row["total_item_dollars"]
        if dollar_diff_pct > 0.20:
            return 1

    # Flag if receiving delay is unusually high (over 30 days)
    if row["avg_receiving_delay"] > 30:
        return 1

    # Flag if freight is disproportionately high vs invoice value
    if row["invoice_dollars"] > 0:
        freight_pct = row["Freight"] / row["invoice_dollars"]
        if freight_pct > 0.20:   # freight > 20% of invoice value is suspicious
            return 1

    # Flag if payment terms are extremely delayed
    if row["days_to_pay"] > 90:
        return 1

    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )

def scale_features(X_train, X_test, scaler_path):
    # FIXES:
    # 1. Changed argument 'X,test' to 'X_test'.
    # 2. Used 'scaler_path' variable in joblib.dump.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled
