"""
Nike Sales Feature Engineering
Author: Your Name
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "nike_sales.csv"

def build_features():
    df = pd.read_csv(DATA_PATH, parse_dates=["Invoice Date"])

    # Basic Cleaning
    df = df[df["Units Sold"] > 0]
    df = df[df["Total Sales"] > 0]

    # Time-Based Features
    df["Year"] = df["Invoice Date"].dt.year
    df["Month"] = df["Invoice Date"].dt.month
    df["Quarter"] = df["Invoice Date"].dt.quarter
    df["Day"] = df["Invoice Date"].dt.day

    # Monthly Aggregation
    monthly_df = (
        df.groupby(["Year", "Month"])
        .agg({
            "Units Sold": "sum",
            "Total Sales": "sum",
            "Price per Unit": "mean"
        })
        .reset_index()
    )

    # Scaling
    scaler = MinMaxScaler()
    features = ["Units Sold", "Price per Unit", "Month", "Quarter"]
    monthly_df[features] = scaler.fit_transform(monthly_df[features])

    # Train/Test Split
    X = monthly_df[features]
    y = monthly_df["Total Sales"]

    split = int(len(monthly_df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Feature engineering complete.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    build_features()
