"""
Nike Sales Feature Engineering
Author: Your Name
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_features():
    # Load data with proper date parsing
    df = pd.read_csv("../data/nike_sales.csv")

    # Convert Invoice Date to datetime (handle different possible formats)
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors='coerce')

    # Check if date conversion was successful
    if df["Invoice Date"].isnull().all():
        print("Warning: Could not parse dates. Checking column names...")
        print("Available columns:", df.columns.tolist())
        raise ValueError("Invoice Date column could not be parsed as datetime")

    print(f"Date range: {df['Invoice Date'].min()} to {df['Invoice Date'].max()}")

    # Basic Cleaning
    df = df[df["Units Sold"] > 0]
    df = df[df["Total Sales"] > 0]

    print(f"Data shape after cleaning: {df.shape}")

    # Time-Based Features
    df["Year"] = df["Invoice Date"].dt.year
    df["Month"] = df["Invoice Date"].dt.month
    df["Quarter"] = df["Invoice Date"].dt.quarter
    df["Day"] = df["Invoice Date"].dt.day

    # Monthly Aggregation
    monthly_df = (
        df.groupby(["Year", "Month", "Quarter"])
        .agg({
            "Units Sold": "sum",
            "Total Sales": "sum",
            "Price per Unit": "mean"
        })
        .reset_index()
    )

    print(f"Monthly aggregated data shape: {monthly_df.shape}")
    print(f"Monthly data preview:\n{monthly_df.head()}")

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

    print(f"\nFeature engineering complete.")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    build_features()