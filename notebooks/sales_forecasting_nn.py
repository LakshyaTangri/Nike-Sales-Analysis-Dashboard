"""
Nike Sales Neural Network Forecasting - FIXED VERSION
Author: Your Name
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


def build_features_fixed():
    """Fixed version with proper scaling"""
    df = pd.read_csv("../data/nike_sales.csv")
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors='coerce')

    print(f"Date range: {df['Invoice Date'].min()} to {df['Invoice Date'].max()}")

    # Basic Cleaning
    df = df[df["Units Sold"] > 0]
    df = df[df["Total Sales"] > 0]

    print(f"Data shape after cleaning: {df.shape}")

    # Time-Based Features
    df["Year"] = df["Invoice Date"].dt.year
    df["Month"] = df["Invoice Date"].dt.month
    df["Quarter"] = df["Invoice Date"].dt.quarter

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

    print(f"\nMonthly aggregated data shape: {monthly_df.shape}")
    print(f"Sales range: ${monthly_df['Total Sales'].min():,.2f} to ${monthly_df['Total Sales'].max():,.2f}")

    # IMPORTANT: Scale features and target separately
    feature_cols = ["Units Sold", "Price per Unit", "Month", "Quarter"]
    X = monthly_df[feature_cols].values
    y = monthly_df["Total Sales"].values

    # Scale features (X)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale target (y) - CRITICAL for neural networks
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train/Test Split
    split = int(len(monthly_df) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    # Keep original y values for plotting
    y_test_original = y[split:]

    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler_y, y_test_original


def train_model():
    X_train, X_test, y_train, y_test, scaler_y, y_test_original = build_features_fixed()

    # Improved Model Architecture
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1)  # Linear activation for regression
    ])

    # Better optimizer and learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mse",
        metrics=["mae"]
    )

    model.summary()

    # Training with early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        callbacks=[early_stop],
        verbose=0  # Cleaner output
    )

    print(f"✓ Training completed after {len(history.history['loss'])} epochs")

    # Predictions (scaled)
    predictions_scaled = model.predict(X_test, verbose=0)

    # IMPORTANT: Transform predictions back to original scale
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test_original, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
    r2 = r2_score(y_test_original, predictions)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE):  ${mae:,.2f}")
    print(f"Root Mean Squared Error:     ${rmse:,.2f}")
    print(f"R² Score:                    {r2:.4f}")
    print("=" * 60)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training History - Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss During Training', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted
    axes[0, 1].plot(range(len(y_test_original)), y_test_original,
                    label="Actual", marker='o', linewidth=2, markersize=8)
    axes[0, 1].plot(range(len(predictions)), predictions,
                    label="Predicted", marker='s', linewidth=2, markersize=8, alpha=0.7)
    axes[0, 1].set_title("Actual vs Predicted Monthly Sales", fontweight='bold')
    axes[0, 1].set_xlabel("Time Period")
    axes[0, 1].set_ylabel("Total Sales ($)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # Plot 3: Scatter Plot (Perfect Prediction Line)
    axes[1, 0].scatter(y_test_original, predictions, alpha=0.6, s=100)
    min_val = min(y_test_original.min(), predictions.min())
    max_val = max(y_test_original.max(), predictions.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_title('Prediction Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Actual Sales ($)')
    axes[1, 0].set_ylabel('Predicted Sales ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Residuals
    residuals = y_test_original - predictions.ravel()
    axes[1, 1].bar(range(len(residuals)), residuals, color='coral', alpha=0.6)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_title('Prediction Errors (Residuals)', fontweight='bold')
    axes[1, 1].set_xlabel('Time Period')
    axes[1, 1].set_ylabel('Error ($)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../models/forecast_results_fixed.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved to: models/forecast_results_fixed.png")
    plt.show()

    # Save Model
    model.save("../models/nn_sales_forecast_fixed.keras")
    print("✓ Model saved to: models/nn_sales_forecast_fixed.keras")

    # Print sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS:")
    print("=" * 60)
    for i, (actual, pred) in enumerate(zip(y_test_original, predictions.ravel())):
        error_pct = abs(actual - pred) / actual * 100
        print(f"Period {i+1}: Actual=${actual:,.2f}, Predicted=${pred:,.2f}, Error={error_pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    train_model()