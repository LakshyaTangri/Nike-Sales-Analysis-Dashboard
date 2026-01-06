"""
Nike Sales Neural Network Forecasting
Author: LT
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from notebooks.feature_engineering import build_features


def train_model():
    X_train, X_test, y_train, y_test = build_features()

    # Model Architecture
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Training
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluation
    predictions = model.predict(X_test)

    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Monthly Sales")
    plt.tight_layout()
    plt.show()

    # Save Model
    model.save("../models/nn_sales_forecast.h5")
    print("Model saved to models/nn_sales_forecast.h5")


if __name__ == "__main__":
    train_model()
