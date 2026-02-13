import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def build_lstm(input_steps):
    model = Sequential([
        LSTM(64, input_shape=(input_steps, 1), activation="tanh"),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_lstm(y_series, lookback=30, val_fraction=0.1, epochs=30, batch_size=32):
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_series.reshape(-1, 1)).flatten()

    X, Y = [], []
    for i in range(lookback, len(y_scaled)):
        X.append(y_scaled[i-lookback:i])
        Y.append(y_scaled[i])
    X = np.array(X)[..., None]
    Y = np.array(Y)

    split = int(len(X) * (1 - val_fraction))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    model = build_lstm(lookback)
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model, scaler
