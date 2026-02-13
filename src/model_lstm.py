import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def build_lstm(input_steps):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(lookback, 1)),  # ← ADD THIS
        Dropout(0.2),  # ← ADD
        LSTM(100),  # ← Better than single layer
        Dropout(0.2),  # ← ADD
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')  # ← CHANGE LR
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
    # OLD:
# model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# NEW:
history = model.fit(
    X_train, y_train,
    epochs=500,          # ← 50 → 500
    batch_size=32,
    validation_split=0.1, # ← ADD validation
    verbose=1            # ← See progress
)
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.yscale('log')
plt.savefig('app/loss_plot.png')  # For Streamlit
plt.show()

# Check final loss
print(f"Final Train Loss: {history.history['loss'][-1]:.6f}")
print(f"Final Val Loss:   {history.history['val_loss'][-1]:.6f}")
model.save('app/lstm_model_improved.h5')  # ← Save new model
scaler.save('app/scaler_improved.pkl')

    return model, scaler
