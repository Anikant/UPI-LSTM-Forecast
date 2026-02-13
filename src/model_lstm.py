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

def train_lstm(series, lookback=30, epochs=500):  # epochs=500!
    """Train improved LSTM model"""
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    # Scale RAW series first (fixes earlier scaler issue)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
    # Create supervised data
    X, y = make_supervised(series_scaled, lookback)
    
    # Split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # IMPROVED MODEL
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    
    # Train with validation
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Plot loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()
    
    print(f"✅ Final Loss: {history.history['loss'][-1]:.6f}")
    
    return model, scaler
