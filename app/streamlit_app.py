import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta 
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_series, make_supervised
from src.model_lstm import train_lstm
from src.evaluation import mape, rmse

# --- Helpers ------------------------------------------------
def humanize(n: float) -> str:
    if n >= 1e7:
        return f"{n / 1e7:.2f} Cr"
    if n >= 1e5:
        return f"{n / 1e5:.2f} L"
    if n >= 1e3:
        return f"{n / 1e3:.2f} K"
    return f"{int(n)}"

def rolling_forecast(model, scaler, history, horizon, lookback):
    """FIXED - Bulletproof version"""
    history = np.asarray(history).flatten().astype(np.float64)
    history = history[~np.isnan(history)]
    
    if len(history) < lookback:
        history = np.pad(history, (0, lookback - len(history)), 'edge')
    
    # Scale last window (scaler expects (n_samples, 1))
    window = history[-lookback:].reshape(-1, 1)
    context = scaler.transform(window).flatten()
    
    preds = []
    for _ in range(horizon):
        x = context[-lookback:].reshape(1, lookback, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        context = np.append(context, pred)
    
    preds_array = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds_array).flatten()

# --- App ----------------------------------------------------
st.set_page_config(page_title="UPI LSTM Forecast", layout="wide")

st.title("UPI Daily Transaction Forecast (LSTM) 🧠")
st.caption("Trained on merged_upi_transactions.xlsx | Improved model")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    horizon = st.slider("Forecast days", min_value=7, max_value=365, value=30, step=7)
    lookback = st.slider("LSTM lookback days", min_value=14, max_value=90, value=30, step=7)
    test_days = st.slider("Test window (days)", min_value=14, max_value=60, value=30, step=2)
    epochs = st.slider("Training epochs", min_value=100, max_value=1000, value=500, step=50)

# Load data
@st.cache_data
def load_cached():
    return load_series()

df = load_cached()
st.subheader("📈 Training data")
st.line_chart(df.set_index("ds")["y"])

# Prepare series
y_all = df["y"].values.astype(np.float64)
train_series = y_all[:-test_days]
test_series = y_all[-test_days:]
train_dates = df["ds"].iloc[:-test_days].values
test_dates = df["ds"].iloc[-test_days:].values

# Train model
with st.spinner(f"Training LSTM (lookback={lookback}, epochs={epochs})..."):
    model, scaler = train_lstm(train_series, lookback=lookback, epochs=epochs)

# FIXED: hist_for_test creation + forecast
hist_for_test = y_all[-lookback*2:-test_days]  # Last 60 days of training
test_preds = rolling_forecast(model, scaler, hist_for_test, test_days, lookback)

# Metrics
mape_val = mape(test_series, test_preds)
rmse_val = rmse(test_series, test_preds)

# Full forecast
last_date = df["ds"].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
full_forecast = rolling_forecast(model, scaler, y_all, horizon, lookback)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": full_forecast
})
forecast_df["Formatted"] = forecast_df["Forecast"].apply(humanize)

# Max projection
max_row = forecast_df.loc[forecast_df["Forecast"].idxmax()]

# --- Results Layout -----------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 History + Forecast")
    chart_df = pd.concat([
        pd.DataFrame({"Date": df["ds"], "Actual": df["y"]}),
        forecast_df[["Date", "Forecast"]]
    ])
    chart_df.set_index("Date", inplace=True)
    st.line_chart(chart_df, use_container_width=True)

with col2:
    st.subheader("🎯 Key Metrics")
    st.metric("Max daily UPI", max_row["Formatted"], 
              f"on {max_row['Date'].date()}")
    st.metric("Test MAPE", f"{mape_val:.1f}%")
    st.metric("Test RMSE", f"{rmse_val:,.0f}")

st.subheader("📋 Daily Forecast")
st.dataframe(forecast_df[["Date", "Formatted"]], use_container_width=True)
