import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta 
import sys
from pathlib import Path

# Ensure project root is on sys.path when run via `streamlit run app/streamlit_app.py`
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
    """Multi-step forecast from last lookback points of 'history'."""
    hist_scaled = scaler.transform(history.reshape(-1, 1)).flatten()
    context = hist_scaled[-lookback:].copy()
    preds_scaled = []
    for _ in range(horizon):
        x = context[-lookback:][None, :, None]
        y_hat_scaled = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(y_hat_scaled)
        context = np.append(context, y_hat_scaled)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds

# --- App ----------------------------------------------------

st.set_page_config(page_title="UPI LSTM Forecast", layout="wide")

st.title("UPI Daily Transaction Forecast (LSTM)")
st.caption("Trained on merged_upi_transactions.xlsx")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    horizon = st.slider("Forecast days", min_value=7, max_value=365, value=30, step=7)
    lookback = st.slider("LSTM lookback days", min_value=14, max_value=90, value=30, step=7)
    test_days = st.slider("Error window (days)", min_value=14, max_value=60, value=30, step=2)
    epochs = st.slider("Training epochs", min_value=10, max_value=100, value=40, step=10)

# Load data
df = load_series()
st.subheader("Training data (daily UPI total)")
st.line_chart(df.set_index("ds")["y"])

# Train / evaluate
y_all = df["y"].values.astype("float32")

# Train on all but last 'test_days' for error estimation
train_series = y_all[:-test_days]
test_series = y_all[-test_days:]
train_dates = df["ds"].iloc[:-test_days].values
test_dates = df["ds"].iloc[-test_days:].values

model, scaler = train_lstm(train_series, lookback=lookback, epochs=epochs)

# Forecast on test window to compute error
# Build supervised only for train+test to align sequences
hist_for_test = y_all[:-test_days + lookback]
test_preds = rolling_forecast(model, scaler, hist_for_test, test_days, lookback)

mape_val = mape(test_series, test_preds)
rmse_val = rmse(test_series, test_preds)

# Full-horizon forecast from end of series
last_date = df["ds"].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
full_forecast = rolling_forecast(model, scaler, y_all, horizon, lookback)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": full_forecast
})
forecast_df["Forecast (pretty)"] = forecast_df["Forecast"].apply(humanize)

# Max forecast day
max_idx = forecast_df["Forecast"].idxmax()
max_row = forecast_df.loc[max_idx]

# --- Layout: charts and tables -----------------------------

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("History + Forecast")
    chart_df = pd.concat([
        df[["ds", "y"]].rename(columns={"ds": "Date", "y": "Actual"}),
        forecast_df[["Date", "Forecast"]]
    ], axis=0)
    chart_df = chart_df.set_index("Date")
    st.line_chart(chart_df)

with col2:
    st.subheader("Key metrics")
    st.metric(
        "Max projected daily UPI",
        humanize(max_row["Forecast"]),
        help=f"On {max_row['Date'].date()}"
    )
    st.write(f"**Date with maximum projected UPI**: {max_row['Date'].date()}")
    st.write(f"**MAPE (last {test_days} days)**: {mape_val:.2f}%")
    st.write(f"**RMSE (last {test_days} days)**: {rmse_val:,.0f}")

st.subheader("Daily forecast table")
st.dataframe(
    forecast_df.style.format({"Forecast": "{:,.0f}"}),
    use_container_width=True
)
