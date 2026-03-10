import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast Engine")

st.sidebar.header("Forecast Settings")

daily_projection = st.sidebar.checkbox(
    "Enable Daily Projection"
)

if daily_projection:

    forecast_days = st.sidebar.slider(
        "Daily Forecast Horizon (Days)",
        7,
        180,
        30
    )

else:

    forecast_months = st.sidebar.slider(
        "Monthly Forecast Horizon (Months)",
        1,
        24,
        6
    )


@st.cache_data
def load_monthly_data():

    path = "data/UPI_Transactions.xlsx"

    df = pd.read_excel(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


@st.cache_data
def load_daily_data():

    path = "data/merged_upi_transactions.xlsx"

    df = pd.read_excel(path)

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df = df.sort_values("DATE")

    df.rename(columns={"DATE": "ds"}, inplace=True)

    return df


# ==============================
# MONTHLY MODE (LSTM)
# ==============================
if not daily_projection:

    df = load_monthly_data()

    fields = ["Remitter", "Benificiary", "Total"]

    selected_field = st.sidebar.selectbox(
        "Select Monthly Projection Field",
        fields
    )

    series = df[selected_field].resample("M").sum()

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(series.values.reshape(-1,1))

    lookback = 12

    X, y = [], []

    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i])
        y.append(data_scaled[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback,1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=100, batch_size=8, verbose=0)

    seq = data_scaled[-lookback:]

    preds = []

    for _ in range(forecast_months):

        p = model.predict(seq.reshape(1,lookback,1), verbose=0)[0]

        preds.append(p)

        seq = np.append(seq[1:], p)

    preds = scaler.inverse_transform(preds)

    future_dates = [
        series.index[-1] + pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    future_vals = preds.flatten()

    series_plot = series


# ==============================
# DAILY MODE (IMPROVED PROPHET)
# ==============================
else:

    df = load_daily_data()

    invalid_cols = ["month", "year", "day"]

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c.lower() not in invalid_cols
    ]

    selected_field = st.sidebar.selectbox(
        "Select Daily Projection Field",
        numeric_cols
    )

    model_df = df[["ds", selected_field]].copy()

    model_df.rename(columns={selected_field: "y"}, inplace=True)

    # capacity limit to avoid aggressive growth
    model_df["cap"] = model_df["y"].max() * 1.2

    model = Prophet(
        growth="logistic",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )

    # monthly payment seasonality
    model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5
    )

    model.fit(model_df)

    future = model.make_future_dataframe(
        periods=forecast_days
    )

    future["cap"] = model_df["cap"].iloc[0]

    forecast = model.predict(future)

    forecast_df = forecast[["ds","yhat"]].tail(forecast_days)

    future_dates = forecast_df["ds"]

    future_vals = forecast_df["yhat"]

    series_plot = model_df.set_index("ds")["y"]

    max_idx = forecast_df["yhat"].idxmax()

    max_day = forecast_df.loc[max_idx,"ds"]

    max_val = forecast_df.loc[max_idx,"yhat"]

    st.write(
        f"Maximum Projected Day: {max_day.date()} | Value: {round(max_val,2)}"
    )


# ==============================
# GRAPH
# ==============================
fig = go.Figure()

recent = series_plot.tail(30)

fig.add_trace(go.Scatter(
    x=recent.index,
    y=recent.values,
    mode="lines",
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_vals,
    mode="lines+markers",
    name="Forecast"
))

fig.update_layout(
    template="plotly_white",
    height=550,
    xaxis_title="Date",
    yaxis_title="Transactions"
)

st.plotly_chart(fig, use_container_width=True)


# ==============================
# FORECAST TABLE WITH GROWTH %
# ==============================
last_actual = series_plot.iloc[-1]

growth_pct = ((np.array(future_vals) - last_actual) / last_actual) * 100

forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_vals,
    "Growth %": growth_pct
})

st.markdown("### Forecast Table")

st.dataframe(forecast_table, use_container_width=True)

st.success("Forecast generated successfully.")
