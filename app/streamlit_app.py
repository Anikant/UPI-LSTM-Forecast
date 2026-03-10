import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast Engine")

# =====================================================
# SIDEBAR
# =====================================================
daily_projection = st.sidebar.toggle(
    "Enable Daily Projection"
)

# =====================================================
# SLIDERS
# =====================================================
if daily_projection:

    forecast_days = st.sidebar.slider(
        "Select Forecast Days",
        7, 180, 30
    )

else:

    forecast_months = st.sidebar.slider(
        "Select Forecast Months",
        1, 24, 6
    )

# =====================================================
# DATA LOADERS
# =====================================================
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


# =====================================================
# HOLIDAYS (same as notebook)
# =====================================================
def get_india_holidays(start, end):

    dates = pd.date_range(start, end)

    holidays = []

    for d in dates:

        if d.month == 10 and d.day in [2, 24]:
            holidays.append(d)

        if d.month == 11 and d.day in [12, 14]:
            holidays.append(d)

    return pd.DataFrame({
        "ds": holidays,
        "holiday": "india_festival"
    })


# =====================================================
# MONTHLY MODE (UNCHANGED LSTM)
# =====================================================
if not daily_projection:

    df = load_monthly_data()

    fields = ["Remitter", "Benificiary", "Total"]

    selected_field = st.sidebar.selectbox(
        "Select Projection Field",
        fields
    )

    series = df[selected_field].resample("M").sum()

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(series.values.reshape(-1,1))

    lookback = 12

    X,y = [],[]

    for i in range(lookback,len(data_scaled)):
        X.append(data_scaled[i-lookback:i])
        y.append(data_scaled[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        LSTM(64,return_sequences=True,input_shape=(lookback,1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam",loss="mse")

    model.fit(X,y,epochs=100,batch_size=8,verbose=0)

    seq = data_scaled[-lookback:]

    preds = []

    for _ in range(forecast_months):

        p = model.predict(seq.reshape(1,lookback,1),verbose=0)[0]

        preds.append(p)

        seq = np.append(seq[1:],p)

    preds = scaler.inverse_transform(preds)

    future_dates = [
        series.index[-1] + pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    future_vals = preds.flatten()

    series_plot = series

# =====================================================
# DAILY MODE (NOTEBOOK PROPHET LOGIC)
# =====================================================
else:

    df = load_daily_data()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    selected_field = st.sidebar.selectbox(
        "Select Daily Field",
        numeric_cols
    )

    model_df = df[["ds",selected_field]].copy()

    model_df.rename(columns={selected_field:"y"},inplace=True)

    holidays = get_india_holidays(
        model_df["ds"].min(),
        model_df["ds"].max()
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays
    )

    model.fit(model_df)

    future = model.make_future_dataframe(
        periods=forecast_days
    )

    forecast = model.predict(future)

    forecast_df = forecast[["ds","yhat"]].tail(forecast_days)

    future_dates = forecast_df["ds"]

    future_vals = forecast_df["yhat"]

    series_plot = model_df.set_index("ds")["y"]

    # ================= MAX DAY =================
    max_idx = forecast_df["yhat"].idxmax()

    max_day = forecast_df.loc[max_idx,"ds"]

    max_val = forecast_df.loc[max_idx,"yhat"]

    st.success(
        f"Maximum Projected Day: {max_day.date()} | Value: {round(max_val,2)}"
    )

# =====================================================
# GRAPH
# =====================================================
fig = go.Figure()

recent = series_plot.last("30D")

fig.add_trace(go.Scatter(
    x=recent.index,
    y=recent.values,
    mode="lines",
    name="Last 30 Days"
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_vals,
    mode="lines+markers",
    name="Forecast"
))

fig.update_layout(
    template="plotly_white",
    height=550
)

st.plotly_chart(fig,use_container_width=True)

# =====================================================
# FORECAST TABLE
# =====================================================
forecast_table = pd.DataFrame({
    "Date":future_dates,
    "Forecast":future_vals
})

st.dataframe(forecast_table)

st.success("Forecast generated successfully.")
