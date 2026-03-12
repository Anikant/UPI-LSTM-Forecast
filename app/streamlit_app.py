import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="UPI Forecast Engine", layout="wide")

st.title("UPI Transaction Forecast Engine")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.header("Forecast Settings")

daily_projection = st.sidebar.checkbox("Enable Daily Projection")

if daily_projection:
    forecast_days = st.sidebar.slider(
        "Daily Forecast Horizon (Days)",
        min_value=7,
        max_value=120,
        value=30
    )
else:
    forecast_months = st.sidebar.slider(
        "Monthly Forecast Horizon (Months)",
        min_value=1,
        max_value=24,
        value=6
    )

# ---------------------------------------------------------
# HOLIDAY CALENDAR (from your attached calendar)
# ---------------------------------------------------------

def get_holidays():

    holidays = pd.DataFrame({
        "holiday": [
            "new_year","mlk_day","presidents_day","memorial_day",
            "independence_day","labor_day","columbus_day",
            "thanksgiving","christmas"
        ],
        "ds": pd.to_datetime([
            "2026-01-01",
            "2026-01-19",
            "2026-02-16",
            "2026-05-25",
            "2026-07-04",
            "2026-09-07",
            "2026-10-12",
            "2026-11-26",
            "2026-12-25"
        ]),
        "lower_window":0,
        "upper_window":1
    })

    return holidays

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def load_monthly():

    df = pd.read_excel("data/UPI_Transactions.xlsx")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df


@st.cache_data
def load_daily():

    df = pd.read_excel("data/merged_upi_transactions.xlsx")

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df.rename(columns={"DATE":"ds"}, inplace=True)

    df = df.sort_values("ds")

    return df


# ---------------------------------------------------------
# MONTHLY FORECAST (LSTM)
# ---------------------------------------------------------

if not daily_projection:

    df = load_monthly()

    field = st.sidebar.selectbox(
        "Select Monthly Projection Field",
        df.columns
    )

    series = df[field].resample("M").sum()

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    lookback = 12

    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback,1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=80, batch_size=8, verbose=0)

    seq = scaled[-lookback:]

    preds = []

    for _ in range(forecast_months):

        p = model.predict(seq.reshape(1,lookback,1), verbose=0)[0]

        preds.append(p)

        seq = np.append(seq[1:], p)

    preds = scaler.inverse_transform(preds).flatten()

    # dampening unrealistic growth
    recent_avg = series.tail(6).mean()

    preds = 0.7 * preds + 0.3 * recent_avg

    future_dates = [
        series.index[-1] + pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    future_vals = preds

    history_series = series

# ---------------------------------------------------------
# DAILY FORECAST (PROPHET + LSTM + HOLIDAYS)
# ---------------------------------------------------------

else:

    df = load_daily()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    field = st.sidebar.selectbox(
        "Select Daily Projection Field",
        numeric_cols
    )

    data = df[["ds", field]].rename(columns={field:"y"})

    data["cap"] = data["y"].max() * 1.1
    data["floor"] = data["y"].min() * 0.95

    holidays = get_holidays()

    prophet = Prophet(
        growth="logistic",
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.02,
        seasonality_prior_scale=5
    )

    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=forecast_days)

    future["cap"] = data["cap"].iloc[0]
    future["floor"] = data["floor"].iloc[0]

    forecast = prophet.predict(future)

    prophet_vals = forecast["yhat"].tail(forecast_days).values

    # ---------- LSTM COMPONENT ----------

    series = data["y"]

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    lookback = 30

    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])

    X = np.array(X)
    y = np.array(y)

    lstm = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback,1)),
        LSTM(16),
        Dense(1)
    ])

    lstm.compile(optimizer="adam", loss="mse")

    lstm.fit(X, y, epochs=50, batch_size=16, verbose=0)

    seq = scaled[-lookback:]

    lstm_preds = []

    for _ in range(forecast_days):

        p = lstm.predict(seq.reshape(1,lookback,1), verbose=0)[0]

        lstm_preds.append(p)

        seq = np.append(seq[1:], p)

    lstm_vals = scaler.inverse_transform(lstm_preds).flatten()

    # ---------- ENSEMBLE ----------

    future_vals = 0.65 * prophet_vals + 0.35 * lstm_vals

    future_vals = pd.Series(future_vals).rolling(3, min_periods=1).mean().values

    future_dates = forecast["ds"].tail(forecast_days)

    history_series = data.set_index("ds")["y"]

    # max projected day
    max_idx = np.argmax(future_vals)

    st.write(
        f"Maximum projected day: {future_dates.iloc[max_idx].date()} | "
        f"Value: {round(future_vals[max_idx],2)}"
    )

# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

fig = go.Figure()

recent = history_series.tail(30)

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

# ---------------------------------------------------------
# FORECAST TABLE
# ---------------------------------------------------------

last_actual = history_series.iloc[-1]

growth = ((future_vals - last_actual) / last_actual) * 100

forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_vals,
    "Growth %": growth
})

st.markdown("### Forecast Table")

st.dataframe(forecast_table, use_container_width=True)
