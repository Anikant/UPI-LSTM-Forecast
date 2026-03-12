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

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.header("Forecast Settings")

daily_projection = st.sidebar.checkbox("Enable Daily Projection")

if daily_projection:
    forecast_days = st.sidebar.slider(
        "Daily Forecast Horizon (Days)",
        7,
        120,
        30
    )
else:
    forecast_months = st.sidebar.slider(
        "Monthly Forecast Horizon (Months)",
        1,
        24,
        6
    )

# -----------------------------------------------------
# HOLIDAY DATA
# -----------------------------------------------------

def get_holidays():

    holidays = pd.DataFrame({
        "holiday":[
            "new_year","mlk","presidents","memorial",
            "independence","labor","columbus",
            "thanksgiving","christmas"
        ],
        "ds":pd.to_datetime([
            "2026-01-01",
            "2026-01-19",
            "2026-02-16",
            "2026-05-25",
            "2026-07-04",
            "2026-09-07",
            "2026-10-12",
            "2026-11-26",
            "2026-12-25"
        ])
    })

    return holidays

# -----------------------------------------------------
# OUTLIER FILTER
# -----------------------------------------------------

def remove_outliers(series):

    z = (series - series.mean()) / series.std()

    series[z.abs() > 3] = series.rolling(7).median()

    return series

# -----------------------------------------------------
# DATA LOADING
# -----------------------------------------------------

@st.cache_data
def load_daily():

    df = pd.read_excel("data/merged_upi_transactions.xlsx")

    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"])

    df.rename(columns={"DATE":"ds"}, inplace=True)

    df = df.sort_values("ds")

    return df


@st.cache_data
def load_monthly():

    df = pd.read_excel("data/UPI_Transactions.xlsx")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df.set_index("Date", inplace=True)

    return df

# -----------------------------------------------------
# MONTHLY MODEL
# -----------------------------------------------------

if not daily_projection:

    df = load_monthly()

    field = st.sidebar.selectbox("Projection Field", df.columns)

    series = df[field].resample("M").sum()

    series = remove_outliers(series)

    growth = np.log(series / series.shift(1)).dropna()

    avg_growth = growth.mean()

    future_vals = []

    last_val = series.iloc[-1]

    for _ in range(forecast_months):

        last_val = last_val * (1 + avg_growth*0.7)

        future_vals.append(last_val)

    future_dates = [
        series.index[-1] + pd.DateOffset(months=i+1)
        for i in range(forecast_months)
    ]

    history = series

# -----------------------------------------------------
# DAILY MODEL
# -----------------------------------------------------

else:

    df = load_daily()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    field = st.sidebar.selectbox("Projection Field", numeric_cols)

    data = df[["ds",field]].rename(columns={field:"y"})

    data["y"] = remove_outliers(data["y"])

    # -----------------------
    # PROPHET
    # -----------------------

    prophet = Prophet(
        holidays=get_holidays(),
        changepoint_prior_scale=0.02,
        seasonality_prior_scale=5
    )

    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=forecast_days)

    forecast = prophet.predict(future)

    prophet_vals = forecast["yhat"].tail(forecast_days).values

    # -----------------------
    # LSTM
    # -----------------------

    series = data["y"]

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    lookback = 30

    X,y=[],[]

    for i in range(lookback,len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])

    X=np.array(X)
    y=np.array(y)

    lstm = Sequential([
        LSTM(32,return_sequences=True,input_shape=(lookback,1)),
        LSTM(16),
        Dense(1)
    ])

    lstm.compile(optimizer="adam",loss="mse")

    lstm.fit(X,y,epochs=40,batch_size=16,verbose=0)

    seq=scaled[-lookback:]

    lstm_preds=[]

    for _ in range(forecast_days):

        p=lstm.predict(seq.reshape(1,lookback,1),verbose=0)[0]

        lstm_preds.append(p)

        seq=np.append(seq[1:],p)

    lstm_vals=scaler.inverse_transform(lstm_preds).flatten()

    # -----------------------
    # TREND MODEL
    # -----------------------

    growth = np.log(series / series.shift(1)).dropna()

    avg_growth = growth.mean()

    last_val = series.iloc[-1]

    trend_vals=[]

    for _ in range(forecast_days):

        last_val = last_val * (1 + avg_growth*0.6)

        trend_vals.append(last_val)

    # -----------------------
    # ENSEMBLE
    # -----------------------

    future_vals = (
        0.5*prophet_vals +
        0.3*lstm_vals +
        0.2*np.array(trend_vals)
    )

    future_vals = pd.Series(future_vals).rolling(3,min_periods=1).mean().values

    future_dates = forecast["ds"].tail(forecast_days)

    history = data.set_index("ds")["y"]

# -----------------------------------------------------
# GRAPH
# -----------------------------------------------------

fig = go.Figure()

recent = history.tail(30)

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

# -----------------------------------------------------
# FORECAST TABLE
# -----------------------------------------------------

last_actual = history.iloc[-1]

growth_pct = ((future_vals-last_actual)/last_actual)*100

table = pd.DataFrame({
    "Date":future_dates,
    "Forecast":future_vals,
    "Growth %":growth_pct
})

st.dataframe(table,use_container_width=True)

max_idx = np.argmax(future_vals)

st.write(
    f"Maximum projected day: {future_dates.iloc[max_idx].date()} | "
    f"Value: {round(future_vals[max_idx],2)}"
)
