import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "merged_upi_transactions.xlsx"

def load_series():
    df = pd.read_excel(DATA_PATH)
    # Align with your notebook logic: detect date and txn col if needed[file:5][file:6]
    df = df.rename(columns={
        "DATE": "ds",
        "Total Upi Transaction": "y"
    })
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return df

def make_supervised(df, lookback=30):
    y = df["y"].values.astype("float32")
    X, Y = [], []
    for i in range(lookback, len(y)):
        X.append(y[i-lookback:i])
        Y.append(y[i])
    X = np.array(X)[..., None]  # (samples, lookback, 1)
    Y = np.array(Y)
    return X, Y
