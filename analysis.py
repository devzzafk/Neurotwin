import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

DATA_PATH = "data/sessions.csv"

def save_session(data):
    data["timestamp"] = datetime.now()
    df = pd.DataFrame([data])

    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

def predict_best_hour(df):
    if len(df) < 5:
        return "Not enough data"

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour"]]
    y = df["focus_score"]

    model = LinearRegression()
    model.fit(X, y)

    hours = np.arange(0, 24).reshape(-1, 1)
    predictions = model.predict(hours)

    best_hour = hours[np.argmax(predictions)][0]
    return f"Best predicted focus hour: {best_hour}:00"
