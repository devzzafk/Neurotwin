import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("NeuroTwin – Cognitive Digital Twin Dashboard")

df = pd.read_csv("data/sessions.csv")

if df.empty:
    st.write("No session data found.")
else:
    st.subheader("Focus Score Trend")
    plt.plot(df["focus_score"])
    plt.xlabel("Session Index")
    plt.ylabel("Focus Score")
    st.pyplot()

    st.subheader("Predicted Best Focus Hour")
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour"]]
    y = df["focus_score"]
    model = LinearRegression().fit(X, y)
    hours = np.arange(0, 24).reshape(-1, 1)
    predictions = model.predict(hours)
    best = hours[np.argmax(predictions)][0]
    st.write(f"Best predicted focus hour: {best}:00")
