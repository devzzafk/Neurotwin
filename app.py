import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analysis import load_data, predict_best_hour

st.title("🧠 NeuroTwin – Cognitive Digital Twin")

st.write("This dashboard analyzes historical focus sessions.")

df = load_data()

if df.empty:
    st.warning("No session data found. Run tracker locally to generate sessions.")
else:
    st.subheader("Focus Trend Over Sessions")

    plt.figure()
    plt.plot(df["focus_score"])
    plt.xlabel("Session")
    plt.ylabel("Focus Score")
    st.pyplot(plt)

    st.subheader("Best Focus Hour Prediction")
    st.write(predict_best_hour(df))
