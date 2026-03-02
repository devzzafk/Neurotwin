import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tracker import run_tracker
from analysis import save_session, load_data, predict_best_hour

st.title("🧠 NeuroTwin – Cognitive Digital Twin")

if st.button("Start 30s Focus Session"):
    st.write("Tracking...")
    data = run_tracker(30)
    save_session(data)
    st.success("Session Saved!")
    st.write(data)

df = load_data()

if not df.empty:
    st.subheader("Focus Trend")
    plt.figure()
    plt.plot(df["focus_score"])
    plt.xlabel("Session")
    plt.ylabel("Focus Score")
    st.pyplot(plt)

    st.subheader("Prediction")
    st.write(predict_best_hour(df))
