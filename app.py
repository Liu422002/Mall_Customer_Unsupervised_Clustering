import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="centered")

st.title("Mall Customer Segmentation App")
st.write("Enter customer information to predict the customer cluster.")

model_path = "models/kmeans_model.pkl"
scaler_path = "models/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Model files not found. Please run main.py first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.subheader("Customer Input")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income (in $1000)", min_value=0, max_value=200, value=60)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict Cluster"):
    input_data = np.array([[age, annual_income, spending_score]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]

    st.success(f"This customer belongs to Cluster {cluster}.")

    st.subheader("Cluster Centers")
    centers_scaled = model.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    for i, center in enumerate(centers_original):
        st.write(
            f"Cluster {i}: "
            f"Age = {center[0]:.2f}, "
            f"Annual Income = {center[1]:.2f}, "
            f"Spending Score = {center[2]:.2f}"
        )