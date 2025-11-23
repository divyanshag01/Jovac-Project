# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("AI-Based Gas Diffusion Layer Performance Classifier")
st.write("Predict the performance category (Low, Medium, High) of GDL based on your parameters.")

# Load model & scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Sidebar inputs
st.sidebar.header("GDL Input Parameters")

porosity = st.sidebar.slider("Porosity", 0.3, 0.9, 0.6, 0.01)
pore_size = st.sidebar.slider("Pore Size (µm)", 1.0, 100.0, 30.0, 1.0)
fiber_angle = st.sidebar.slider("Fiber Arrangement Angle", 0, 90, 45, 1)
wettability = st.sidebar.slider("Wettability Contact Angle", 0.0, 120.0, 60.0, 1.0)

if st.button("Predict Performance Category"):
    # Build input DF
    X = pd.DataFrame([{
        "Porosity": porosity,
        "Pore_Size_um": pore_size,
        "Fiber_Arrangement_Angle": fiber_angle,
        "Wettability_Contact_Angle": wettability
    }])

    # Scale input
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    st.subheader(f"Predicted Performance Category: **{prediction}**")

    if prediction == "Low":
        st.info("This design may perform poorly. Consider optimizing porosity & pore size.")
    elif prediction == "Medium":
        st.warning("Average performance. Adjust fiber angle and wettability.")
    else:
        st.success("High performance design!")

st.markdown("---")
st.caption("Powered by Gaussian Naive Bayes Model")
