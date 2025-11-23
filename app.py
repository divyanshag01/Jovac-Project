import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------
# UI Styling
# ---------------------------------------------
st.set_page_config(page_title="GDL Performance Predictor", layout="centered")

st.markdown("""
    <style>
        .big-title {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtext {
            text-align: center;
            font-size: 16px;
            color: #888;
            margin-bottom: 30px;
        }
        .result-chip {
            font-size: 24px;
            font-weight: 700;
            padding: 10px 25px;
            border-radius: 12px;
            display: inline-block;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">⚡ AI-Based GDL Performance Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Predict fuel-cell performance using porosity & microstructure parameters</div>', unsafe_allow_html=True)

# ---------------------------------------------
# Load Model + Scaler
# ---------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------
# Sidebar Inputs
# ---------------------------------------------
st.sidebar.header("🔧 Configure GDL Parameters")

porosity = st.sidebar.slider("Porosity", 0.3, 0.9, 0.6)
pore_size = st.sidebar.slider("Pore Size (µm)", 1.0, 100.0, 35.0)
fiber_angle = st.sidebar.slider("Fiber Arrangement Angle (°)", 0, 90, 45)
wettability = st.sidebar.slider("Wettability Contact Angle (°)", 0.0, 120.0, 60.0)

# ---------------------------------------------
# Predict Button
# ---------------------------------------------
if st.button("🔍 Predict Performance", use_container_width=True):

    X = pd.DataFrame([{
        "Porosity": porosity,
        "Pore_Size_um": pore_size,
        "Fiber_Arrangement_Angle": fiber_angle,
        "Wettability_Contact_Angle": wettability
    }])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    # Get prediction probabilities
    probs = model.predict_proba(X_scaled)[0]
    prob_map = {
        "Low": probs[list(model.classes_).index("Low")],
        "Medium": probs[list(model.classes_).index("Medium")],
        "High": probs[list(model.classes_).index("High")]
    }

    # Color-coded prediction chip
    if pred == "Low":
        color = "#ff4d4d"
    elif pred == "Medium":
        color = "#ffa31a"
    else:
        color = "#28a745"

    st.markdown(
        f"<div class='result-chip' style='background-color:{color}; color:white;'>Predicted: {pred}</div>",
        unsafe_allow_html=True
    )

    # Probability bars
    st.subheader("📊 Class Probability")
    fig = px.bar(
        x=list(prob_map.keys()),
        y=list(prob_map.values()),
        labels={"x": "Category", "y": "Probability"},
        color=list(prob_map.keys()),
        color_discrete_map={"Low": "red", "Medium": "orange", "High": "green"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart for parameters
    st.subheader("📡 Input Parameter Profile")
    radar_fig = go.Figure()

    categories = ["Porosity", "Pore Size", "Fiber Angle", "Wettability"]
    values = [porosity, pore_size, fiber_angle, wettability]

    radar_fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Parameters'
    ))

    st.plotly_chart(radar_fig, use_container_width=True)

# ---------------------------------------------
# Recommended Ranges
# ---------------------------------------------
st.markdown("### ⭐ Recommended Best-Performance Ranges")
st.info("""
Based on training data and Naive Bayes probability trends,  
**High performance GDLs generally follow:**

- **Porosity:** 0.65 – 0.80  
- **Pore Size:** 25 – 55 µm  
- **Fiber Angle:** 35° – 55°  
- **Wettability:** 40° – 80°  

These ranges correlate strongly with *High* performance outcomes.
""")
