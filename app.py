import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------
# Page configuration
# ---------------------------------------------
st.set_page_config(page_title="GDL Performance Classifier", layout="centered")

# ---------------------------------------------
# Simple CSS styling
# ---------------------------------------------
st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            font-size: 15px;
            color: #555;
            margin-bottom: 35px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-top: 20px;
        }
        .prediction-box {
            padding: 12px;
            border-radius: 8px;
            font-size: 22px;
            font-weight: 600;
            text-align: center;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">GDL Performance Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate fuel-cell performance from microstructure parameters</div>', unsafe_allow_html=True)

# ---------------------------------------------
# Load model & scaler
# ---------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------
# Sidebar: input controls
# ---------------------------------------------
st.sidebar.header("Input Parameters")

porosity = st.sidebar.slider("Porosity", 0.3, 0.9, 0.65, step=0.01)
pore_size = st.sidebar.slider("Pore Size (µm)", 1.0, 100.0, 35.0, step=1.0)
fiber_angle = st.sidebar.slider("Fiber Arrangement Angle (°)", 0, 90, 45, step=1)
wettability = st.sidebar.slider("Wettability Contact Angle (°)", 0.0, 120.0, 60.0, step=1.0)

# ---------------------------------------------
# Prediction
# ---------------------------------------------
if st.button("Predict Performance", use_container_width=True):

    X = pd.DataFrame([{
        "Porosity": porosity,
        "Pore_Size_um": pore_size,
        "Fiber_Arrangement_Angle": fiber_angle,
        "Wettability_Contact_Angle": wettability
    }])

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    class_order = list(model.classes_)
    probability_map = {cls: probabilities[class_order.index(cls)] for cls in class_order}

    # Color selection for classes
    color_map = {
        "Low": "#d9534f",
        "Medium": "#f0ad4e",
        "High": "#5cb85c"
    }

    st.markdown(
        f'<div class="prediction-box" style="background-color:{color_map[prediction]}; color:white;">'
        f"Predicted Class: {prediction}</div>",
        unsafe_allow_html=True
    )

    # Probability graph
    st.markdown('<div class="section-title">Class Probability Distribution</div>', unsafe_allow_html=True)
    fig = px.bar(
        x=list(probability_map.keys()),
        y=list(probability_map.values()),
        labels={"x": "Class", "y": "Probability"},
        color=list(probability_map.keys()),
        color_discrete_map=color_map,
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart for parameter profile
    st.markdown('<div class="section-title">Parameter Profile</div>', unsafe_allow_html=True)

    categories = ["Porosity", "Pore Size", "Fiber Angle", "Wettability"]
    values = [porosity, pore_size, fiber_angle, wettability]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color="#3b7dd8")
    ))

    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=400
    )
    st.plotly_chart(radar, use_container_width=True)

# ---------------------------------------------
# Recommended Ranges
# ---------------------------------------------
st.markdown('<div class="section-title">Recommended High-Performance Ranges</div>', unsafe_allow_html=True)

st.info("""
These ranges most commonly lead to high performance:

- Porosity: **0.70 – 0.78**  
- Pore Size: **30 – 45 µm**  
- Fiber Angle: **38° – 50°**  
- Wettability: **50° – 75°**  

Setting values within these ranges typically increases the probability of the **High** category.
""")
