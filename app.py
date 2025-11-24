import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------
# Page Config
# ---------------------------------------------
st.set_page_config(page_title="GDL Performance Classifier", layout="centered")

# ---------------------------------------------
# CSS Styling
# ---------------------------------------------
st.markdown("""
<style>
    .title { font-size: 32px; font-weight: 700; text-align: center; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 15px; color: #555; margin-bottom: 35px; }
    .section-title { font-size: 20px; font-weight: 600; margin-top: 20px; margin-bottom: 8px; }
    .prediction-box {
        padding: 12px; border-radius: 8px; font-size: 22px;
        font-weight: 600; text-align: center; margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">GDL Performance Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict performance from GDL microstructure parameters</div>', unsafe_allow_html=True)

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
# Optimal Parameter Finder
# ---------------------------------------------
def find_optimal_parameters(model, scaler):
    """
    Searches across a wide parameter grid to find the combination
    that yields the highest probability for the 'High' class.
    """

    porosity_range = np.linspace(0.3, 0.9, 25)
    pore_range = np.linspace(1, 100, 25)
    angle_range = np.linspace(0, 90, 25)
    wet_range = np.linspace(0, 120, 25)

    best_prob = -1
    best_params = None

    for p in porosity_range:
        for ps in pore_range:
            for a in angle_range:
                for w in wet_range:

                    X = pd.DataFrame([{
                        "Porosity": p,
                        "Pore_Size_um": ps,
                        "Fiber_Arrangement_Angle": a,
                        "Wettability_Contact_Angle": w
                    }])

                    X_scaled = scaler.transform(X)
                    probabilities = model.predict_proba(X_scaled)[0]

                    idx = list(model.classes_).index("High")
                    high_prob = probabilities[idx]

                    if high_prob > best_prob:
                        best_prob = high_prob
                        best_params = (p, ps, a, w)

    return best_params, best_prob

# ---------------------------------------------
# INPUT MODE SELECTOR
# ---------------------------------------------
st.sidebar.header("Input Method")
input_mode = st.sidebar.radio(
    "Choose input mode:",
    ["Sliders", "Custom Input"]
)

# ---------------------------------------------
# INPUT SECTION
# ---------------------------------------------
st.sidebar.header("Parameters")

if input_mode == "Sliders":
    porosity = st.sidebar.slider("Porosity", 0.3, 0.9, 0.65, step=0.01)
    pore_size = st.sidebar.slider("Pore Size (µm)", 1.0, 100.0, 35.0, step=1.0)
    fiber_angle = st.sidebar.slider("Fiber Arrangement Angle (°)", 0, 90, 45, step=1)
    wettability = st.sidebar.slider("Wettettability Contact Angle (°)", 0.0, 120.0, 60.0, step=1.0)

else:  # Custom manual entry
    porosity = st.sidebar.number_input("Porosity (0.3 - 0.9)", 0.3, 0.9, 0.65, step=0.001, format="%.3f")
    pore_size = st.sidebar.number_input("Pore Size (µm)", 1.0, 100.0, 35.0, step=0.5)
    fiber_angle = st.sidebar.number_input("Fiber Arrangement Angle (°)", 0, 90, 45, step=0.5)
    wettability = st.sidebar.number_input("Wettability Contact Angle (°)", 0.0, 120.0, 60.0, step=0.5)

# ---------------------------------------------
# PREDICTION
# ---------------------------------------------
if st.button("Predict Performance", use_container_width=True):

    X = pd.DataFrame([{
        "Porosity": porosity,
        "Pore_Size_um": pore_size,
        "Fiber_Arrangement_Angle": fiber_angle,
        "Wettability_Contact_Angle": wettability
    }])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    class_order = list(model.classes_)
    prob_map = {cls: probs[class_order.index(cls)] for cls in class_order}

    color_map = {"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}

    st.markdown(
        f'<div class="prediction-box" style="background-color:{color_map[pred]}; color:white;">'
        f"Predicted Class: {pred}</div>",
        unsafe_allow_html=True
    )

    # Probability Chart
    st.markdown('<div class="section-title">Class Probability</div>', unsafe_allow_html=True)
    fig = px.bar(
        x=list(prob_map.keys()),
        y=list(prob_map.values()),
        labels={"x": "Class", "y": "Probability"},
        color=list(prob_map.keys()),
        color_discrete_map=color_map,
        height=330
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar Chart
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
    radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=400)
    st.plotly_chart(radar, use_container_width=True)

# ---------------------------------------------
# OPTIMAL PARAMETER FINDER BUTTON
# ---------------------------------------------
st.markdown('<div class="section-title">Optimal Parameter Finder</div>', unsafe_allow_html=True)

if st.button("Find Optimal Parameters", use_container_width=True):

    with st.spinner("Searching for best-performing configuration..."):
        best_params, best_prob = find_optimal_parameters(model, scaler)

    p, ps, a, w = best_params

    st.success(f"Highest probability for 'High' class: {best_prob:.4f}")

    st.write("### Best Parameter Set Found:")
    st.write(f"- Porosity: **{p:.3f}**")
    st.write(f"- Pore Size: **{ps:.2f} µm**")
    st.write(f"- Fiber Angle: **{a:.1f}°**")
    st.write(f"- Wettability: **{w:.1f}°**")

    # Radar chart for optimal parameters
    categories = ["Porosity", "Pore Size", "Fiber Angle", "Wettability"]
    values = [p, ps, a, w]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color="#2a72d6")
    ))
    radar.update_layout(showlegend=False, height=400)
    st.plotly_chart(radar, use_container_width=True)

# ---------------------------------------------
# RECOMMENDED RANGES
# ---------------------------------------------
st.markdown('<div class="section-title">Recommended High-Performance Ranges</div>', unsafe_allow_html=True)

st.info("""
Best observed ranges for high performance:

- Porosity: **0.70 – 0.78**  
- Pore Size: **30 – 45 µm**  
- Fiber Angle: **38° – 50°**  
- Wettability: **50° – 75°**

Values in this region generally yield the highest probability of the "High" class.
""")
