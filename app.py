import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
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
    .subtitle { text-align: center; font-size: 15px; color: #bbb; margin-bottom: 35px; }
    .section-title { font-size: 20px; font-weight: 600; margin-top: 20px; margin-bottom: 8px; }
    .prediction-box {
        padding: 12px; border-radius: 8px; font-size: 22px;
        font-weight: 600; text-align: center; margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Title ----
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
# FAST OPTIMAL PARAMETER FINDER
# ---------------------------------------------
def find_optimal_parameters_fast(model, scaler, samples=1000):
    best_prob = -1
    best_params = None

    for _ in range(samples):
        p = random.uniform(0.3, 0.9)
        ps = random.uniform(1, 100)
        a = random.uniform(0, 90)
        w = random.uniform(0, 120)

        X = pd.DataFrame([{
            "Porosity": p,
            "Pore_Size_um": ps,
            "Fiber_Arrangement_Angle": a,
            "Wettability_Contact_Angle": w
        }])

        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]

        idx = list(model.classes_).index("High")
        high_prob = probs[idx]

        if high_prob > best_prob:
            best_prob = high_prob
            best_params = (p, ps, a, w)

    return best_params, best_prob


# --------------------------------------------------------------
# SIDEBAR — SEGMENTED CONTROL TOGGLE
# --------------------------------------------------------------
st.sidebar.markdown("### Input Mode")

input_mode = st.sidebar.segmented_control(
    "Select Input Type",
    options=["Sliders", "Custom Input"],
    default="Sliders"
)

# --------------------------------------------------------------
# PARAMETER INPUT SECTION
# --------------------------------------------------------------
st.sidebar.header("Parameters")

if input_mode == "Sliders":
    porosity = st.sidebar.slider("Porosity", 0.3, 0.9, 0.65, step=0.01)
    pore_size = st.sidebar.slider("Pore Size (µm)", 1.0, 100.0, 35.0, step=1.0)
    fiber_angle = st.sidebar.slider("Fiber Arrangement Angle (°)", 0, 90, 45, step=1)
    wettability = st.sidebar.slider("Wettability Contact Angle (°)", 0.0, 120.0, 60.0, step=1.0)

else:
    porosity = st.sidebar.number_input(
        "Porosity (0.3 - 0.9)",
        min_value=0.3, max_value=0.9, value=0.65,
        step=0.001, format="%.3f"
    )
    pore_size = st.sidebar.number_input(
        "Pore Size (µm)",
        min_value=1.0, max_value=100.0, value=35.0,
        step=0.5
    )
    fiber_angle = st.sidebar.number_input(
        "Fiber Arrangement Angle (°)",
        min_value=0.0, max_value=90.0, value=45.0,
        step=0.5
    )
    wettability = st.sidebar.number_input(
        "Wettability Contact Angle (°)",
        min_value=0.0, max_value=120.0, value=60.0,
        step=0.5
    )


# --------------------------------------------------------------
# MAIN PAGE — PREDICTION
# --------------------------------------------------------------
st.markdown("### Performance Prediction")

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
        f"""
        <div style="background:{color_map[pred]}; 
                    padding:12px; border-radius:8px;
                    font-size:22px; color:white; 
                    font-weight:600; text-align:center;
                    margin-top:10px;">
            Predicted Class: {pred}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability Bar Chart
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


# --------------------------------------------------------------
# OPTIMAL PERFORMANCE PARAMETER FINDER
# --------------------------------------------------------------
st.markdown("### Optimal Performance Parameters")

if st.button("Find Optimal Parameters", use_container_width=True):

    with st.spinner("Searching for high-performance configuration..."):
        best_params, best_prob = find_optimal_parameters_fast(model, scaler, samples=1000)

    p, ps, a, w = best_params

    st.success(f"Highest probability for 'High' class: **{best_prob:.4f}**")

    st.write("### Best Parameter Set Found:")
    st.write(f"- Porosity: **{p:.3f}**")
    st.write(f"- Pore Size: **{ps:.2f} µm**")
    st.write(f"- Fiber Angle: **{a:.2f}°**")
    st.write(f"- Wettability: **{w:.2f}°**")

    # Radar chart for optimal parameters
    categories = ["Porosity", "Pore Size", "Fiber Angle", "Wettability"]
    values = [p, ps, a, w]

    radar2 = go.Figure()
    radar2.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color="#2a72d6")
    ))
    radar2.update_layout(showlegend=False, height=400)
    st.plotly_chart(radar2, use_container_width=True)


# --------------------------------------------------------------
# RECOMMENDED RANGES
# --------------------------------------------------------------
st.markdown("### Recommended High-Performance Ranges")

st.info("""
Values that commonly lead to the **High** performance class:

- Porosity: **0.70 – 0.78**
- Pore Size: **30 – 45 µm**
- Fiber Angle: **38° – 50°**
- Wettability: **50° – 75°**

These ranges usually produce the highest-performance GDL configurations.
""")

