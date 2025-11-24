import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="GDL Performance Classifier", layout="centered")

# -------------------------------------------------
# CSS STYLING
# -------------------------------------------------
st.markdown("""
<style>
    .title { font-size: 32px; font-weight: 700; text-align: center; margin-bottom: 5px; }
    .subtitle { text-align: center; font-size: 15px; color: #bbb; margin-bottom: 35px; }
    .prediction-box {
        padding: 12px; border-radius: 8px;
        font-size: 22px; font-weight: 600;
        color: white; text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown('<div class="title">GDL Performance Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict performance from GDL microstructure parameters</div>', unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
def init_state():
    defaults = {
        "porosity_val": 0.65,
        "pore_val": 35.0,
        "angle_val": 45.0,
        "wet_val": 60.0,
        "opt_params": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# -------------------------------------------------
# LOAD MODEL + SCALER
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------------------------
# FAST OPTIMAL PARAMETER FINDER
# -------------------------------------------------
def find_optimal_parameters_fast(model, scaler, samples=1000):
    best_prob = -1
    best_params = None

    for _ in range(samples):
        p = random.uniform(0.3, 0.9)
        ps = random.uniform(1.0, 100.0)
        a = random.uniform(0.0, 90.0)
        w = random.uniform(0.0, 120.0)

        X = pd.DataFrame([{
            "Porosity": p,
            "Pore_Size_um": ps,
            "Fiber_Arrangement_Angle": a,
            "Wettability_Contact_Angle": w
        }])

        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[0]

        high_idx = list(model.classes_).index("High")
        high_prob = probs[high_idx]

        if high_prob > best_prob:
            best_prob = high_prob
            best_params = (p, ps, a, w)

    return best_params, best_prob

# -------------------------------------------------
# SIDEBAR — SEGMENTED CONTROL TOGGLE
# -------------------------------------------------
st.sidebar.subheader("Input Mode")

input_mode = st.sidebar.segmented_control(
    "Select Input Type",
    options=["Sliders", "Custom Input"],
    default="Sliders"
)

# -------------------------------------------------
# AUTO-APPLY OPTIMAL PARAMETERS ON NEXT RUN
# -------------------------------------------------
if st.session_state.opt_params:
    p, ps, a, w = st.session_state.opt_params

    st.session_state.porosity_val = float(p)
    st.session_state.pore_val = float(ps)
    st.session_state.angle_val = float(a)
    st.session_state.wet_val = float(w)

    st.session_state.opt_params = None

# -------------------------------------------------
# PARAMETERS (SLIDER + CUSTOM INPUT) WITH KEYS
# -------------------------------------------------
st.sidebar.header("Parameters")

if input_mode == "Sliders":
    porosity = st.sidebar.slider(
        "Porosity", 
        0.3, 0.9,
        float(st.session_state.porosity_val),
        step=0.01,
        key="porosity_slider"
    )
    pore_size = st.sidebar.slider(
        "Pore Size (µm)",
        1.0, 100.0,
        float(st.session_state.pore_val),
        step=1.0,
        key="pore_slider"
    )
    fiber_angle = st.sidebar.slider(
        "Fiber Arrangement Angle (°)",
        0.0, 90.0,
        float(st.session_state.angle_val),
        step=0.5,
        key="angle_slider"
    )
    wettability = st.sidebar.slider(
        "Wettability Contact Angle (°)",
        0.0, 120.0,
        float(st.session_state.wet_val),
        step=1.0,
        key="wet_slider"
    )

else:  # Custom inputs
    porosity = st.sidebar.number_input(
        "Porosity (0.3 - 0.9)", 
        min_value=0.3, max_value=0.9,
        value=float(st.session_state.porosity_val),
        step=0.001, format="%.3f",
        key="porosity_input"
    )
    pore_size = st.sidebar.number_input(
        "Pore Size (µm)",
        min_value=1.0, max_value=100.0,
        value=float(st.session_state.pore_val),
        step=0.5,
        key="pore_input"
    )
    fiber_angle = st.sidebar.number_input(
        "Fiber Arrangement Angle (°)",
        min_value=0.0, max_value=90.0,
        value=float(st.session_state.angle_val),
        step=0.5,
        key="angle_input"
    )
    wettability = st.sidebar.number_input(
        "Wettability Contact Angle (°)",
        min_value=0.0, max_value=120.0,
        value=float(st.session_state.wet_val),
        step=0.5,
        key="wet_input"
    )

# update current values
st.session_state.porosity_val = porosity
st.session_state.pore_val = pore_size
st.session_state.angle_val = fiber_angle
st.session_state.wet_val = wettability

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
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

    color_map = {"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}

    st.markdown(
        f"""
        <div class="prediction-box" style="background:{color_map[pred]};">
            Predicted Class: {pred}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability bar chart
    fig = px.bar(
        x=model.classes_,
        y=probs,
        color=model.classes_,
        color_discrete_map=color_map,
        labels={"x": "Class", "y": "Probability"},
        height=330
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    labels = ["Porosity", "Pore Size", "Fiber Angle", "Wettability"]
    values = [porosity, pore_size, fiber_angle, wettability]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself'
    ))
    radar.update_layout(height=400, showlegend=False)
    st.plotly_chart(radar, use_container_width=True)

# -------------------------------------------------
# OPTIMAL PARAMETERS
# -------------------------------------------------
st.markdown("### Optimal Performance Parameters")

if st.button("Find Optimal Parameters", use_container_width=True):
    with st.spinner("Searching (1000 samples)..."):
        best_params, best_prob = find_optimal_parameters_fast(model, scaler)

    p, ps, a, w = best_params

    st.success(f"Best probability of 'High': **{best_prob:.4f}**")

    st.write(f"- Porosity: **{p:.3f}**")
    st.write(f"- Pore Size: **{ps:.2f} µm**")
    st.write(f"- Fiber Angle: **{a:.2f}°**")
    st.write(f"- Wettability: **{w:.2f}°**")

    # Radar chart
    labels = ["Porosity", "Pore Size", "Fiber Angle", "Wettability"]
    values = [p, ps, a, w]

    radar2 = go.Figure()
    radar2.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself'
    ))
    radar2.update_layout(height=400, showlegend=False)
    st.plotly_chart(radar2, use_container_width=True)

    # Save to session_state for auto-fill
    if st.button("Use These Parameters", use_container_width=True):
        st.session_state.opt_params = best_params
        st.experimental_rerun()

# -------------------------------------------------
# RECOMMENDED RANGES
# -------------------------------------------------
st.markdown("### Recommended High-Performance Ranges")

st.info("""
Ranges associated with **High** performance:

- Porosity: **0.70 – 0.78**
- Pore Size: **30 – 45 µm**
- Fiber Angle: **38° – 50°**
- Wettability: **50° – 75°**
""")
