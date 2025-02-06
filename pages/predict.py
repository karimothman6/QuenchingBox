import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sys

def resource_path(relative_path):
    """Get absolute path to resources, works for dev and PyInstaller"""
    try:
        base_path = sys._MEIPASS  # PyInstaller creates a temp folder
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load the trained model pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load(resource_path("Random Forest 25_pipeline.joblib"))

pipeline = load_pipeline()

st.set_page_config(
    page_title="Strength Predictor",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSDdvw54ABycnSpE-o_dWtBKsJGGqtPLwi0w&s"
)

st.title("Strength Predictor")

# Add under the title
st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiVe1HRt5eIRvbsvsnGjlKVqJTIJbLQbBWgSErE-AkE5JZeAIAjMoq87bteilcF-rLyRM8uFv4kj9Cc18a_OxnnJnxKScepazpcLnc_p3RHdKUtBxXMY74AQ31XjYDBBJzCd4aGpEeNjTeY/s640/logo-2.png")

st.subheader("Enter Chemical Composition and Parameters")

# Input fields for chemical composition
c = st.number_input("Carbon (C)", 0.2, 0.32, 0.25, step=0.01)
si = st.number_input("Silicon (Si)", 0.14, 0.55, 0.3, step=0.01)
mn = st.number_input("Manganese (Mn)", 0.6, 1.8, 1.2, step=0.1)
p = st.number_input("Phosphorus (P)", 0.006, 0.04, 0.02, step=0.001)
s = st.number_input("Sulfur (S)", 0.009, 0.04, 0.02, step=0.001)
ni = st.number_input("Nickel (Ni)", 0.01, 0.25, 0.1, step=0.01)
cr = st.number_input("Chromium (Cr)", 0.026, 0.3, 0.15, step=0.01)
mo = st.number_input("Molybdenum (Mo)", 0.007, 0.1, 0.05, step=0.01)
cu = st.number_input("Copper (Cu)", 0.131, 0.7, 0.4, step=0.01)
v = st.number_input("Vanadium (V)", 0.00024, 0.013, 0.005, step=0.001)
n = st.number_input("Nitrogen (N)", 0.0056, 0.014, 0.01, step=0.001)
ce = st.number_input("CE%", 0.35, 0.61, 0.5, step=0.01)
do_value = st.selectbox("Select 'do' value", [10, 12, 16, 18, 22, 25, 32])

# Predict yield strength
if st.button("Predict Yield Strength"):
    input_data = pd.DataFrame([{ 
        'C': c, 'Si': si, 'Mn': mn, 'P': p, 'S': s,
        'Ni': ni, 'Cr': cr, 'Mo': mo, 'Cu': cu, 'V': v,
        'N': n, 'CE% ': ce, 'do': do_value
    }])
    
    S_y_pred, S_u_pred = pipeline.predict(input_data)[0]
    ratio_pred = S_u_pred / S_y_pred
    
    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Yield Strength (S_y)", f"{S_y_pred:.2f} MPa")
    with col2:
        st.metric("Ultimate Strength (S_u)", f"{S_u_pred:.2f} MPa")
    with col3:
        st.metric("S_u/S_y Ratio", f"{ratio_pred:.2f}")
