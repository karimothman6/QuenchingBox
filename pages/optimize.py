import streamlit as st
import numpy as np
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import joblib
import os
import sys

def resource_path(relative_path):
    """Get absolute path to resources, works for dev and PyInstaller"""
    try:
        base_path = sys._MEIPASS  # PyInstaller creates a temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

@st.cache_resource
def load_pipeline():
    return joblib.load(resource_path("Random Forest 25_pipeline.joblib"))

pipeline = load_pipeline()

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="Optimize Chemical Composition",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSDdvw54ABycnSpE-o_dWtBKsJGGqtPLwi0w&s"
)

# Home Button to return to the main page
if st.button("Home"):
    st.switch_page("app.py")

st.title("Chemical Composition Optimizer")
st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiVe1HRt5eIRvbsvsnGjlKVqJTIJbLQbBWgSErE-AkE5JZeAIAjMoq87bteilcF-rLyRM8uFv4kj9Cc18a_OxnnJnxKScepazpcLnc_p3RHdKUtBxXMY74AQ31XjYDBBJzCd4aGpEeNjTeY/s640/logo-2.png")

st.write("Enter chemical composition ranges and target objectives below:")

with st.expander("🧪 CHEMICAL COMPOSITION", expanded=True):
    st.subheader("Chemical Composition Ranges")
    c_min, c_max = st.slider("Carbon (C)", 0.2, 0.32, (0.2, 0.32))
    si_min, si_max = st.slider("Silicon (Si)", 0.14, 0.55, (0.14, 0.55))
    mn_min, mn_max = st.slider("Manganese (Mn)", 0.6, 1.8, (0.6, 1.8))
    p_min, p_max = st.slider("Phosphorus (P)", 0.006, 0.04, (0.006, 0.04))
    s_min, s_max = st.slider("Sulfur (S)", 0.009, 0.04, (0.009, 0.04))
    ni_min, ni_max = st.slider("Nickel (Ni)", 0.01, 0.25, (0.01, 0.25))
    cr_min, cr_max = st.slider("Chromium (Cr)", 0.026, 0.3, (0.026, 0.3))
    mo_min, mo_max = st.slider("Molybdenum (Mo)", 0.007, 0.1, (0.007, 0.1))
    cu_min, cu_max = st.slider("Copper (Cu)", 0.131, 0.7, (0.131, 0.7))
    v_min, v_max = st.slider("Vanadium (V)", 0.00024, 0.013, (0.00024, 0.013))
    n_min, n_max = st.slider("Nitrogen (N)", 0.0056, 0.014, (0.0056, 0.014))
    ce_min, ce_max = st.slider("CE%", 0.35, 0.61, (0.35, 0.61))
    do_value = st.selectbox("Select 'do' value", [10, 12, 16, 18, 22, 25, 32])

with st.expander("🎯 TARGET OBJECTIVES", expanded=True):
    st.subheader("Target Objectives")
    sy_min, sy_max = st.slider("Yield Strength (MPa)", 400, 700, (500, 650))
    target_ratio = st.slider("Target S_u/S_y Ratio", 1.0, 1.5, 1.25)

if st.button("Run Optimization"):
    # Define variable boundaries (12 parameters)
    varbound = np.array([
        [c_min, c_max], [si_min, si_max], [mn_min, mn_max],
        [p_min, p_max], [s_min, s_max], [ni_min, ni_max],
        [cr_min, cr_max], [mo_min, mo_max], [cu_min, cu_max],
        [v_min, v_max], [n_min, n_max], [ce_min, ce_max]
    ])

    # Fitness function definition
    def fitness_function(X):
        features = {
            'C': X[0], 'Si': X[1], 'Mn': X[2], 'P': X[3], 'S': X[4],
            'Ni': X[5], 'Cr': X[6], 'Mo': X[7], 'Cu': X[8], 'V': X[9],
            'N': X[10], 'CE% ': X[11], 'do': do_value
        }
        input_data = pd.DataFrame([features])
        S_y, S_u = pipeline.predict(input_data)[0]
        penalty_yield = max(sy_min - S_y, 0) + max(S_y - sy_max, 0)
        penalty_ratio = abs((S_u / S_y) - target_ratio)
        return penalty_yield + 10 * penalty_ratio + 100 * X[2]  # Example penalty formulation

    ga_model = ga(
        function=fitness_function,
        dimension=12,
        variable_type_mixed=np.array(['real'] * 12),
        variable_boundaries=varbound,
        algorithm_parameters={
            'max_num_iteration': 100,
            'population_size': 50,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'parents_portion': 0.3,
            'crossover_probability': 0.8,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }
    )

    with st.spinner("Optimizing..."):
        ga_model.run()

    solution = ga_model.output_dict
    best_params = solution['variable']

    st.success("Optimization Complete!")
    st.subheader("Optimal Parameters:")
    params = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'N', 'CE% ']
    params_df = pd.DataFrame({
        'Parameter': params + ['do'],
        'Value': list(best_params) + [do_value]
    })
    st.dataframe(
        params_df.style.format({'Value': '{:.4f}'}),
        hide_index=True,
        use_container_width=True
    )

    input_dict = {param: value for param, value in zip(params, best_params)}
    input_dict['do'] = do_value
    input_data = pd.DataFrame([input_dict])
    S_y_pred, S_u_pred = pipeline.predict(input_data)[0]
    ratio_pred = S_u_pred / S_y_pred

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Yield Strength (S_y)", f"{S_y_pred:.2f} MPa")
    with col2:
        st.metric("Ultimate Strength (S_u)", f"{S_u_pred:.2f} MPa")
    with col3:
        st.metric("S_u/S_y Ratio", f"{ratio_pred:.2f}")
