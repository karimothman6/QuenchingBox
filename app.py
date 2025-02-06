import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Quenching Box",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSDdvw54ABycnSpE-o_dWtBKsJGGqtPLwi0w&s"
)

# Title and Logo
st.title("Quenching Box - Main Page")
st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiVe1HRt5eIRvbsvsnGjlKVqJTIJbLQbBWgSErE-AkE5JZeAIAjMoq87bteilcF-rLyRM8uFv4kj9Cc18a_OxnnJnxKScepazpcLnc_p3RHdKUtBxXMY74AQ31XjYDBBJzCd4aGpEeNjTeY/s640/logo-2.png")

# Main Page Content
st.write("Welcome to the Quenching Box Application. Select an option to proceed.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Optimize Chemical Composition"):
        st.switch_page("pages/optimize.py")

with col2:
    if st.button("Predict Strength"):
        st.switch_page("pages/predict.py")
