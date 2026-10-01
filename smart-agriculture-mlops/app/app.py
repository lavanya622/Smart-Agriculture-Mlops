import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

project_dir = Path(__file__).resolve().parents[1]

model_dir = project_dir / "models"


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Smart Agriculture",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Smart Agriculture")
st.subheader("AI-Powered Agricultural Prediction System")

st.write(
    "Use Machine Learning to get smart agricultural predictions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌾 Smart Agriculture")

option = st.sidebar.selectbox(
    "Select Prediction",
    [
        "Crop Recommendation",
        "Irrigation Need",
        "Crop Price Prediction",
        "Yield Prediction"
    ]
)


# ============================================================
# CROP RECOMMENDATION
# ============================================================

if option == "Crop Recommendation":

    st.header("🌱 Crop Recommendation")

    st.write(
        "Enter soil and weather conditions."
    )

    col1, col2 = st.columns(2)

    with col1:

        N = st.number_input(
            "Nitrogen (N)",
            value=90.0
        )

        P = st.number_input(
            "Phosphorus (P)",
            value=42.0
        )

        K = st.number_input(
            "Potassium (K)",
            value=43.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=21.0
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            value=82.0
        )

        ph = st.number_input(
            "Soil pH",
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            value=203.0
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button("🌱 Recommend Crop"):

        model_path = (
            model_dir /
            "crop_recommendation_model.pkl"
        )

        model = joblib.load(model_path)

        new_data = pd.DataFrame({
            "N": [N],
            "P": [P],
            "K": [K],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall]
        })

        prediction = model.predict(new_data)

        st.success(
            f"Recommended Crop: 🌾 {prediction[0]}"
        )