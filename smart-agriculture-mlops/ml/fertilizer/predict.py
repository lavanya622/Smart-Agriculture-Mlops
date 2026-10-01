# ==========================================
# RECOMMENDED FERTILIZER PREDICTION
# PREDICTION SCRIPT
# ==========================================

import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# 1. PROJECT PATH
# ==========================================

project_dir = Path(__file__).resolve().parents[2]


# ==========================================
# 2. MODEL PATH
# ==========================================

model_path = (
    project_dir
    / "models"
    / "fertilizer_prediction_model.pkl"
)


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = joblib.load(model_path)

print("=" * 100)
print("MODEL LOADED SUCCESSFULLY!")


# ==========================================
# 4. NEW INPUT DATA
# ==========================================

new_data = pd.DataFrame({
    "Soil_Type": ["Clay"],
    "Soil_pH": [6.5],
    "Soil_Moisture": [35.0],
    "Organic_Carbon": [0.9],
    "Electrical_Conductivity": [1.5],
    "Nitrogen_Level": [100],
    "Phosphorus_Level": [50],
    "Potassium_Level": [80],
    "Temperature": [25.0],
    "Humidity": [65.0],
    "Rainfall": [1500.0],
    "Crop_Type": ["Wheat"],
    "Crop_Growth_Stage": ["Vegetative"],
    "Season": ["Rabi"],
    "Irrigation_Type": ["Canal"],
    "Previous_Crop": ["Maize"],
    "Region": ["South"],
    "Fertilizer_Used_Last_Season": [150.0],
    "Yield_Last_Season": [4.5]
})


# ==========================================
# 5. DISPLAY INPUT
# ==========================================

print("=" * 100)
print("NEW INPUT DATA")

print(new_data)


# ==========================================
# 6. PREDICTION
# ==========================================

prediction = model.predict(new_data)


# ==========================================
# 7. PREDICTION PROBABILITY
# ==========================================

probability = model.predict_proba(new_data)


# ==========================================
# 8. DISPLAY PREDICTION
# ==========================================

print("=" * 100)

print(
    "RECOMMENDED FERTILIZER:",
    prediction[0]
)


# ==========================================
# 9. DISPLAY PROBABILITIES
# ==========================================

print("=" * 100)

print("PREDICTION PROBABILITIES:")

for class_name, prob in zip(
    model.classes_,
    probability[0]
):

    print(
        f"{class_name}: {prob:.2%}"
    )


print("=" * 100)