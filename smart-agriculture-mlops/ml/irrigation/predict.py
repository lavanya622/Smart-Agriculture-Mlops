# ==========================================
# IRRIGATION NEED PREDICTION
# PREDICTION SCRIPT
# ==========================================


# Step 1: Import libraries

import pandas as pd

import joblib

from pathlib import Path


# ==========================================
# Step 2: Find Project Directory
# ==========================================

project_dir = Path(__file__).resolve().parents[2]


# ==========================================
# Step 3: Load Saved Model
# ==========================================

model_path = (
    project_dir
    / "models"
    / "irrigation_prediction_model.pkl"
)


model = joblib.load(
    model_path
)


print("=" * 100)

print("MODEL LOADED SUCCESSFULLY!")


# ==========================================
# Step 4: Create New Input
# ==========================================

new_data = pd.DataFrame({

    "Soil_Type": ["Clay"],

    "Soil_pH": [6.5],

    "Soil_Moisture": [35.0],

    "Organic_Carbon": [0.9],

    "Electrical_Conductivity": [30.0],

    "Temperature_C": [1.0],

    "Humidity": [2.0],

    "Rainfall_mm": [25.0],

    "Sunlight_Hours": [70.0],

    "Wind_Speed_kmh": [10.0],

    "Crop_Type": ["Wheat"],

    "Crop_Growth_Stage": ["Vegetative"],

    "Season": ["Rabi"],

    "Irrigation_Type": ["Rainfed"],

    "Water_Source": ["Reservoir"],

    "Field_Area_hectare": [5.0],

    "Mulching_Used": ["Yes"],

    "Previous_Irrigation_mm": [20.0],

    "Region": ["South"]
})
# ==========================================
# Step 5: Display Input
# ==========================================

print("=" * 100)

print("NEW INPUT DATA")

print(new_data)


# ==========================================
# Step 6: Make Prediction
# ==========================================

prediction = model.predict(
    new_data
)


# ==========================================
# Step 7: Prediction Probability
# ==========================================

probability = model.predict_proba(
    new_data
)


# ==========================================
# Step 8: Display Prediction
# ==========================================

print("=" * 100)

print(
    "PREDICTED IRRIGATION NEED:",
    prediction[0]
)


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