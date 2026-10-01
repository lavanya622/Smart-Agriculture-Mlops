# ==========================================
# CROP PRICE PREDICTION
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
    / "crop_price_prediction_model.pkl"
)


model = joblib.load(model_path)


print("=" * 100)
print("MODEL LOADED SUCCESSFULLY!")


# ==========================================
# Step 4: Create New Input Data
# ==========================================


new_data = pd.DataFrame({

    "commodity_name": [
        "Maize"
    ],

    "avg_min_price": [
        1500.0
    ],

    "avg_max_price": [
        2500.0
    ],

    "state_name": [
        "Andhra Pradesh"
    ],

    "district_name": [
        "Guntur"
    ],

    "calculationType": [
        "Monthly"
    ],

    "change": [
        50.0
    ],

    "Year": [
        2025
    ],

    "Month_Number": [
        3
    ],

    "Quarter": [
        1
    ]
})


# ==========================================
# Step 5: Display Input Data
# ==========================================


print("=" * 100)

print("NEW INPUT DATA")

print(new_data)


# ==========================================
# Step 6: Predict Crop Price
# ==========================================


prediction = model.predict(
    new_data
)


# ==========================================
# Step 7: Display Prediction
# ==========================================


print("=" * 100)

print(
    "PREDICTED AVG MODAL PRICE:",
    round(prediction[0], 2)
)


print("=" * 100)