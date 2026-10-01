# ==========================================
# CROP RECOMMENDATION
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
    / "crop_recommendation_model.pkl"
)


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = joblib.load(model_path)

print("=" * 100)
print("MODEL LOADED SUCCESSFULLY!")


# ==========================================
# 4. NEW FARM DATA
# ==========================================

new_data = pd.DataFrame({

    "N": [90],

    "P": [42],

    "K": [43],

    "temperature": [21.0],

    "humidity": [82.0],

    "ph": [6.5],

    "rainfall": [203.0]

})


print("=" * 100)
print("NEW INPUT DATA")

print(new_data)


# ==========================================
# 5. PREDICTION
# ==========================================

prediction = model.predict(new_data)


# ==========================================
# 6. PREDICTION PROBABILITY
# ==========================================

probability = model.predict_proba(new_data)


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("=" * 100)

print("RECOMMENDED CROP:", prediction[0])


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