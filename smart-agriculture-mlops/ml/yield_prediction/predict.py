# step1 import the lib
import pandas as pd
import joblib
from pathlib import Path

# step2: find the saved model
project_dir = Path(__file__).resolve().parents[2]

model_path = project_dir / "models" / "yield_prediction_model.pkl"

model = joblib.load(model_path)

print("Model loaded successfully!")
#Step 13: Give a new crop input

new_data = pd.DataFrame({
    'Crop Type': ['Maize'],
    'Soil Type': ['Loamy'],
    'Field Size (hectares)': [2.5],
    'Fertilizer Used': ['Yes'],
    'Irrigation Type': ['Sprinkler Irrigation'],
    'Planting_Month': [6],
    'Planting_Year': [2025],
    'Planting_Day': [15],
    'Planting_DayOfYear': [166],
    'Planting_quarter': [2]
})

prediction = model.predict(new_data)

print("Predicted Yield:", prediction[0], "t/ha")