# ==========================================
# IRRIGATION NEED PREDICTION
# TRAINING SCRIPT
# ==========================================


# Step 1: Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

import joblib

from pathlib import Path


# ==========================================
# Step 2: Load the Dataset
# ==========================================

project_dir = Path(__file__).resolve().parents[2]

data_path = project_dir / "data" / "irrigation_prediction.csv"

df = pd.read_csv(data_path)


print("=" * 100)
print("FIRST 5 ROWS")
print(df.head())


print("=" * 100)
print("DATASET SHAPE")

print(df.shape)


# ==========================================
# Step 3: Data Preprocessing
# ==========================================


print("=" * 100)
print("DATA TYPES")

print(df.dtypes)


print("=" * 100)
print("MISSING VALUES")

print(df.isnull().sum())


print("=" * 100)
print("DUPLICATE ROWS")

print(df.duplicated().sum())


print("=" * 100)
print("STATISTICAL SUMMARY")

print(df.describe())


# ==========================================
# Step 4: Remove Duplicate Rows
# ==========================================

df = df.drop_duplicates()


print("=" * 100)
print("SHAPE AFTER REMOVING DUPLICATES")

print(df.shape)


# ==========================================
# Step 5: Check Target Values
# ==========================================

print("=" * 100)
print("IRRIGATION NEED VALUES")

print(df["Irrigation_Need"].value_counts())


print("=" * 100)
print("UNIQUE IRRIGATION NEED VALUES")

print(df["Irrigation_Need"].unique())


# ==========================================
# Step 6: Check Categorical Columns
# ==========================================

categorical_features = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region"
]


for col in categorical_features:

    print("=" * 100)

    print(f"UNIQUE VALUES IN {col}")

    print(df[col].unique())


# ==========================================
# Step 7: Divide Dataset into X and y
# ==========================================

X = df.drop(
    columns=["Irrigation_Need"]
)

y = df["Irrigation_Need"]


print("=" * 100)
print("FEATURE COLUMNS")

print(X.columns.tolist())


print("=" * 100)
print("TARGET COLUMN")

print(y.name)


# ==========================================
# Step 8: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("=" * 100)

print("TRAINING DATA SHAPE")

print(X_train.shape)


print("=" * 100)

print("TESTING DATA SHAPE")

print(X_test.shape)


# ==========================================
# Step 9: Numerical Features
# ==========================================

numerical_features = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm"
]


# ==========================================
# Step 10: Numerical Pipeline
# ==========================================

numerical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="mean"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==========================================
# Step 11: Categorical Pipeline
# ==========================================

categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==========================================
# Step 12: Column Transformer
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            numerical_pipeline,
            numerical_features
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# Step 13: Create Random Forest Classifier
# ==========================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1
)


# ==========================================
# Step 14: Create Complete Pipeline
# ==========================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ==========================================
# Step 15: Train Model
# ==========================================

print("=" * 100)

print("MODEL TRAINING STARTED...")


pipeline.fit(
    X_train,
    y_train
)


print("MODEL TRAINING COMPLETED!")


# ==========================================
# Step 16: Make Predictions
# ==========================================

y_pred = pipeline.predict(
    X_test
)


# ==========================================
# Step 17: Evaluate Model
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("=" * 100)

print("MODEL EVALUATION")


print("Accuracy:", accuracy)


# ==========================================
# Step 18: Classification Report
# ==========================================

print("=" * 100)

print("CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# Step 19: Confusion Matrix
# ==========================================

print("=" * 100)

print("CONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# Step 20: Actual vs Predicted
# ==========================================

results = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": y_pred

})


print("=" * 100)

print("ACTUAL VS PREDICTED")

print(
    results.head(20)
)


# ==========================================
# Step 21: Save the Model
# ==========================================

model_dir = project_dir / "models"


model_dir.mkdir(
    parents=True,
    exist_ok=True
)


model_path = (
    model_dir /
    "irrigation_prediction_model.pkl"
)


joblib.dump(
    pipeline,
    model_path
)


print("=" * 100)

print("MODEL SUCCESSFULLY SAVED!")

print(
    model_path.resolve()
)

print("=" * 100)