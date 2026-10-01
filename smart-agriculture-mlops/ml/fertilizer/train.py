# ==========================================
# RECOMMENDED FERTILIZER PREDICTION
# TRAINING SCRIPT
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import joblib
from pathlib import Path


# ==========================================
# 1. PROJECT PATH
# ==========================================

project_dir = Path(__file__).resolve().parents[2]

data_path = project_dir / "data" / "fertilizer_recommendation.csv"


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(data_path)

print("=" * 100)
print("FIRST 5 ROWS")
print(df.head())

print("=" * 100)
print("DATASET SHAPE")
print(df.shape)

print("=" * 100)
print("COLUMN NAMES")
print(df.columns.tolist())

print("=" * 100)
print("DATA TYPES")
print(df.dtypes)

print("=" * 100)
print("MISSING VALUES")
print(df.isnull().sum())

print("=" * 100)
print("DUPLICATE ROWS")
print(df.duplicated().sum())


# ==========================================
# 3. STATISTICAL SUMMARY
# ==========================================

print("=" * 100)
print("STATISTICAL SUMMARY")

print(df.describe())


# ==========================================
# 4. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("=" * 100)
print("SHAPE AFTER REMOVING DUPLICATES")

print(df.shape)


# ==========================================
# 5. TARGET ANALYSIS
# ==========================================

print("=" * 100)
print("RECOMMENDED FERTILIZER COUNTS")

print(df["Recommended_Fertilizer"].value_counts())

print("=" * 100)
print("UNIQUE FERTILIZER VALUES")

print(df["Recommended_Fertilizer"].unique())


# ==========================================
# 6. CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region"
]


# ==========================================
# 7. NUMERICAL FEATURES
# ==========================================

numerical_features = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Fertilizer_Used_Last_Season",
    "Yield_Last_Season"
]


# ==========================================
# 8. UNIQUE VALUES
# ==========================================

for col in categorical_features:

    print("=" * 100)
    print(f"UNIQUE VALUES IN {col}")

    print(df[col].unique())


# ==========================================
# 9. X AND Y
# ==========================================

X = df.drop(columns=["Recommended_Fertilizer"])

y = df["Recommended_Fertilizer"]

print("=" * 100)
print("FEATURE COLUMNS")

print(X.columns.tolist())

print("=" * 100)
print("TARGET COLUMN")

print(y.name)


# ==========================================
# 10. TRAIN TEST SPLIT
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
# 11. NUMERICAL PIPELINE
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 12. CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================
# 13. COLUMN TRANSFORMER
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
# 14. RANDOM FOREST CLASSIFIER
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 15. COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 16. MODEL TRAINING
# ==========================================

print("=" * 100)
print("MODEL TRAINING STARTED...")

pipeline.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")


# ==========================================
# 17. PREDICTION
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 18. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 100)
print("MODEL EVALUATION")

print("Accuracy:", accuracy)


# ==========================================
# 19. CLASSIFICATION REPORT
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
# 20. CONFUSION MATRIX
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
# 21. ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("=" * 100)
print("ACTUAL VS PREDICTED")

print(results.head(20))


# ==========================================
# 22. SAVE MODEL
# ==========================================

model_dir = project_dir / "models"

model_dir.mkdir(
    parents=True,
    exist_ok=True
)

model_path = (
    model_dir
    / "fertilizer_prediction_model.pkl"
)

joblib.dump(
    pipeline,
    model_path
)

print("=" * 100)
print("MODEL SUCCESSFULLY SAVED!")

print(model_path.resolve())

print("=" * 100)