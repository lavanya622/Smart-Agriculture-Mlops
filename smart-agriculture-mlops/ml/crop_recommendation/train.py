# ==========================================
# CROP RECOMMENDATION
# TRAINING SCRIPT
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib
from pathlib import Path


# ==========================================
# 1. PROJECT PATH
# ==========================================

project_dir = Path(__file__).resolve().parents[2]

data_path = project_dir / "data" / "Crop_recommendation _dataset.csv"


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
# 3. DATA PREPROCESSING
# ==========================================

# Remove duplicate rows

df = df.drop_duplicates()

print("=" * 100)
print("SHAPE AFTER REMOVING DUPLICATES")
print(df.shape)


# ==========================================
# 4. TARGET ANALYSIS
# ==========================================

print("=" * 100)
print("CROP COUNTS")

print(df["label"].value_counts())

print("=" * 100)
print("UNIQUE CROPS")

print(df["label"].unique())


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

# Input features

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


# ==========================================
# 6. DIVIDE DATA INTO X AND Y
# ==========================================

X = df[features]

y = df["label"]

print("=" * 100)
print("X SHAPE")
print(X.shape)

print("=" * 100)
print("Y SHAPE")
print(y.shape)

print("=" * 100)
print("FEATURE COLUMNS")
print(X.columns.tolist())

print("=" * 100)
print("TARGET COLUMN")
print(y.name)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
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
# 8. PREPROCESSING PIPELINE
# ==========================================

preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================
# 9. RANDOM FOREST CLASSIFIER
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 10. COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 11. MODEL TRAINING
# ==========================================

print("=" * 100)
print("MODEL TRAINING STARTED...")

pipeline.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")


# ==========================================
# 12. PREDICTION
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 13. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 100)
print("MODEL EVALUATION")

print("Accuracy:", accuracy)


print("=" * 100)
print("CLASSIFICATION REPORT")

print(classification_report(y_test, y_pred))


print("=" * 100)
print("CONFUSION MATRIX")

print(confusion_matrix(y_test, y_pred))


# ==========================================
# 14. ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("=" * 100)
print("ACTUAL VS PREDICTED")

print(results.head(20))


# ==========================================
# 15. SAVE MODEL
# ==========================================

model_dir = project_dir / "models"

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


model_path = model_dir / "crop_recommendation_model.pkl"


joblib.dump(
    pipeline,
    model_path
)


print("=" * 100)
print("MODEL SUCCESSFULLY SAVED!")

print(model_path.resolve())

print("=" * 100)