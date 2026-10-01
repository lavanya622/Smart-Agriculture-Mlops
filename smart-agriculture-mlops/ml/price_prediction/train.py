# ==========================================
# CROP PRICE PREDICTION
# TRAINING SCRIPT
# ==========================================


# Step 1: Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

data_path = project_dir / "data" / "crop_price_dataset.csv"

df = pd.read_csv(data_path)


print("=" * 100)
print("FIRST 5 ROWS OF DATASET")
print(df.head())

print("=" * 100)
print("SHAPE OF DATASET")
print(df.shape)


# ==========================================
# Step 3: Data Preprocessing
# ==========================================


# Check data types

print("=" * 100)
print("DATA TYPES")
print(df.dtypes)


# Check missing values

print("=" * 100)
print("MISSING VALUES")
print(df.isnull().sum())


# Check duplicate rows

print("=" * 100)
print("NUMBER OF DUPLICATE ROWS")
print(df.duplicated().sum())


# Basic statistical summary

print("=" * 100)
print("STATISTICAL SUMMARY")
print(df.describe())


# ==========================================
# Step 4: Handle Missing Values
# ==========================================


# We do not drop all rows because only the
# 'change' column has some missing values.
# SimpleImputer will handle missing values later.


# Drop duplicate rows

df = df.drop_duplicates()


print("=" * 100)
print("SHAPE AFTER REMOVING DUPLICATES")
print(df.shape)


# ==========================================
# Step 5: Feature Engineering
# ==========================================


# Convert month column to datetime

df["month"] = pd.to_datetime(
    df["month"],
    errors="coerce"
)


# Check invalid dates

print("=" * 100)
print("INVALID DATES")
print(df["month"].isnull().sum())


# Extract date features

df["Year"] = df["month"].dt.year

df["Month_Number"] = df["month"].dt.month

df["Quarter"] = df["month"].dt.quarter


# Drop original month column

df = df.drop(columns=["month"])


print("=" * 100)
print("DATASET AFTER FEATURE ENGINEERING")
print(df.head())


# ==========================================
# Step 6: Check Categorical Columns
# ==========================================


categorical_features = [
    "commodity_name",
    "state_name",
    "district_name",
    "calculationType"
]


for col in categorical_features:

    print("=" * 100)
    print(f"UNIQUE VALUES IN {col}")

    print(df[col].unique())


# ==========================================
# Step 7: Check Outliers in Target
# ==========================================


Q1 = df["avg_modal_price"].quantile(0.25)

Q3 = df["avg_modal_price"].quantile(0.75)

IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


outliers = df[
    (df["avg_modal_price"] < lower_bound) |
    (df["avg_modal_price"] > upper_bound)
]


print("=" * 100)
print("NUMBER OF OUTLIERS IN avg_modal_price")
print(outliers.shape[0])


# IMPORTANT:
# We are only checking outliers here.
# We are NOT removing them automatically because
# extreme crop prices may be real market values.


# ==========================================
# Step 8: Divide Dataset into X and y
# ==========================================


X = df.drop(columns=["avg_modal_price"])

y = df["avg_modal_price"]


print("=" * 100)
print("FEATURES")
print(X.columns.tolist())

print("=" * 100)
print("TARGET")
print(y.name)


# ==========================================
# Step 9: Train-Test Split
# ==========================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("=" * 100)
print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)


# ==========================================
# Step 10: Define Numerical Features
# ==========================================


numerical_features = [
    "avg_min_price",
    "avg_max_price",
    "change",
    "Year",
    "Month_Number",
    "Quarter"
]


# ==========================================
# Step 11: Numerical Pipeline
# ==========================================


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="mean")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==========================================
# Step 12: Categorical Pipeline
# ==========================================


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
# Step 13: Create Preprocessor
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
# Step 14: Create Random Forest Model
# ==========================================


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# Step 15: Create Complete Pipeline
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
# Step 16: Train the Model
# ==========================================


print("=" * 100)
print("MODEL TRAINING STARTED...")


pipeline.fit(
    X_train,
    y_train
)


print("MODEL TRAINING COMPLETED!")


# ==========================================
# Step 17: Make Predictions
# ==========================================


y_pred = pipeline.predict(X_test)


# ==========================================
# Step 18: Evaluate Model
# ==========================================


mse = mean_squared_error(
    y_test,
    y_pred
)


mae = mean_absolute_error(
    y_test,
    y_pred
)


r2 = r2_score(
    y_test,
    y_pred
)


rmse = mse ** 0.5


print("=" * 100)
print("MODEL EVALUATION RESULTS")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R-squared Score:", r2)


# ==========================================
# Step 19: Actual vs Predicted
# ==========================================


results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})


print("=" * 100)
print("ACTUAL VS PREDICTED")

print(results.head(10))


# ==========================================
# Step 20: Save the Model
# ==========================================


model_dir = project_dir / "models"


model_dir.mkdir(
    parents=True,
    exist_ok=True
)


model_path = model_dir / "crop_price_prediction_model.pkl"


joblib.dump(
    pipeline,
    model_path
)


print("=" * 100)
print("MODEL SUCCESSFULLY SAVED!")

print(model_path.resolve())

print("=" * 100)
