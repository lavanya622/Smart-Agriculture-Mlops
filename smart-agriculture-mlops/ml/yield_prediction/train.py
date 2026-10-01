# Yield Prediction 
# step 1: Import necessary libraries
# Step 1: Import necessary libraries

# %pip install pandas numpy scikit-learn joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
# step 2 : Load the dataset
# data_path = path(__file__).parent / 'data'
df = pd.read_csv("../../data/yield_dataset.csv")
df
# data preproccessing
# Step 2: Load the dataset

df = pd.read_csv("../../data/yield_dataset.csv")

print(df.head())
print(df.shape)
# Step 3: Data Preprocessing

# Check dataset information
# print(df.info())

# # Check missing values
# print("\nMissing values:")
# print(df.isnull().sum())

# # Check duplicate rows
# print("\nDuplicate rows:")
# print(df.duplicated().sum())

# # Check basic statistics
# print("\nStatistical summary:")
# print(df.describe())
# # Step 4: Exploratory Data Analysis (EDA)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. Distribution of Yield
# plt.figure(figsize=(8, 5))
# sns.histplot(df["Yield_t_per_ha"], bins=30, kde=True)
# plt.title("Distribution of Crop Yield")
# plt.xlabel("Yield (t per hectare)")
# plt.ylabel("Frequency")
# plt.show()
# Yield Prediction

# # Step 1:Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error,r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# import joblib
# from pathlib import Path

# # Step 2: Load the dataset
# project_dir = Path(__file__).resolve().parents[2]
# df = pd.read_csv(project_dir / 'data' / 'yield_dataset.csv')


# # Step 3: Data Preprocessing
print(df.head())

print('#'*100)

# Check the the shape
print(f"Shape of the dataset: {df.shape}")

print('#'*100)

# check the data types of the columns
print(f"Data types of the columns: {df.dtypes}")

# check for missing values
print(f"Missing values in the dataset: {df.isnull().sum()}")

print('#'*100)
# check for duplicates
print(f"Number of duplicate rows in the dataset: {df.duplicated().sum()}")

# Step 4: Handle missing values
df = df.dropna()

# check the unique value of crop_type column
print(f"Unique values in the 'crop_type' column: {df['Crop Type'].unique()}")
print("#"*100)

# drop the duplicate rows
df = df.drop_duplicates()

# Step 5: Feature Engineering

#Select the categorical

Categorical_columns = ['Crop Type', 'Soil Type', 'Fertilizer Used','Irrigation Type']

for col in Categorical_columns:
    print(f"Unique values in the '{col}' column: {df[col].unique()}")
    print("#"*100)

# hanlded the Planting Date column
df['Planting Date'] = pd.to_datetime(
                    df['Planting Date'],
                    format='%d/%m/%Y',
                    errors='coerce'
                    )

# Extract the month and year from the Planting Date column
df['Planting_Month'] = df['Planting Date'].dt.month
df['Planting_Year'] = df['Planting Date'].dt.year
df['Planting_Day'] = df['Planting Date'].dt.day
df['Planting_DayOfYear'] = df['Planting Date'].dt.dayofyear
df['Planting_quarter'] = df['Planting Date'].dt.quarter
# drop the Planting Date column
df = df.drop(columns=['Planting Date'])

# check the invalid value in the Field Size (hectares) column
invalid_field_size = df[df['Field Size (hectares)'] <= 0]
print(f"Invalid values in the 'Field Size (hectares)' column: {invalid_field_size}")

# check the invalid value in the Yield_t_per_ha column
invalid_yield = df[df['Yield_t_per_ha'] <= 0]

print(f"Invalid values in the 'Yield_t_per_ha' column: {invalid_yield}")
print('#',100)
# check the outlier Yield_t_per_ha column
Q1 = df['Yield_t_per_ha'].quantile(0.25)
Q3 = df['Yield_t_per_ha'].quantile(0.75)

IQR = Q3 - Q1

Lower_Bound = Q1 - 1.5 * IQR
Upper_Bound = Q3 + 1.5 * IQR

Outliers = df[(df['Yield_t_per_ha'] < Lower_Bound) | (df['Yield_t_per_ha'] > Upper_Bound)]
print(f"Outliers in the 'Yield_t_per_ha' column: {Outliers}")
print(f"Outliers in the 'Yield_t_per_ha' column: {Outliers.shape[0]}")
print('#'*100)

# Step 6: Split the dataset into features and target variable
X = df.drop(columns=['Yield_t_per_ha'])
y = df['Yield_t_per_ha']

# Step 7: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Category and numerical columns
categorical_features = [
    'Crop Type',
    'Soil Type', 
    'Fertilizer Used',
    'Irrigation Type'
]

# Numerical features
numerical_features = [
    'Field Size (hectares)',
    'Planting_Month',
    'Planting_Year',
    'Planting_Day',
    'Planting_DayOfYear',
    'Planting_quarter'
]
# handled the missing values and categorical features
numerical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
)

# handled the categorical features
categorical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
# # handled the categorical features
# categorical_pipeline = Pipeline(
#     steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ]
# )

# create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)
# first model is Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42,
    n_jobs=-1
)

# create the pipeline
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

# Step 8: Train the model
pipeline.fit(X_train, y_train)

# Step 9: Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# save the model
# save the trained model to a file
# model_path = project_dir / "backend"/"app"/"models"
# model_dir = mkdir(parents= True,exit_ok=True)
# # saved the trained pipeline model
# model_path=model_dir / "yield_prediction_model.pkl"
# joblib.dump(rf_pipeline,model_path)
# print('=',100)
# print("Model Successfully Saved at :", {model_path.reslove()})

# Step 11: Save the trained model

project_dir = Path(__file__).resolve().parents[2]

model_dir = project_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "yield_prediction_model.pkl"

joblib.dump(pipeline, model_path)

print("=" * 100)
print("Model successfully saved at:")
print(model_path.resolve())