from pathlib import Path

# Root project location
ROOT = Path(r"C:\Users\guntu\OneDrive\Desktop\Project\smart-agriculture-mlops")

# Directories
directories = [
    # Frontend
    "frontend/public",
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/services",
    "frontend/src/hooks",
    "frontend/src/utils",

    # Backend
    "backend/app/api/routes",
    "backend/app/models",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/database",

    # ML Services
    "ml/crop_recommendation",
    "ml/yield_prediction",
    "ml/fertilizer",
    "ml/irrigation",
    "ml/price_prediction",

    # MLOps
    "data",
    "pipelines",
    "tests",
    "monitoring",

    # CI/CD
    ".github/workflows",
]

# Create directories
for directory in directories:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)


# Files
files = [
    # Frontend
    "frontend/src/App.jsx",
    "frontend/src/main.jsx",

    # Backend API Routes
    "backend/app/api/routes/crop.py",
    "backend/app/api/routes/yield.py",
    "backend/app/api/routes/fertilizer.py",
    "backend/app/api/routes/irrigation.py",
    "backend/app/api/routes/price.py",

    # Backend Core
    "backend/app/config.py",
    "backend/app/main.py",
    "backend/requirements.txt",
    "backend/Dockerfile",

    # ML - Crop Recommendation
    "ml/crop_recommendation/train.py",
    "ml/crop_recommendation/predict.py",

    # ML - Yield Prediction
    "ml/yield_prediction/train.py",
    "ml/yield_prediction/predict.py",

    # ML - Fertilizer Recommendation
    "ml/fertilizer/train.py",
    "ml/fertilizer/predict.py",

    # ML - Irrigation
    "ml/irrigation/train.py",
    "ml/irrigation/predict.py",

    # ML - Price Prediction
    "ml/price_prediction/train.py",
    "ml/price_prediction/predict.py",

    # CI/CD
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",

    # Root
    "docker-compose.yml",
    "README.md",
]

# Create files
for file in files:
    file_path = ROOT / file
    file_path.touch(exist_ok=True)


print("=" * 60)
print("✅ SMART AGRICULTURE MLOPS PROJECT CREATED")
print("=" * 60)
print(f"📁 Location: {ROOT}")
print("=" * 60)