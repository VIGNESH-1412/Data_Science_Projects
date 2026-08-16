"""
Model Loader Utility
====================
Download or generate the XGBoost stroke prediction model.
Place your trained model .pkl file in the data/ folder, or use this script
to generate a compatible model from the BRFSS 2022 dataset.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(MODEL_DIR, "Stroke_Prediction.pkl")


def create_sample_model():
    """Create a sample trained model for demonstration purposes."""
    np.random.seed(42)

    # Generate synthetic training data matching the 10 features
    n_samples = 5000
    data = pd.DataFrame({
        "GeneralHealth": np.random.choice([1, 2, 3, 4, 5], n_samples),
        "HeartAttackHistory": np.random.choice([1, 2], n_samples),
        "CoronaryHeartDisease": np.random.choice([1, 2], n_samples),
        "Diabetes": np.random.choice([1, 3], n_samples),
        "Gender": np.random.choice([1, 2], n_samples),
        "Age": np.random.randint(18, 81, n_samples),
        "BMI": np.random.randint(1500, 4500, n_samples),
        "PhysicalActivity": np.random.choice([1, 2], n_samples),
        "SmokingStatus": np.random.choice([1, 2, 3, 4], n_samples),
        "HeavyAlcoholUse": np.random.choice([1, 2], n_samples),
    })

    # Generate target (stroke: 1=Yes, 2=No) based on risk factors
    risk_score = (
        3.5 * (data["HeartAttackHistory"] == 1).astype(int) +
        0.5 * (data["GeneralHealth"] - 1) +
        1.7 * (data["Diabetes"] == 1).astype(int) +
        1.0 * (data["CoronaryHeartDisease"] == 1).astype(int) +
        0.02 * (data["Age"] - 30).clip(lower=0) +
        0.05 * (data["SmokingStatus"].isin([1, 2])).astype(int) +
        0.4 * (data["PhysicalActivity"] == 2).astype(int) +
        0.03 * (data["BMI"].apply(lambda x: 1 if x >= 3000 else (0.5 if x >= 2500 else 0))) +
        0.1 * (data["Gender"] == 1).astype(int) +
        0.3 * (data["HeavyAlcoholUse"] == 2).astype(int)
    )

    threshold = np.percentile(risk_score, 92)
    target = np.where(risk_score >= threshold, 1, 2)
    data["target"] = target

    features = [
        "GeneralHealth", "HeartAttackHistory", "CoronaryHeartDisease",
        "Diabetes", "Gender", "Age", "BMI", "PhysicalActivity",
        "SmokingStatus", "HeavyAlcoholUse"
    ]

    X = data[features]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"[OK] Sample model trained — Test accuracy: {accuracy:.2%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved to {MODEL_PATH}")

    return model


def check_model():
    """Check if the model file exists."""
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[OK] Model loaded successfully — Type: {type(model).__name__}")
    else:
        print(f"[INFO] No model found at {MODEL_PATH}")
        print("[INFO] Run this script to create a sample model, or place your trained .pkl file in data/")
        create_sample_model()


if __name__ == "__main__":
    check_model()
