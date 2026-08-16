"""
train_model.py — Train the ICU mortality prediction model.

Generates synthetic ICU patient data and trains a gradient-boosted tree
classifier. The trained pipeline is saved as model/mortality_model.pkl.

Run before deploying to Vercel:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# ─── Configuration ───────────────────────────────────────────────────────
N_SAMPLES = 5000
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Categorical mappings ────────────────────────────────────────────────
BODY_SYSTEMS = [
    "Cardiovascular", "Gastrointestinal", "Genitourinary",
    "Gynecological", "Hematological", "Metabolic",
    "Musculoskeletal/Skin", "Neurological", "Respiratory",
    "Sepsis", "Trauma"
]
ADMIT_SOURCES = [
    "Accident & Emergency", "Floor",
    "Operating Room / Recovery", "Other Hospital", "Other ICU"
]

# ─── Feature columns ─────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "bmi",
    "gcs_motor_apache", "gcs_eyes_apache", "gcs_verbal_apache",
    "d1_heartrate_max", "d1_heartrate_min",
    "d1_sysbp_min", "d1_mbp_min", "d1_spo2_min",
    "d1_resprate_max", "d1_temp_min", "d1_temp_max",
    "d1_lactate_min", "d1_lactate_max",
    "d1_bun_max", "d1_creatinine_max",
    "d1_albumin_min",
    "d1_platelets_min", "d1_wbc_max",
    "d1_glucose_max", "d1_sodium_min",
    "ph_apache", "d1_arterial_ph_min",
    "apache_2_diagnosis",
    "apache_4a_hospital_death_prob", "apache_4a_icu_death_prob",
    "aids", "cirrhosis", "hepatic_failure", "immunosuppression",
    "leukemia", "lymphoma", "solid_tumor_with_metastasis",
    "diabetes_mellitus", "arf_apache", "ventilated_apache",
    "elective_surgery", "apache_post_operative"
]

CATEGORICAL_FEATURES = ["gender", "apache_3j_bodysystem", "icu_admit_source"]

OPTIONAL_LABS = [
    "d1_lactate_min", "d1_lactate_max",
    "d1_albumin_min", "ph_apache", "d1_arterial_ph_min"
]


def generate_data(n):
    """Generate synthetic ICU patient data with realistic mortality signals."""
    data = {}

    # Age
    data["age"] = np.random.normal(62, 16, n).clip(16, 110).astype(int)

    # Gender
    data["gender"] = np.random.choice(["M", "F"], n)

    # BMI
    data["bmi"] = np.random.normal(27, 7, n).clip(12, 70).round(1)

    # GCS components (lower = worse)
    data["gcs_motor_apache"] = np.random.choice(range(1, 7), n, p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
    data["gcs_eyes_apache"] = np.random.choice(range(1, 5), n, p=[0.1, 0.15, 0.3, 0.45])
    data["gcs_verbal_apache"] = np.random.choice(range(1, 6), n, p=[0.05, 0.1, 0.2, 0.3, 0.35])

    # Heart rate
    data["d1_heartrate_min"] = np.random.normal(78, 18, n).clip(20, 250).astype(int)
    data["d1_heartrate_max"] = (data["d1_heartrate_min"] + np.random.exponential(25, n)).clip(None, 250).astype(int)

    # Blood pressure
    data["d1_sysbp_min"] = np.random.normal(110, 30, n).clip(40, 260).astype(int)
    data["d1_mbp_min"] = np.random.normal(75, 20, n).clip(30, 200).astype(int)

    # SpO2
    data["d1_spo2_min"] = np.random.normal(93, 6, n).clip(50, 100).astype(int)

    # Respiratory rate
    data["d1_resprate_max"] = np.random.normal(22, 8, n).clip(4, 80).astype(int)

    # Temperature
    data["d1_temp_min"] = np.random.normal(36.0, 1.5, n).clip(25, 42).round(1)
    data["d1_temp_max"] = (data["d1_temp_min"] + np.random.exponential(1.0, n)).clip(None, 43).round(1)

    # Optional labs (~30% missing)
    for col in OPTIONAL_LABS:
        present = np.random.random(n) > 0.30
        if col in ["d1_lactate_min", "d1_lactate_max"]:
            vals = np.random.lognormal(0.2, 0.7, n).clip(0, 30).round(1)
        elif col == "d1_albumin_min":
            vals = np.random.normal(3.0, 0.8, n).clip(0.5, 6).round(1)
        else:  # pH fields
            vals = np.random.normal(7.35, 0.15, n).clip(6.5, 7.8).round(2)
        vals[~present] = np.nan
        data[col] = vals

    # Required labs
    data["d1_bun_max"] = np.random.lognormal(1.5, 0.6, n).clip(1, 200).round(1)
    data["d1_creatinine_max"] = np.random.lognormal(0.0, 0.5, n).clip(0.1, 15).round(2)
    data["d1_platelets_min"] = np.random.lognormal(5.2, 0.6, n).clip(5, 1000).astype(int)
    data["d1_wbc_max"] = np.random.lognormal(2.2, 0.5, n).clip(0, 100).round(1)
    data["d1_glucose_max"] = np.random.lognormal(4.8, 0.4, n).clip(20, 1000).astype(int)
    data["d1_sodium_min"] = np.random.normal(138, 6, n).clip(100, 170).round(1)

    # APACHE context
    data["apache_2_diagnosis"] = np.random.randint(1, 800, n)
    data["apache_3j_bodysystem"] = np.random.choice(BODY_SYSTEMS, n)
    data["icu_admit_source"] = np.random.choice(ADMIT_SOURCES, n)

    # APACHE probabilities
    data["apache_4a_hospital_death_prob"] = np.random.beta(1, 4, n).round(3)
    data["apache_4a_icu_death_prob"] = np.random.beta(1, 4, n).round(3)

    # Comorbidities (mostly 0)
    for col in ["aids", "cirrhosis", "hepatic_failure", "immunosuppression",
                "leukemia", "lymphoma", "solid_tumor_with_metastasis",
                "diabetes_mellitus", "arf_apache", "ventilated_apache",
                "elective_surgery", "apache_post_operative"]:
        prob = 0.08 if col not in ["elective_surgery", "apache_post_operative"] else 0.25
        data[col] = np.random.binomial(1, prob, n)

    df = pd.DataFrame(data)

    # ─── Generate mortality label with realistic signals ──────────────────
    logit = (
        +0.03 * (df["age"] - 60)
        -0.15 * df["gcs_motor_apache"]
        -0.10 * df["gcs_eyes_apache"]
        -0.12 * df["gcs_verbal_apache"]
        +0.008 * (df["d1_heartrate_max"] - 80)
        -0.02 * (df["d1_spo2_min"] - 90)
        +0.003 * (df["d1_bun_max"] - 20)
        -0.10 * (df["d1_platelets_min"] / 1000)
        +0.5 * df["hepatic_failure"]
        +0.4 * df["cirrhosis"]
        +0.6 * df["solid_tumor_with_metastasis"]
        +0.3 * df["arf_apache"]
        +0.25 * df["ventilated_apache"]
        +0.15 * df["diabetes_mellitus"]
        +0.8 * df["apache_4a_hospital_death_prob"]
        +0.5 * df["apache_4a_icu_death_prob"]
    )
    logit = logit.fillna(0)
    prob_death = 1 / (1 + np.exp(-logit))
    df["hospital_death"] = (np.random.random(n) < prob_death).astype(int)

    return df


def main():
    print("Generating synthetic ICU data...")
    df = generate_data(N_SAMPLES)
    print(f"Dataset shape: {df.shape}")
    print(f"Mortality rate: {df['hospital_death'].mean():.2%}")

    # Save training data
    train_path = os.path.join("data", "train.csv")
    df.to_csv(train_path, index=False)
    print(f"Training data saved to {train_path}")

    # Save a small sample
    sample_path = os.path.join("data", "sample.csv")
    df.head(10).to_csv(sample_path, index=False)
    print(f"Sample data saved to {sample_path}")

    # ─── Prepare features ─────────────────────────────────────────────────
    X = df.drop(columns=["hospital_death"]).copy()
    y = df["hospital_death"]

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(X[col].fillna("Unknown"))
        X[col] = le.transform(X[col].fillna("Unknown"))
        label_encoders[col] = le

    # Fill NaN optional labs with -1 (sentinel for missing)
    for col in OPTIONAL_LABS:
        X[col] = X[col].fillna(-1)

    # ─── Train model ──────────────────────────────────────────────────────
    print("Training Gradient Boosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE
    )

    # Evaluate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full data
    model.fit(X, y)
    print("Model trained successfully.")

    # ─── Save artifacts ───────────────────────────────────────────────────
    artifact = {
        "model": model,
        "label_encoders": label_encoders,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "optional_labs": OPTIONAL_LABS,
        "threshold": 0.35,
        "feature_order": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }

    model_path = os.path.join("model", "mortality_model.pkl")
    joblib.dump(artifact, model_path)
    print(f"Model saved to {model_path}")
    print(f"Features ({len(artifact['feature_order'])}): {artifact['feature_order']}")


if __name__ == "__main__":
    main()
