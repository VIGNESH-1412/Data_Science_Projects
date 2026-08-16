"""
app.py — Local development Flask entry point.

For local testing, run:
    python app.py

For Vercel deployment, api/index.py is used automatically.
"""

import io
import os
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

# ─── Flask setup ──────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "mortality_model.pkl")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Load model at startup ────────────────────────────────────────────────
_artifact = None


def load_model():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python train_model.py` first."
            )
        _artifact = joblib.load(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
        print(f"       Features: {len(_artifact['feature_order'])}")
        print(f"       Threshold: {_artifact['threshold']}")
    return _artifact


def risk_band(prob, threshold=0.35):
    """Classify probability into a risk band."""
    if prob >= 0.70:
        return "Critical"
    elif prob >= threshold:
        return "High"
    elif prob >= 0.15:
        return "Moderate"
    else:
        return "Low"


def label_for_band(band):
    """Return a short human-readable label."""
    mapping = {
        "Low": "Low mortality risk — routine monitoring appropriate.",
        "Moderate": "Moderate risk — close observation recommended.",
        "High": "High risk — consider escalation of care.",
        "Critical": "Critical risk — urgent clinical review advised.",
    }
    return mapping.get(band, "Unknown risk")


def preprocess_single(payload: dict):
    """Convert a JSON payload into a feature DataFrame for the model."""
    artifact = load_model()
    le = artifact["label_encoders"]
    optional = artifact["optional_labs"]

    row = {}
    # Use the model's actual feature names in the correct order
    model_features = artifact["model"].feature_names_in_.tolist()
    for feat in model_features:
        val = payload.get(feat, "")
        if feat in optional:
            if val == "" or val is None:
                row[feat] = -1.0
            else:
                row[feat] = float(val)
        elif feat in le:
            encoder = le[feat]
            classes = set(encoder.classes_)
            classes.add("Unknown")
            if val not in classes:
                val = "Unknown"
            row[feat] = encoder.transform([val])[0]
        else:
            row[feat] = float(val) if val not in ("", None) else 0.0

    df = pd.DataFrame([row], columns=model_features)
    return df


# ─── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "model_loaded": True})
    except RuntimeError:
        return jsonify({"status": "ok", "model_loaded": False})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Empty request body"}), 400

        X = preprocess_single(payload)
        artifact = load_model()
        prob = float(artifact["model"].predict_proba(X)[0][1])
        prob = round(prob, 4)
        band = risk_band(prob, artifact["threshold"])
        label = label_for_band(band)

        return jsonify({
            "probability": prob,
            "risk_band": band,
            "label": label,
            "threshold": artifact["threshold"],
        })

    except (ValueError, KeyError, TypeError) as exc:
        traceback.print_exc()
        return jsonify({"error": f"Invalid input: {str(exc)}"}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        df = pd.read_csv(io.BytesIO(uploaded.read()))
        artifact = load_model()
        le = artifact["label_encoders"]
        optional = artifact["optional_labs"]

        results = []
        for idx, row in df.iterrows():
            model_features = artifact["model"].feature_names_in_.tolist()
            feature_row = {}
            for feat in model_features:
                val = row.get(feat)
                if feat in optional and (pd.isna(val) or val == ""):
                    feature_row[feat] = -1.0
                elif feat in le:
                    encoder = le[feat]
                    classes = set(encoder.classes_)
                    classes.add("Unknown")
                    val_str = str(val) if not pd.isna(val) and val != "" else "Unknown"
                    if val_str not in classes:
                        val_str = "Unknown"
                    feature_row[feat] = encoder.transform([val_str])[0]
                else:
                    try:
                        feature_row[feat] = float(val) if not pd.isna(val) else 0.0
                    except (ValueError, TypeError):
                        feature_row[feat] = 0.0

            X_row = pd.DataFrame([feature_row], columns=model_features)
            prob = float(artifact["model"].predict_proba(X_row)[0][1])
            prob = round(prob, 4)
            band = risk_band(prob, artifact["threshold"])
            label = label_for_band(band)
            pred = 1 if prob >= artifact["threshold"] else 0

            results.append({
                "row": int(idx + 1),
                "probability": prob,
                "prediction": pred,
                "risk_band": band,
                "label": label,
            })

        return jsonify({
            "count": len(results),
            "threshold": artifact["threshold"],
            "results": results,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Batch processing failed: {str(exc)}"}), 500


# ─── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        load_model()
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("       Run `python train_model.py` to train and save the model.")

    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting VitalSign AI on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    
