"""
Stroke Prediction System - Enhanced Flask Backend
==================================================
Serves the trained XGBoost model (Stroke_Prediction.pkl) built from the
BRFSS 2022 dataset.

NEW FEATURES (v2.0):
  1. Model version + inference timestamp logging for audit trails
  2. PDF report generation per prediction (risk, confidence, insights, next steps)
  3. CSV export endpoint for full prediction history

Run:
    pip install flask flask-cors xgboost scikit-learn pandas numpy reportlab
    python app.py
    -> http://127.0.0.1:5000
"""

import csv
import io
import json
import os
import pickle
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "Stroke_Prediction.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "XGBoost v1.0")
MODEL_DATASET = os.environ.get("MODEL_DATASET", "BRFSS 2022")

FEATURES = [
    "GeneralHealth",
    "HeartAttackHistory",
    "CoronaryHeartDisease",
    "Diabetes",
    "Gender",
    "Age",
    "BMI",
    "PhysicalActivity",
    "SmokingStatus",
    "HeavyAlcoholUse",
]

FEATURE_IMPORTANCE = {
    "HeartAttackHistory": 0.3555,
    "GeneralHealth": 0.1849,
    "Diabetes": 0.1716,
    "CoronaryHeartDisease": 0.1030,
    "Age": 0.0773,
    "SmokingStatus": 0.0281,
    "PhysicalActivity": 0.0234,
    "Gender": 0.0203,
    "BMI": 0.0185,
    "HeavyAlcoholUse": 0.0174,
}

# ---------------------------------------------------------------------------
# In-memory prediction audit log
# ---------------------------------------------------------------------------
PREDICTION_LOG = []  # list of dicts; acts as audit trail

# Seed with demo history data matching the original dashboard
DEMO_HISTORY = [
    {"patientId": "PT-1042", "name": "Aarav Sharma", "age": 68, "risk": 91, "level": "High",
     "date": "2026-07-31", "confidence": 94.2, "modelVersion": "XGBoost v1.0",
     "inferenceTimestamp": "2026-07-31T10:23:15Z"},
    {"patientId": "PT-1041", "name": "Meera Iyer", "age": 54, "risk": 42, "level": "Moderate",
     "date": "2026-07-31", "confidence": 87.5, "modelVersion": "XGBoost v1.0",
     "inferenceTimestamp": "2026-07-31T09:15:42Z"},
    {"patientId": "PT-1040", "name": "John Carter", "age": 47, "risk": 18, "level": "Low",
     "date": "2026-07-30", "confidence": 91.8, "modelVersion": "XGBoost v1.0",
     "inferenceTimestamp": "2026-07-30T14:07:33Z"},
    {"patientId": "PT-1039", "name": "Sara Ahmed", "age": 72, "risk": 77, "level": "High",
     "date": "2026-07-30", "confidence": 92.1, "modelVersion": "XGBoost v1.0",
     "inferenceTimestamp": "2026-07-30T11:42:09Z"},
    {"patientId": "PT-1038", "name": "Liu Wei", "age": 39, "risk": 12, "level": "Low",
     "date": "2026-07-29", "confidence": 89.3, "modelVersion": "XGBoost v1.0",
     "inferenceTimestamp": "2026-07-29T16:30:21Z"},
]
PREDICTION_LOG.extend(DEMO_HISTORY)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as exc:  # pragma: no cover
    model = None
    print(f"[WARN] Could not load model ({exc}). Falling back to rule-based scoring.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_patient_id():
    """Generate next patient ID based on existing log."""
    max_id = 1037
    for entry in PREDICTION_LOG:
        pid = entry.get("patientId", "")
        if pid.startswith("PT-"):
            try:
                num = int(pid.split("-")[1])
                if num > max_id:
                    max_id = num
            except (ValueError, IndexError):
                pass
    return f"PT-{max_id + 1}"


def build_frame(payload: dict) -> pd.DataFrame:
    """Validate the incoming JSON and build the model input frame."""
    row = {
        "GeneralHealth": float(payload.get("generalHealth", 3)),
        "HeartAttackHistory": float(payload.get("heartAttackHistory", 2)),
        "CoronaryHeartDisease": float(payload.get("coronaryHeartDisease", 2)),
        "Diabetes": float(payload.get("diabetes", 3)),
        "Gender": float(payload.get("gender", 1)),
        "Age": float(payload.get("age", 50)),
        "BMI": float(payload.get("bmi", 25)) * 100.0,
        "PhysicalActivity": float(payload.get("physicalActivity", 1)),
        "SmokingStatus": float(payload.get("smokingStatus", 4)),
        "HeavyAlcoholUse": float(payload.get("heavyAlcoholUse", 1)),
    }
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)


def fallback_score(payload: dict) -> float:
    """Weighted rule-based estimate used when the .pkl file is unavailable."""
    s = 0.0
    s += 35 if int(payload.get("heartAttackHistory", 2)) == 1 else 0
    s += (float(payload.get("generalHealth", 3)) - 1) * 4.6
    s += 17 if int(payload.get("diabetes", 3)) == 1 else 0
    s += 10 if int(payload.get("coronaryHeartDisease", 2)) == 1 else 0
    s += max(0.0, (float(payload.get("age", 50)) - 30) / 50) * 12
    smoke = int(payload.get("smokingStatus", 4))
    s += {1: 6, 2: 4, 3: 2}.get(smoke, 0)
    s += 5 if int(payload.get("physicalActivity", 1)) == 2 else 0
    bmi = float(payload.get("bmi", 25))
    s += 4 if bmi >= 30 else (2 if bmi >= 25 else 0)
    s += 2 if int(payload.get("gender", 1)) == 1 else 0
    s += 3.5 if int(payload.get("heavyAlcoholUse", 1)) == 2 else 0
    return float(np.clip(s, 2, 97))


def build_insights(payload: dict, risk: float) -> dict:
    """Build clinical insights from payload and risk score."""
    factors, tips = [], []

    if int(payload.get("heartAttackHistory", 2)) == 1:
        factors.append("Previous heart attack")
        tips.append("Schedule a cardiology review and adhere to prescribed cardiac medication.")
    if int(payload.get("coronaryHeartDisease", 2)) == 1:
        factors.append("Coronary heart disease")
    if int(payload.get("diabetes", 3)) == 1:
        factors.append("Diabetes")
        tips.append("Maintain HbA1c below 7% with regular glucose monitoring.")
    if float(payload.get("generalHealth", 3)) >= 4:
        factors.append("Poor self-rated general health")
        tips.append("Book a full health assessment to address declining general health.")
    if float(payload.get("age", 50)) >= 60:
        factors.append("Age above 60")
    if int(payload.get("smokingStatus", 4)) in (1, 2):
        factors.append("Current smoker")
        tips.append("Begin a supervised smoking-cessation programme.")
    if int(payload.get("physicalActivity", 1)) == 2:
        factors.append("Physical inactivity")
        tips.append("Target 150 minutes of moderate aerobic activity per week.")
    if float(payload.get("bmi", 25)) >= 25:
        factors.append("Elevated BMI")
        tips.append("Aim for a 5-10% weight reduction with a dietitian-led plan.")
    if int(payload.get("heavyAlcoholUse", 1)) == 2:
        factors.append("Heavy alcohol use")
        tips.append("Reduce alcohol intake to within national safe limits.")

    if not tips:
        tips.append("Maintain current lifestyle and repeat screening in 12 months.")

    level = "High" if risk >= 60 else ("Moderate" if risk >= 30 else "Low")
    next_steps = (
        "Refer for urgent neurology / stroke-clinic assessment."
        if level == "High"
        else "Arrange a follow-up review within 3 months."
        if level == "Moderate"
        else "Routine annual screening is sufficient."
    )

    return {
        "level": level,
        "factors": factors or ["No major clinical risk factors detected"],
        "recommendations": tips,
        "nextSteps": next_steps,
    }


# ---------------------------------------------------------------------------
# PDF Report Generator (ReportLab)
# ---------------------------------------------------------------------------
def generate_pdf_report(record: dict) -> bytes:
    """Generate a professional PDF report for a single prediction."""
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#7c3aed"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=16,
        spaceAfter=8,
        borderWidth=0,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#374151"),
        spaceAfter=4,
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    elements = []

    # Header
    elements.append(Paragraph("Stroke Prediction Report", title_style))
    elements.append(Paragraph(
        "AI Clinical Decision Support System &mdash; Generated by XGBoost Model",
        subtitle_style,
    ))
    elements.append(Spacer(1, 4 * mm))

    # Divider
    elements.append(Paragraph("_" * 78, ParagraphStyle("hr", parent=styles["Normal"],
                 fontSize=8, textColor=colors.HexColor("#e5e7eb"))))
    elements.append(Spacer(1, 6 * mm))

    # Patient Info Section
    elements.append(Paragraph("Patient Information", section_style))
    patient_data = [
        ["Patient ID:", record.get("patientId", "N/A")],
        ["Patient Name:", record.get("name", "Anonymous")],
        ["Age:", str(record.get("age", "N/A"))],
        ["Gender:", record.get("gender", "N/A")],
        ["BMI:", str(record.get("bmi", "N/A"))],
        ["Prediction Date:", record.get("date", "N/A")],
    ]
    pt_table = Table(patient_data, colWidths=[120, 280])
    pt_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b7280")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1f2937")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(pt_table)
    elements.append(Spacer(1, 6 * mm))

    # Prediction Results Section
    elements.append(Paragraph("Prediction Results", section_style))

    risk_level = record.get("level", "N/A")
    risk_color = {"High": "#dc2626", "Moderate": "#f59e0b", "Low": "#22c55e"}.get(risk_level, "#6b7280")

    results_data = [
        ["Metric", "Value"],
        ["Stroke Risk Score", f"{record.get('risk', 'N/A')}%"],
        ["Risk Level", risk_level],
        ["Model Confidence", f"{record.get('confidence', 'N/A')}%"],
        ["Model Used", record.get("modelVersion", "N/A")],
        ["Inference Timestamp", record.get("inferenceTimestamp", "N/A")],
    ]
    res_table = Table(results_data, colWidths=[160, 240])
    res_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f3ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6b7280")),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor(risk_color)),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
    ]))
    elements.append(res_table)
    elements.append(Spacer(1, 6 * mm))

    # Contributing Factors
    elements.append(Paragraph("Contributing Risk Factors", section_style))
    factors = record.get("contributingFactors", [])
    if factors:
        for i, factor in enumerate(factors, 1):
            elements.append(Paragraph(f"{i}. {factor}", body_style))
    else:
        elements.append(Paragraph("No major clinical risk factors detected.", body_style))
    elements.append(Spacer(1, 4 * mm))

    # Recommendations
    elements.append(Paragraph("Clinical Recommendations", section_style))
    recommendations = record.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", body_style))
    else:
        elements.append(Paragraph("No specific recommendations at this time.", body_style))
    elements.append(Spacer(1, 6 * mm))

    # Next Medical Steps
    elements.append(Paragraph("Next Medical Steps", section_style))
    next_steps = record.get("nextSteps", "Follow up as clinically indicated.")
    elements.append(Paragraph(f"<b>{next_steps}</b>", body_style))
    elements.append(Spacer(1, 10 * mm))

    # Disclaimer
    elements.append(Spacer(1, 20 * mm))
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#9ca3af"),
        alignment=1,
    )
    elements.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-powered clinical decision support system "
        "and is intended to assist healthcare professionals. It should not replace professional "
        "medical judgment. All predictions should be reviewed by a qualified clinician.",
        disclaimer_style,
    ))
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        f"Report generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"Model: {record.get('modelVersion', 'N/A')}",
        disclaimer_style,
    ))

    doc.build(elements)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CSV Export Generator
# ---------------------------------------------------------------------------
def generate_csv_export():
    """Generate a CSV export of the full prediction history."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Patient ID", "Name", "Age", "Gender", "BMI",
        "Risk %", "Risk Level", "Confidence %",
        "Model Version", "Inference Timestamp", "Prediction Date",
        "Contributing Factors", "Recommendations", "Next Steps"
    ])
    for entry in PREDICTION_LOG:
        writer.writerow([
            entry.get("patientId", ""),
            entry.get("name", ""),
            entry.get("age", ""),
            entry.get("gender", ""),
            entry.get("bmi", ""),
            entry.get("risk", ""),
            entry.get("level", ""),
            entry.get("confidence", ""),
            entry.get("modelVersion", ""),
            entry.get("inferenceTimestamp", ""),
            entry.get("date", ""),
            "; ".join(entry.get("contributingFactors", [])),
            "; ".join(entry.get("recommendations", [])),
            entry.get("nextSteps", ""),
        ])
    return output.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    try:
        return render_template("stroke_dashboard.html")
    except Exception:
        return jsonify({"message": "Stroke Prediction API running. POST /predict"})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "modelVersion": MODEL_VERSION,
        "modelDataset": MODEL_DATASET,
    })


@app.route("/model-info")
def model_info():
    """Return current model metadata including version info."""
    return jsonify({
        "modelVersion": MODEL_VERSION,
        "modelDataset": MODEL_DATASET,
        "featureCount": len(FEATURES),
        "features": FEATURES,
        "modelLoaded": model is not None,
        "lastModelLoad": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


@app.route("/feature-importance")
def feature_importance():
    return jsonify(FEATURE_IMPORTANCE)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({"error": "No input data provided"}), 400

    try:
        inference_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if model is not None:
            frame = build_frame(payload)
            proba = model.predict_proba(frame)[0]
            risk = float(proba[0]) * 100.0
            confidence = float(np.max(proba)) * 100.0
            model_used = MODEL_VERSION
        else:
            risk = fallback_score(payload)
            confidence = 78 + abs(risk - 50) * 0.34
            model_used = f"{MODEL_VERSION} (rule-based fallback)"

        risk = round(min(max(risk, 0.0), 100.0), 2)
        insights = build_insights(payload, risk)

        patient_id = _next_patient_id()
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Map numeric codes to human-readable labels for the log
        gender_label = "Male" if int(payload.get("gender", 1)) == 1 else "Female"
        health_labels = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
        health_label = health_labels.get(int(payload.get("generalHealth", 3)), "Good")

        # Store full audit record
        audit_record = {
            "patientId": patient_id,
            "name": payload.get("name", "Anonymous"),
            "age": int(payload.get("age", 50)),
            "gender": gender_label,
            "generalHealth": health_label,
            "heartAttackHistory": "Yes" if int(payload.get("heartAttackHistory", 2)) == 1 else "No",
            "coronaryHeartDisease": "Yes" if int(payload.get("coronaryHeartDisease", 2)) == 1 else "No",
            "diabetes": "Yes" if int(payload.get("diabetes", 3)) == 1 else "No",
            "physicalActivity": "Active" if int(payload.get("physicalActivity", 1)) == 1 else "Inactive",
            "smokingStatus": {1: "Daily", 2: "Some days", 3: "Former", 4: "Never"}.get(
                int(payload.get("smokingStatus", 4)), "Never"
            ),
            "heavyAlcoholUse": "Yes" if int(payload.get("heavyAlcoholUse", 1)) == 2 else "No",
            "bmi": float(payload.get("bmi", 25)),
            "risk": risk,
            "level": insights["level"],
            "confidence": round(confidence, 2),
            "contributingFactors": insights["factors"],
            "recommendations": insights["recommendations"],
            "nextSteps": insights["nextSteps"],
            "featureImportance": FEATURE_IMPORTANCE,
            "modelVersion": model_used,
            "modelDataset": MODEL_DATASET,
            "inferenceTimestamp": inference_ts,
            "date": current_date,
        }

        PREDICTION_LOG.append(audit_record)

        return jsonify({
            "patientId": patient_id,
            "patient": payload.get("name", "Anonymous"),
            "riskPercentage": risk,
            "riskLevel": insights["level"],
            "confidence": round(confidence, 2),
            "contributingFactors": insights["factors"],
            "recommendations": insights["recommendations"],
            "nextSteps": insights["nextSteps"],
            "featureImportance": FEATURE_IMPORTANCE,
            "modelUsed": model_used,
            "modelVersion": MODEL_VERSION,
            "inferenceTimestamp": inference_ts,
            "date": current_date,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/prediction-history")
def prediction_history():
    """Return full prediction history for the audit trail."""
    return jsonify(PREDICTION_LOG)


@app.route("/download-pdf/<patient_id>")
def download_pdf(patient_id):
    """Generate and download a PDF report for a specific prediction."""
    record = None
    for entry in PREDICTION_LOG:
        if entry.get("patientId") == patient_id:
            record = entry
            break

    if record is None:
        return jsonify({"error": f"Prediction {patient_id} not found"}), 404

    pdf_bytes = generate_pdf_report(record)
    filename = f"StrokePrediction_{patient_id}_{record.get('date', 'report')}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/export-csv")
def export_csv():
    """Download full prediction history as CSV."""
    csv_data = generate_csv_export()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_history_{timestamp}.csv"

    return send_file(
        io.BytesIO(csv_data.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/download-pdf-inline", methods=["POST"])
def download_pdf_inline():
    """Generate PDF from inline POST data (for client-side generated predictions)."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "No data provided"}), 400

    record = {
        "patientId": payload.get("patientId", _next_patient_id()),
        "name": payload.get("name", "Anonymous"),
        "age": payload.get("age", "N/A"),
        "gender": payload.get("gender", "N/A"),
        "bmi": payload.get("bmi", "N/A"),
        "risk": payload.get("riskPercentage", payload.get("risk", "N/A")),
        "level": payload.get("riskLevel", payload.get("level", "N/A")),
        "confidence": payload.get("confidence", "N/A"),
        "contributingFactors": payload.get("contributingFactors", []),
        "recommendations": payload.get("recommendations", []),
        "nextSteps": payload.get("nextSteps", "N/A"),
        "modelVersion": payload.get("modelVersion", MODEL_VERSION),
        "inferenceTimestamp": payload.get("inferenceTimestamp",
                                          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        "date": payload.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
    }

    pdf_bytes = generate_pdf_report(record)
    filename = f"StrokePrediction_{record['patientId']}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
