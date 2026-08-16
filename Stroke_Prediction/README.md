# AI Stroke Prediction System

> Clinical Stroke Risk Dashboard — AI-Powered Clinical Decision Support

---

## Project Structure

```
Stroke_Prediction_System/
├── app.py                          # Flask backend (main application)
├── load_model.py                   # Model loader / sample model generator
├── requirements.txt                # Python dependencies
├── run.sh                          # Quick-start shell script
├── .gitignore                      # Git ignore rules
│
├── templates/
│   └── stroke_dashboard.html       # Frontend UI (single-file HTML)
│
├── static/
│   ├── css/                        # Custom stylesheets (optional)
│   ├── js/                         # Custom JavaScript (optional)
│   └── img/                        # Image assets (optional)
│
├── data/
│   └── Stroke_Prediction.pkl       # Trained XGBoost model (place your .pkl here)
│
└── docs/
    └── api_documentation.md        # API endpoint documentation
```

---

## Features

### Core Features
- **10 clinical inputs** used by the trained XGBoost model
- **Stroke risk scoring** with percentage and risk level (Low/Moderate/High)
- **Model confidence** score with visual gauge
- **AI Insights** with contributing factors, recommendations, and next medical steps
- **Feature importance** bar chart visualization
- **Risk distribution** donut chart

### New Features (v2.0)
- **Model Version & Inference Timestamp Logging** — Every prediction is audited with model version, inference timestamp, and unique patient ID
- **Downloadable PDF Reports** — Generate professional PDF reports per prediction including risk score, confidence, AI insights, and next medical steps
- **CSV Export** — Download the complete prediction history as CSV with all audit fields

---

## Quick Start

### Option 1: Using the Shell Script

```bash
cd Stroke_Prediction_System
bash run.sh
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample model (or place your trained .pkl in data/)
python3 load_model.py

# 3. Run the application
python3 app.py
```

### Option 3: Using Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 load_model.py
python3 app.py
```

**Open:** `http://127.0.0.1:5000`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the dashboard UI |
| `/health` | GET | Health check with model status |
| `/model-info` | GET | Model version, dataset, and feature metadata |
| `/feature-importance` | GET | Feature importance weights |
| `/predict` | POST | Run stroke prediction (returns risk, confidence, audit data) |
| `/prediction-history` | GET | Full prediction audit log |
| `/download-pdf/<patient_id>` | GET | Download PDF report for a prediction |
| `/download-pdf-inline` | POST | Generate PDF from client-side data |
| `/export-csv` | GET | Download full prediction history as CSV |

---

## Prediction Input Format

Send a POST request to `/predict` with JSON:

```json
{
  "name": "Patient Name",
  "age": 55,
  "gender": 1,
  "generalHealth": 3,
  "heartAttackHistory": 2,
  "coronaryHeartDisease": 2,
  "diabetes": 3,
  "physicalActivity": 1,
  "smokingStatus": 4,
  "heavyAlcoholUse": 1,
  "bmi": 27
}
```

### Field Encodings

| Field | Values |
|---|---|
| `gender` | 1 = Male, 2 = Female |
| `generalHealth` | 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor |
| `heartAttackHistory` | 1 = Yes, 2 = No |
| `coronaryHeartDisease` | 1 = Yes, 2 = No |
| `diabetes` | 1 = Yes, 3 = No |
| `physicalActivity` | 1 = Active, 2 = Inactive |
| `smokingStatus` | 1 = Daily, 2 = Some days, 3 = Former, 4 = Never |
| `heavyAlcoholUse` | 1 = No, 2 = Yes |

---

## PDF Report Contents

Each PDF report includes:
1. Patient Information (ID, Name, Age, Gender, BMI, Date)
2. Prediction Results (Risk Score, Risk Level, Confidence, Model Version, Timestamp)
3. Contributing Risk Factors
4. Clinical Recommendations
5. Next Medical Steps
6. Disclaimer

---

## CSV Export Fields

The CSV export includes all audit fields:
- Patient ID, Name, Age, Gender, BMI
- Risk %, Risk Level, Confidence %
- Model Version, Inference Timestamp, Prediction Date
- Contributing Factors, Recommendations, Next Steps

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Stroke_Prediction.pkl` | Path to the .pkl model file |
| `MODEL_VERSION` | `XGBoost v1.0` | Model version identifier |
| `MODEL_DATASET` | `BRFSS 2022` | Training dataset name |
| `PORT` | `5000` | Server port |

---

## Deployment

### Local / Development
```bash
python3 app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```bash
# Dockerfile (create in project root)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "app.py"]
```

---

## Tech Stack

- **Backend:** Flask (Python)
- **ML Model:** XGBoost (or rule-based fallback)
- **Frontend:** Single-file HTML + CSS + JavaScript
- **PDF Generation:** ReportLab
- **Dataset:** BRFSS 2022
- **Accuracy:** 94.6% (ROC AUC: 0.96)

---

## License

This project is for clinical decision support purposes. All AI-generated predictions should be reviewed by qualified healthcare professionals.
