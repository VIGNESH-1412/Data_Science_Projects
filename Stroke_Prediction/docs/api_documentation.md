# Stroke Prediction System — API Documentation

## Base URL
```
http://127.0.0.1:5000
```

---

## Endpoints

### 1. Health Check
**GET** `/health`

Returns the application health status and model loading state.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "modelVersion": "XGBoost v1.0",
  "modelDataset": "BRFSS 2022"
}
```

---

### 2. Model Info
**GET** `/model-info`

Returns detailed model metadata including version, dataset, and feature list.

**Response:**
```json
{
  "modelVersion": "XGBoost v1.0",
  "modelDataset": "BRFSS 2022",
  "featureCount": 10,
  "features": [
    "GeneralHealth",
    "HeartAttackHistory",
    "CoronaryHeartDisease",
    "Diabetes",
    "Gender",
    "Age",
    "BMI",
    "PhysicalActivity",
    "SmokingStatus",
    "HeavyAlcoholUse"
  ],
  "modelLoaded": true,
  "lastModelLoad": "2026-08-01T12:00:00Z"
}
```

---

### 3. Feature Importance
**GET** `/feature-importance`

Returns the feature importance weights from the trained model.

**Response:**
```json
{
  "HeartAttackHistory": 0.3555,
  "GeneralHealth": 0.1849,
  "Diabetes": 0.1716,
  "CoronaryHeartDisease": 0.1030,
  "Age": 0.0773,
  "SmokingStatus": 0.0281,
  "PhysicalActivity": 0.0234,
  "Gender": 0.0203,
  "BMI": 0.0185,
  "HeavyAlcoholUse": 0.0174
}
```

---

### 4. Predict Stroke Risk
**POST** `/predict`

Runs a stroke risk prediction using the trained model. Returns risk score, confidence, clinical insights, and audit metadata.

**Request Body (JSON):**
```json
{
  "name": "Aarav Sharma",
  "age": 68,
  "gender": 1,
  "generalHealth": 3,
  "heartAttackHistory": 1,
  "coronaryHeartDisease": 0,
  "diabetes": 1,
  "physicalActivity": 2,
  "smokingStatus": 4,
  "heavyAlcoholUse": 1,
  "bmi": 29.5
}
```

**Response:**
```json
{
  "patientId": "PT-1043",
  "patient": "Aarav Sharma",
  "riskPercentage": 91.0,
  "riskLevel": "High",
  "confidence": 94.2,
  "contributingFactors": [
    "Previous heart attack",
    "Diabetes",
    "Age above 60",
    "Elevated BMI",
    "Physical inactivity"
  ],
  "recommendations": [
    "Schedule a cardiology review and adhere to prescribed cardiac medication.",
    "Maintain HbA1c below 7% with regular glucose monitoring.",
    "Target 150 minutes of moderate aerobic activity per week.",
    "Aim for a 5-10% weight reduction with a dietitian-led plan."
  ],
  "nextSteps": "Refer for urgent neurology / stroke-clinic assessment.",
  "featureImportance": { ... },
  "modelUsed": "XGBoost v1.0",
  "modelVersion": "XGBoost v1.0",
  "inferenceTimestamp": "2026-08-01T12:30:45Z",
  "date": "2026-08-01"
}
```

---

### 5. Prediction History (Audit Log)
**GET** `/prediction-history`

Returns the full prediction audit log with all stored predictions including model version and inference timestamps.

**Response:** Array of prediction records (see predict response format).

---

### 6. Download PDF Report
**GET** `/download-pdf/<patient_id>`

Generates and downloads a PDF report for a specific prediction.

**Response:** PDF file download

**Example:** `GET /download-pdf/PT-1043`

---

### 7. Download PDF (Inline POST)
**POST** `/download-pdf-inline`

Generates a PDF report from client-side prediction data without requiring the prediction to be in the audit log.

**Request Body (JSON):**
```json
{
  "patientId": "PT-TEST",
  "name": "Test Patient",
  "age": 55,
  "riskPercentage": 75,
  "riskLevel": "High",
  "confidence": 92.5,
  "date": "2026-08-01",
  "modelVersion": "XGBoost v1.0",
  "inferenceTimestamp": "2026-08-01T12:00:00Z",
  "gender": "Male",
  "bmi": 28.5,
  "contributingFactors": ["Heart attack history", "Age above 60"],
  "recommendations": ["Schedule a cardiology review"],
  "nextSteps": "Refer for urgent neurology / stroke-clinic assessment."
}
```

**Response:** PDF file download

---

### 8. Export CSV
**GET** `/export-csv`

Downloads the complete prediction history as a CSV file with all audit fields.

**Response:** CSV file download

**CSV Columns:**
Patient ID, Name, Age, Gender, BMI, Risk %, Risk Level, Confidence %, Model Version, Inference Timestamp, Prediction Date, Contributing Factors, Recommendations, Next Steps

---

## Error Responses

| Status Code | Description |
|---|---|
| 400 | Missing or invalid input data |
| 404 | Prediction record not found (PDF endpoint) |
| 500 | Internal server error |

---

## Field Encodings Reference

| Field | Value | Meaning |
|---|---|---|
| `gender` | 1 | Male |
| `gender` | 2 | Female |
| `generalHealth` | 1 | Excellent |
| `generalHealth` | 2 | Very good |
| `generalHealth` | 3 | Good |
| `generalHealth` | 4 | Fair |
| `generalHealth` | 5 | Poor |
| `heartAttackHistory` | 1 | Yes |
| `heartAttackHistory` | 2 | No |
| `coronaryHeartDisease` | 1 | Yes |
| `coronaryHeartDisease` | 2 | No |
| `diabetes` | 1 | Yes |
| `diabetes` | 3 | No |
| `physicalActivity` | 1 | Active (past 30 days) |
| `physicalActivity` | 2 | Inactive |
| `smokingStatus` | 1 | Smokes daily |
| `smokingStatus` | 2 | Smokes some days |
| `smokingStatus` | 3 | Former smoker |
| `smokingStatus` | 4 | Never smoked |
| `heavyAlcoholUse` | 1 | No |
| `heavyAlcoholUse` | 2 | Yes |
