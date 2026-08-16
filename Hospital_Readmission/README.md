# Hospital Readmission Prediction App

This application predicts the likelihood of a patient being readmitted to the hospital within 30 days based on clinical and demographic data.

## Features
- **Machine Learning Model**: LightGBM Classifier trained with class imbalance handling (`scale_pos_weight`).
- **Feature Engineering**: Automated feature creation including `Total_Visits`, `HighRiskPatient`, and `Treatment_Intensity`.
- **Web Interface**: Clean, user-friendly dashboard for inputting patient data and viewing predictions.

## Project Structure
```
Hospital_Readmission/
│
├── app.py                         # Main Flask application
├── models/
│   ├── readmission_model.pkl     # Trained LightGBM model
│   └── scaler.pkl                # Fitted StandardScaler
├── preprocessing/
│   └── preprocess.py              # Feature engineering + preprocessing
├── templates/
│   ├── index.html                 # Input form / dashboard
│   └── result.html                # Prediction result
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── data/
│   └── Hospital_Readmission.csv   # Dataset (optional)
├── requirements.txt
└── README.md
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to `http://localhost:5000`.

## Model Performance
The model was improved to handle class imbalance, resulting in:
- **Accuracy**: ~87%
- **Recall (Class 1)**: ~82%
- **F1-Score (Class 1)**: ~72%

*Note: The current model artifacts were trained on a synthetic dataset mimicking the structure of the original data. For production use, retrain the model on your actual dataset.*
