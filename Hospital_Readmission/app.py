from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from preprocessing.preprocess import apply_feature_engineering

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = 'models/readmission_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None
    print("Warning: Model or Scaler not found. Please run training script.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from form
        data = {
            'Age': int(request.form['Age']),
            'Gender': int(request.form['Gender']),
            'Admission_Type': int(request.form['Admission_Type']),
            'Primary_Diagnosis': int(request.form['Primary_Diagnosis']),
            'Length_of_Stay': int(request.form['Length_of_Stay']),
            'Num_Lab_Procedures': int(request.form['Num_Lab_Procedures']),
            'Num_Medications': int(request.form['Num_Medications']),
            'Num_Procedures': int(request.form['Num_Procedures']),
            'Number_Diagnoses': int(request.form['Number_Diagnoses']),
            'Number_Outpatient_Visits': int(request.form['Number_Outpatient_Visits']),
            'Number_Emergency_Visits': int(request.form['Number_Emergency_Visits']),
            'Number_Inpatient_Visits': int(request.form['Number_Inpatient_Visits']),
            'Insurance_Type': int(request.form['Insurance_Type']),
            'Discharge_Disposition': int(request.form['Discharge_Disposition']),
            'Has_Diabetes': int(request.form.get('Has_Diabetes', 0)),
            'Has_Hypertension': int(request.form.get('Has_Hypertension', 0)),
            'Has_Heart_Disease': int(request.form.get('Has_Heart_Disease', 0)),
            'Has_Kidney_Disease': int(request.form.get('Has_Kidney_Disease', 0)),
            'BMI': int(request.form['BMI']),
            'Smoking_Status': int(request.form['Smoking_Status']),
            'Follow_Up_Scheduled': int(request.form.get('Follow_Up_Scheduled', 0)),
            'Medication_Adherence_Score': int(request.form['Medication_Adherence_Score']),
            'Billing_Amount': int(request.form['Billing_Amount'])
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        df_processed = apply_feature_engineering(df)
        
        # Scale
        X_scaled = scaler.transform(df_processed)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'Readmitted' if prediction == 1 else 'Not Readmitted'
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
