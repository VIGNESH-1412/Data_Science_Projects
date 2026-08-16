import pandas as pd
import numpy as np

def apply_feature_engineering(df):
    """
    Applies the same feature engineering as used during training.
    """
    # Total Visits
    df["Total_Visits"] = (
        df["Number_Outpatient_Visits"] +
        df["Number_Emergency_Visits"] +
        df["Number_Inpatient_Visits"]
    )
    
    # Total Treatment
    df["Total_Treatment"] = (
        df["Num_Lab_Procedures"] +
        df["Num_Procedures"]
    )
    
    # Treatment Intensity
    df["Treatment_Intensity"] = (
        df["Num_Lab_Procedures"] +
        df["Num_Medications"] +
        df["Num_Procedures"]
    )
    
    # High Risk Patient
    def high_risk_patient(row):
        if row["Age"] >= 65 and row["Number_Inpatient_Visits"] >= 2:
            return 1
        else:
            return 0
    
    df["HighRiskPatient"] = df.apply(high_risk_patient, axis=1)
    
    # Frequent Visitor
    def frequent_visitor(row):
        total_visits = (
            row["Number_Outpatient_Visits"] +
            row["Number_Emergency_Visits"] +
            row["Number_Inpatient_Visits"]
        )
        if total_visits >= 5:
            return 1
        else:
            return 0
            
    df["FrequentVisitor"] = df.apply(frequent_visitor, axis=1)
    
    # Long Stay
    def long_stay(row):
        if row["Length_of_Stay"] >= 7:
            return 1
        else:
            return 0
            
    df["LongStay"] = df.apply(long_stay, axis=1)
    
    # Ensure correct feature order
    feature_order = [
        'Age', 'Gender', 'Admission_Type', 'Primary_Diagnosis', 'Length_of_Stay',
        'Num_Lab_Procedures', 'Num_Medications', 'Num_Procedures', 'Number_Diagnoses',
        'Number_Outpatient_Visits', 'Number_Emergency_Visits', 'Number_Inpatient_Visits',
        'Insurance_Type', 'Discharge_Disposition', 'Has_Diabetes', 'Has_Hypertension',
        'Has_Heart_Disease', 'Has_Kidney_Disease', 'BMI', 'Smoking_Status',
        'Follow_Up_Scheduled', 'Medication_Adherence_Score', 'Billing_Amount',
        'Total_Visits', 'Total_Treatment', 'Treatment_Intensity',
        'HighRiskPatient', 'FrequentVisitor', 'LongStay'
    ]
    
    return df[feature_order]
