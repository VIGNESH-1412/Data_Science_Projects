import os
import joblib
import pandas as pd
import numpy as np
from app.config import Config

# Cache objects
_MODEL = None
_DF = None
_COMPANIES_MAPPING = {
    'ADANIPORTS': 1, 'ASIANPAINT': 2, 'AXISBANK': 3, 'BAJAJ-AUTO': 4, 'BAJAJFINSV': 5, 
    'BAJFINANCE': 6, 'BHARTIARTL': 7, 'BPCL': 8, 'BRITANNIA': 9, 'CIPLA': 10, 
    'COALINDIA': 11, 'DIVISLAB': 12, 'DRREDDY': 13, 'EICHERMOT': 14, 'GRASIM': 15, 
    'HCLTECH': 16, 'HDFC': 17, 'HDFCBANK': 18, 'HDFCLIFE': 19, 'HEROMOTOCO': 20, 
    'HINDALCO': 21, 'HINDUNILVR': 22, 'ICICIBANK': 23, 'INDUSINDBK': 24, 'INFY': 25, 
    'IOC': 26, 'ITC': 27, 'JSWSTEEL': 28, 'KOTAKBANK': 29, 'LT': 30, 
    'M&M': 31, 'MARUTI': 32, 'NESTLEIND': 33, 'NTPC': 34, 'ONGC': 35, 
    'POWERGRID': 36, 'RELIANCE': 37, 'SBILIFE': 38, 'SBIN': 39, 'SHREECEM': 40, 
    'SUNPHARMA': 41, 'TATACONSUM': 42, 'TATAMOTORS': 43, 'TATASTEEL': 44, 'TCS': 45, 
    'TECHM': 46, 'TITAN': 47, 'ULTRACEMCO': 48, 'UPL': 49, 'WIPRO': 50
}
_REVERSE_COMPANIES_MAPPING = {v: k for k, v in _COMPANIES_MAPPING.items()}

def load_model():
    global _MODEL
    if _MODEL is None:
        if os.path.exists(Config.MODEL_PATH):
            print(f"Loading stock prediction model from {Config.MODEL_PATH}...")
            _MODEL = joblib.load(Config.MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model file not found at {Config.MODEL_PATH}.")
            # We can create a fallback mock model for development if the file is missing
            _MODEL = None
    return _MODEL

def load_data():
    global _DF
    if _DF is None:
        if os.path.exists(Config.CSV_PATH):
            print(f"Loading historical stock CSV dataset from {Config.CSV_PATH}...")
            _DF = pd.read_csv(Config.CSV_PATH)
            _DF['Date'] = pd.to_datetime(_DF['Date'])
            # Sort chronologically
            _DF = _DF.sort_values(by=['Company', 'Date'])
            print(f"Dataset loaded successfully with {_DF.shape[0]} rows.")
        else:
            print(f"Warning: CSV file not found at {Config.CSV_PATH}.")
            _DF = None
    return _DF

def get_companies():
    """Returns sorted list of company tickers."""
    return sorted(list(_COMPANIES_MAPPING.keys()))

def get_company_id(name):
    return _COMPANIES_MAPPING.get(name, 1)

def get_company_name(comp_id):
    return _REVERSE_COMPANIES_MAPPING.get(int(comp_id), 'Unknown')

def get_latest_stock_data(company_name):
    """Retrieves the latest available stock record from the CSV for autofill."""
    df = load_data()
    if df is None:
        return None
    
    company_df = df[df['Company'] == company_name]
    if company_df.empty:
        return None
    
    # Grab the last row (sorted by date)
    latest_row = company_df.iloc[-1]
    return {
        'Date': latest_row['Date'].strftime('%Y-%m-%d'),
        'Open': float(latest_row['Open']),
        'High': float(latest_row['High']),
        'Low': float(latest_row['Low']),
        'Close': float(latest_row['Close']),
        'Adj_Close': float(latest_row['Adj Close']),
        'Volume': float(latest_row['Volume']),
        'Company': company_name,
        'Company_ID': get_company_id(company_name)
    }

def get_stock_history(company_name, limit=100):
    """Retrieves historical prices for graphing."""
    df = load_data()
    if df is None:
        return []
    
    company_df = df[df['Company'] == company_name].tail(limit)
    if company_df.empty:
        return []
    
    history = []
    for _, row in company_df.iterrows():
        history.append({
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Open': float(row['Open']),
            'High': float(row['High']),
            'Low': float(row['Low']),
            'Close': float(row['Close']),
            'Adj_Close': float(row['Adj Close']),
            'Volume': float(row['Volume'])
        })
    return history

def predict_close(company_name, open_val, high_val, low_val, adj_close, volume):
    """Executes model prediction using the loaded joblib RandomForest model."""
    model = load_model()
    comp_id = get_company_id(company_name)
    
    # X columns: Open, High, Low, Adj Close, Volume, Company (based on notebook)
    # df.drop(['Close', 'Date'], axis=1) -> ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Company']
    input_data = pd.DataFrame({
        'Open': [float(open_val)],
        'High': [float(high_val)],
        'Low': [float(low_val)],
        'Adj Close': [float(adj_close)],
        'Volume': [float(volume)],
        'Company': [int(comp_id)]
    })
    
    if model is not None:
        predicted_val = model.predict(input_data)[0]
    else:
        # Fallback simulated close calculation if model failed to load
        # Close should be roughly between Low and High, close to Adj Close
        predicted_val = (float(open_val) + float(high_val) + float(low_val) + float(adj_close)) / 4.0
        
    return float(predicted_val)
