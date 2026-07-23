import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'apex-stock-prediction-secret-key-9821'
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATABASE_PATH = os.path.join(BASE_DIR, 'predictions.db')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    
    # Model and CSV Data Paths (located at workspace root)
    MODEL_PATH = os.path.join(BASE_DIR, 'Stock_Market_Analysis.joblib')
    CSV_PATH = os.path.join(BASE_DIR, 'Indian Stock Market Data.csv')
