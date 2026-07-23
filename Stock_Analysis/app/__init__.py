import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from flask import Flask
from app.config import Config
from app.database import init_db
from app.model_loader import load_model, load_data

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize Database
    print("Initializing SQLite database...")
    init_db()
    
    # Load ML Model and Stock CSV Data during startup
    print("Pre-loading machine learning model and CSV dataset...")
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model during startup: {e}")
        
    try:
        load_data()
    except Exception as e:
        print(f"Error loading CSV data during startup: {e}")

    # Register blueprints/routes
    from app.routes import bp
    app.register_blueprint(bp)
    
    return app
