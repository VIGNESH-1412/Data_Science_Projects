# Aether Stock Intelligence & Prediction Platform

Aether is a production-quality financial forecasting and stock analytics platform built to showcase a machine learning Random Forest Regressor model trained on constituents of the Indian Stock Market.

The platform provides a highly responsive, premium glassmorphic UI/UX featuring interactive simulations, historical candlestick analysis, and local history logging.

---

## Technical Stack
- **Backend:** Flask, Python
- **Machine Learning Core:** RandomForestRegressor (`scikit-learn`), `joblib`, `pandas`, `numpy`
- **Database:** SQLite (built-in history logging)
- **Frontend:** HTML5, CSS3, JavaScript, Tailwind CSS (Utility classes), Custom CSS (Glassmorphism & animations)
- **Data Visualizations:** ApexCharts (premium financial candlestick & volume tracking)

---

## Features
1. **Interactive Prediction Simulator:**
   - Select one of 50 constituents (e.g. RELIANCE, WIPRO, SBIN).
   - Autofills inputs (Open, High, Low, Adj Close, Volume) with the latest historical trade from the CSV.
   - Adjust precise sliders sized dynamically around the stock price to simulate custom scenarios.
   - Triggers Python machine learning inference in the background to output simulated closing prices.
2. **Dynamic Trend & Signals:**
   - Displays indicators showing BULLISH/BEARISH biases.
   - Generates simulated confidence percentages based on prediction variances relative to adjusted close values.
3. **SQLite Auditing Ledger:**
   - Local records of simulation parameters, outputs, signals, and timestamps are saved in `predictions.db`.
   - Easily review, delete, or clear registers directly from the dashboard.
4. **Candlestick Financial Visualizations:**
   - Explore historical candlesticks and volume trends for any stock using ApexCharts.
   - View recent statistics such as trading range, average share volume, and price trends.
5. **Detailed Model Diagnostics:**
   - Inspect train and test partition R² metrics (Train: `0.999995`, Test: `0.999971`).
   - Review Gini split weights (Feature Importances) showing which parameters influence outputs.
6. **Responsive Sleek Aesthetics:**
   - Features modern glassmorphic panels, animated glowing borders, radial flows, custom slider bars, and unified Dark & Light mode toggle.

---

## Directory Structure
```
c:/Users/Admin/Documents/Placement_Training/ML_Projects/Stock_Analysis/
├── app/
│   ├── __init__.py          # Flask app factory, registers routes and eagerly loads datasets
│   ├── config.py            # Global paths config
│   ├── database.py          # SQLite database connection & history logging actions
│   ├── model_loader.py      # Joblib loading, pandas CSV parsing, and model predict wrappers
│   ├── routes.py            # Views and RESTful JSON APIs
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Glassmorphism tokens, custom theme elements, and animations
│   │   └── js/
│   │       ├── main.js      # Theme manager, mobile controls, and notification toasts
│   │       ├── dashboard.js # Sliders synchronization, validation, simulator POSTs, and history triggers
│   │       └── analytics.js # ApexCharts constructor and stat calculations
│   └── templates/
│       ├── base.html        # Shared head elements, navbar, theme-toggles, and layout structures
│       ├── landing.html     # Hero panel and grid items
│       ├── dashboard.html   # Main prediction workspace and SQLite audit listings
│       ├── analytics.html   # Charts page with candlestick wrappers and stats
│       ├── about.html       # ML pipeline descriptions
│       ├── model_info.html  # R2 scores and feature weight progress bars
│       ├── contact.html     # Glassmorphic feedback form
│       ├── 404.html         # Custom 404 error template
│       └── 500.html         # Custom 500 system failure template
├── .venv/                   # Python virtual environment
├── Indian Stock Market Data.csv # Stock history dataset (CSV)
├── Stock_Market_Analysis.joblib # Trained Random Forest model file (joblib)
├── run.py                   # App entrypoint
├── requirements.txt         # Package dependencies list
└── README.md                # Platform documentation
```

---

## Getting Started: Local Setup

### Prerequisite Files
Verify that the following files exist in the project root:
- `Stock_Market_Analysis.joblib` (371 MB model file)
- `Indian Stock Market Data.csv` (6.06 MB dataset file)

### Installation Steps

1. **Activate Virtual Environment:**
   - **On Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **On macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute Application Server:**
   ```bash
   python run.py
   ```
   *Note: Upon startup, Flask will automatically initialize the database `predictions.db` and eagerly parse the 6MB stock CSV dataset and 371MB model file so that predictions run instantaneously.*

4. **Access UI:**
   Open your browser and navigate to `http://localhost:5000`.

---

## Production Deployment Steps

### Option A: Render (Recommended)
1. **Prepare configuration:**
   Add a `gunicorn` runner to your dependency list. Create a `wsgi.py` in the root:
   ```python
   # wsgi.py
   from app import create_app
   app = create_app()
   ```
2. **Push Code to Github:**
   Initialize git and push changes to a repository.
3. **Connect to Render:**
   - Create a new **Web Service** on Render.
   - Connect your GitHub repository.
   - Choose **Python** runtime.
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app`
   - **Environment Variables:** Set `SECRET_KEY` if desired.

### Option B: Railway
1. Install Railway CLI or connect your GitHub.
2. Railway will auto-detect the Python project using `requirements.txt`.
3. Set your Start Command in settings: `gunicorn wsgi:app` (ensure `gunicorn` is installed) or `python run.py`.
4. Deploy.
