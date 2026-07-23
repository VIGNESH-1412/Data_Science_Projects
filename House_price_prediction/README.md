# House_price_prediction 🏡

A machine learning project that predicts house prices from property features (bedrooms, bathrooms, living area, grade, location, etc.), wrapped in a Streamlit web app with a light "architectural blueprint" themed UI.

---

## What this project does

Given details about a house — size, rooms, grade, year built, location — the model predicts a sale price. It's trained on the classic King County (Seattle) house sales dataset (~21,600 records).

The notebook covers the full pipeline: cleaning the data, exploring it, engineering new features, training two different models, comparing them, and exporting the better one for use in a web app.

---

## Dataset

- **Rows:** 21,613 house sale records
- **Columns:** 21 (price + 20 features like bedrooms, bathrooms, sqft_living, grade, waterfront, lat/long, etc.)
- No missing values found. No duplicate rows.

## Data cleaning & EDA

- Dropped `id` and `date` columns (not useful for prediction).
- Checked for outliers using the IQR method — found **1,146 outlier rows (~5.3%)** in price, mostly very high-end homes. These were inspected (via boxplots and scatterplots vs `sqft_living`) but **kept in the data** rather than removed, since most were legitimate expensive homes, not data errors. A rule-based check flagged homes with unrealistic bedroom/bathroom counts as "potentially invalid" outliers for reference.
- Looked at correlations with price. Strongest ones: `sqft_living` (0.70), `grade` (0.67), `sqft_above` (0.61), `sqft_living15` (0.59), `bathrooms` (0.53).
- Visualized price distribution, price vs living area, price vs grade, and price vs year built.

## Feature engineering

Three new binary/categorical features were added to capture buyer-relevant "vibes" a raw sqft number doesn't:

| Feature | Logic |
|---|---|
| `FamilyFriendlyHome` | 1 if bedrooms ≥ 3, bathrooms ≥ 2, and sqft_living ≥ 1800 |
| `ModernHome` | 1 if built in 2000+ or has been renovated |
| `SpaciousHome` | 0 / 1 / 2 scale based on sqft_living (≥2500 = 2, ≥1500 = 1, else 0) |

## Models trained

Two regressors were trained and compared on an 80/20 train-test split:

| Model | Train R² | Test R² | MAE | RMSE |
|---|---|---|---|---|
| Random Forest (1500 trees) | 0.983 | 0.858 | ₹72,343 | ₹146,258 |
| **XGBoost** (300 trees, lr=0.05, depth=6) | 0.964 | **0.883** | **₹68,787** | **₹132,891** |

**XGBoost won** — it generalized better (smaller gap between train and test R²) and had lower error across the board. It was the model saved and shipped to the app.

> Note: Random Forest's train R² (0.983) vs test R² (0.858) shows some overfitting. XGBoost's smaller gap (0.964 vs 0.883) means it's the more trustworthy pick for real predictions, not just the notebook's numbers.

### Top features driving price (XGBoost importance)

1. `grade` (39.5%) — overall build/design quality, by far the biggest driver
2. `sqft_living` (16%)
3. `waterfront` (13%)
4. `lat` (7%)
5. `view`, `long`, `yr_built`, `sqft_living15`, `bathrooms` — smaller but real contributions

Bedrooms, floors, and lot size barely moved the needle — a good reminder that raw square footage and build quality matter far more than room counts.

## The app (`app.py`)

A Streamlit app with:
- A landing page built in HTML/CSS, embedded via `streamlit.components.v1`, styled with Space Grotesk + JetBrains Mono fonts in a minimalist blueprint aesthetic.
- Query-parameter-based navigation from the landing page's CTA buttons into the actual prediction form.
- The form collects all 25 features (18 original + 4 engineered + 3 derived like `house_age`, `is_renovated`, `total_rooms`, `total_sqft`) and feeds them to the trained XGBoost model in the exact order it was trained on.

**To run it:**
```bash
pip install streamlit pandas numpy joblib
streamlit run app.py
```
Make sure the model file path in `MODEL_PATH` at the top of `app.py` matches wherever you save the `.pkl` file.

## Files in this project

- `HouseWHouse_price_prediction.ipynb` — the full notebook (cleaning → EDA → feature engineering → modeling → export)
- `app.py` — the Streamlit prediction app
- `house_price_model.pkl` / `House_price_prediction.pkl` — the saved, trained XGBoost model

## Known gaps / honest notes

- The model was trained without scaling numeric features — XGBoost doesn't strictly need it, but it means the pickle isn't paired with a scaler. If you swap in a linear model later, add one.
- `FEATURE_COLUMNS` in `app.py` includes engineered columns (`house_age`, `is_renovated`, `total_rooms`, `total_sqft`) that aren't built in the notebook shown here — make sure that feature-engineering step is added before training if you want the app's feature list to match exactly.
- Outlier handling was exploratory only — outliers were flagged and visualized but not dropped from the training data, so extreme luxury homes still influence the model.
- Test R² of 0.88 is solid but not exceptional — location-heavy real estate data usually needs geospatial features (distance to city center, school zones, etc.) to push meaningfully higher.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn (RandomForestRegressor) · XGBoost · Streamlit · Pickle
