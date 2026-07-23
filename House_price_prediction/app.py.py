"""
House Price Predictor — Interactive Streamlit App
---------------------------------------------------
Light, minimalist, "architectural blueprint" themed UI (Space Grotesk + JetBrains Mono)
built with Tailwind-flavoured custom CSS/HTML injected via streamlit.components.v1.

HOW TO USE
1. Put your trained model file next to this script and update MODEL_PATH below.
   (Optionally a fitted scaler at SCALER_PATH, if you scaled numeric features.)
2. Update FEATURE_COLUMNS to EXACTLY match the column order your model was trained on.
3. streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ============================================================
# CONFIG — edit these to match your trained model
# ============================================================
MODEL_PATH = "xgb_house_model.pkl"
SCALER_PATH = "scaler.pkl"  # optional, app skips silently if not found

# Final column order fed to model.predict(). Edit to match your training notebook exactly.
FEATURE_COLUMNS = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
    "zipcode", "lat", "long", "sqft_living15", "sqft_lot15",
    "house_age", "is_renovated", "total_rooms", "total_sqft",
    "FamilyFriendlyHome", "ModernHome", "SpaciousHome",
]

st.set_page_config(page_title="HouseWorth AI", page_icon="🏡", layout="wide")

# ============================================================
# STYLE — light, minimalist "blueprint" theme
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #F6F7F4;
  --card:      #FFFFFF;
  --ink:       #1E2A24;
  --muted:     #667066;
  --pine:      #2D5B4E;
  --pine-dark: #1F4438;
  --amber:     #C9963C;
  --line:      #E4E7E1;
  --key-bg:    #FBF3E3;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }
h1, h2, h3, .block-title { font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
.mono-num { font-family: 'JetBrains Mono', monospace !important; }

/* header */
.hero-wrap {
  padding: 8px 4px 18px 4px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 22px;
}
.eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--pine); font-weight: 600;
}
.hero-title {
  font-size: 34px; font-weight: 700; margin: 4px 0 2px 0; color: var(--ink);
}
.hero-sub { color: var(--muted); font-size: 14.5px; }

/* section card */
.section-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 20px 22px 8px 22px;
  margin-bottom: 18px;
  box-shadow: 0 1px 2px rgba(30,42,36,0.04);
}
.section-head {
  display:flex; align-items:center; gap:10px; margin-bottom: 6px;
}
.section-num {
  font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--pine);
  background: #EAF1EE; border-radius: 6px; padding: 3px 8px; font-weight:600;
}
.section-title { font-weight: 600; font-size: 16.5px; }
.section-desc { color: var(--muted); font-size: 13px; margin-bottom: 14px; }

/* key-factor badge */
.key-badge {
  display:inline-flex; align-items:center; gap:5px;
  background: var(--key-bg); color: var(--amber); border: 1px solid #EFDCAE;
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight:600;
  letter-spacing:0.04em; text-transform:uppercase;
  padding: 2px 8px; border-radius: 999px; margin-bottom: 6px;
}
.key-badge::before { content: "●"; font-size: 8px; }

label, .stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {
  font-weight: 500 !important; font-size: 13.5px !important; color: var(--ink) !important;
}

/* buttons */
.stButton>button {
  background: var(--pine); color: white; border: none; border-radius: 10px;
  font-weight: 600; padding: 12px 20px; font-size: 15px;
  transition: all .15s ease; box-shadow: 0 2px 6px rgba(45,91,78,0.25);
}
.stButton>button:hover { background: var(--pine-dark); transform: translateY(-1px); box-shadow: 0 4px 10px rgba(45,91,78,0.3); }

/* toggle-like checkbox rows */
.toggle-row {
  display:flex; justify-content:space-between; align-items:center;
  border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
  background: #FBFCFA;
}
hr { border-color: var(--line); }
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-wrap">
  <div class="eyebrow">HOUSE PRICE PREDICTOR · XGBoost</div>
  <div class="hero-title">🏡 Enna vilai varum indha veedu?</div>
  <div class="hero-sub">Fill in the property details below — the important, price-driving factors are marked <b>Key Factor</b>. Prediction updates instantly on click.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def key_badge():
    st.markdown('<div class="key-badge">Key Factor</div>', unsafe_allow_html=True)

def section_head(num, title, desc):
    st.markdown(f"""
    <div class="section-head">
      <div class="section-num">{num}</div>
      <div class="section-title">{title}</div>
    </div>
    <div class="section-desc">{desc}</div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

model = load_model()
scaler = load_scaler()

# ============================================================
# INPUT FORM
# ============================================================
col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    # ---- Section 1: Basic details ----
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_head("01", "Basic Details", "Room counts and floor layout.")
        b1, b2, b3 = st.columns(3)
        with b1:
            bedrooms = st.number_input("Bedrooms", 0, 15, 3, step=1)
        with b2:
            bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, step=0.25)
        with b3:
            floors = st.number_input("Floors", 1.0, 4.0, 1.0, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Section 2: Size ----
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_head("02", "Size & Layout", "Living space is the single strongest price driver.")
        key_badge()
        sqft_living = st.slider("Living area (sqft)", 300, 12000, 1800, step=50)
        s1, s2 = st.columns(2)
        with s1:
            sqft_above = st.number_input("Sqft above ground", 0, 10000, int(sqft_living * 0.8), step=50)
        with s2:
            sqft_basement = st.number_input("Sqft basement", 0, 6000, max(0, sqft_living - int(sqft_living*0.8)), step=50)
        s3, s4 = st.columns(2)
        with s3:
            sqft_lot = st.number_input("Lot size (sqft)", 300, 200000, 5000, step=100)
        with s4:
            sqft_living15 = st.number_input("Neighbours' avg living sqft", 300, 12000, 1800, step=50)
        sqft_lot15 = st.number_input("Neighbours' avg lot sqft", 300, 200000, 5000, step=100)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Section 3: Quality & Condition ----
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_head("03", "Quality & Condition", "Construction grade moves valuation more than most people expect.")
        q1, q2 = st.columns(2)
        with q1:
            key_badge()
            grade = st.slider("Construction grade (1 = low, 13 = luxury)", 1, 13, 7)
        with q2:
            condition = st.slider("Condition (1 = poor, 5 = excellent)", 1, 5, 3)
        y1, y2 = st.columns(2)
        with y1:
            yr_built = st.number_input("Year built", 1900, datetime.now().year, 2005, step=1)
        with y2:
            yr_renovated = st.number_input("Year renovated (0 = never)", 0, datetime.now().year, 0, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Section 4: Location & special features ----
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_head("04", "Location & Views", "Waterfront and view rating are rare but high-impact factors.")
        l1, l2, l3 = st.columns(3)
        with l1:
            zipcode = st.number_input("Zipcode", 90000, 99999, 98103, step=1)
        with l2:
            lat = st.number_input("Latitude", 47.0, 48.0, 47.55, step=0.001, format="%.4f")
        with l3:
            long = st.number_input("Longitude", -123.0, -121.0, -122.25, step=0.001, format="%.4f")
        f1, f2 = st.columns(2)
        with f1:
            key_badge()
            waterfront = st.selectbox("Waterfront property?", ["No", "Yes"])
        with f2:
            key_badge()
            view = st.slider("View rating (0–4)", 0, 4, 0)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Section 5: Home style flags ----
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_head("05", "Home Style", "Auto-suggested from your inputs above — toggle to override.")
        auto_family = bedrooms >= 3 and condition >= 3
        auto_modern = yr_built >= 2005 or yr_renovated > 0
        auto_spacious = sqft_living >= 2200

        t1, t2, t3 = st.columns(3)
        with t1:
            family_friendly = st.checkbox("👨‍👩‍👧 Family Friendly Home", value=auto_family)
        with t2:
            modern_home = st.checkbox("✨ Modern Home", value=auto_modern)
        with t3:
            spacious_home = st.checkbox("📐 Spacious Home", value=auto_spacious)
        st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔮 Predict House Price", use_container_width=True)

# ============================================================
# SIDE PANEL — live snapshot while filling the form
# ============================================================
with col_side:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="toggle-row"><span>Total rooms</span><span class="mono-num">{bedrooms + bathrooms:.1f}</span></div>
    <div class="toggle-row"><span>Total sqft</span><span class="mono-num">{sqft_living + sqft_basement}</span></div>
    <div class="toggle-row"><span>House age</span><span class="mono-num">{datetime.now().year - yr_built} yrs</span></div>
    <div class="toggle-row"><span>Renovated</span><span class="mono-num">{"Yes" if yr_renovated > 0 else "No"}</span></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc" style="margin-bottom:0;">
    • <b>Grade</b> and <b>sqft_living</b> usually dominate the model's decision.<br><br>
    • <b>Waterfront</b> + high <b>view</b> score can push price up sharply even for a modest-sized home.<br><br>
    • Renovation resets a lot of the "age penalty" the model applies.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FEATURE ENGINEERING — must mirror training-time logic
# ============================================================
def build_feature_row():
    house_age = datetime.now().year - yr_built
    is_renovated = 1 if yr_renovated > 0 else 0
    total_rooms = bedrooms + bathrooms
    total_sqft = sqft_living + sqft_basement

    row = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": sqft_lot, "floors": floors,
        "waterfront": 1 if waterfront == "Yes" else 0, "view": view,
        "condition": condition, "grade": grade,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "yr_built": yr_built, "yr_renovated": yr_renovated,
        "zipcode": zipcode, "lat": lat, "long": long,
        "sqft_living15": sqft_living15, "sqft_lot15": sqft_lot15,
        "house_age": house_age, "is_renovated": is_renovated,
        "total_rooms": total_rooms, "total_sqft": total_sqft,
        "FamilyFriendlyHome": int(family_friendly),
        "ModernHome": int(modern_home),
        "SpaciousHome": int(spacious_home),
    }
    df = pd.DataFrame([row])
    df = df.reindex(columns=FEATURE_COLUMNS)  # enforce exact training order
    return df

# ============================================================
# PREDICTION + RESULT CARD (Tailwind, animated count-up)
# ============================================================
def render_result(price_value, low, high):
    price_fmt = f"{price_value:,.0f}"
    low_fmt = f"{low:,.0f}"
    high_fmt = f"{high:,.0f}"
    html = f"""
    <script src="https://cdn.tailwindcss.com"></script>
    <div class="w-full font-sans">
      <div class="rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50 via-white to-amber-50 p-8 shadow-sm">
        <div class="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-emerald-700 font-semibold mb-2">
          <span class="w-2 h-2 rounded-full bg-emerald-600"></span> Estimated Market Value
        </div>
        <div id="price" class="text-5xl font-bold text-slate-900 tracking-tight" style="font-family:'Space Grotesk',sans-serif;">$0</div>
        <div class="text-sm text-slate-500 mt-2">Likely range: <span class="font-mono font-medium text-slate-700">${low_fmt}</span> — <span class="font-mono font-medium text-slate-700">${high_fmt}</span></div>
        <div class="w-full h-2 rounded-full bg-slate-100 mt-5 overflow-hidden">
          <div class="h-full bg-gradient-to-r from-emerald-500 to-amber-400 rounded-full" style="width:100%"></div>
        </div>
      </div>
    </div>
    <script>
      const target = {price_value:.0f};
      const el = document.getElementById('price');
      let current = 0;
      const step = Math.max(1, Math.floor(target / 60));
      const timer = setInterval(() => {{
        current += step;
        if (current >= target) {{ current = target; clearInterval(timer); }}
        el.innerText = '$' + current.toLocaleString();
      }}, 16);
    </script>
    """
    components.html(html, height=210)

st.markdown("<br>", unsafe_allow_html=True)

if predict_clicked:
    X = build_feature_row()

    if model is None:
        st.warning(f"⚠️ Model file `{MODEL_PATH}` not found next to app.py. Place your trained "
                   f"XGBoost model there (or update MODEL_PATH) to get real predictions. "
                   f"Showing the feature row that would be sent to the model instead:")
        st.dataframe(X, use_container_width=True)
    else:
        X_input = X.copy()
        if scaler is not None:
            X_input = pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)
        pred = float(model.predict(X_input)[0])
        low, high = pred * 0.92, pred * 1.08
        render_result(pred, low, high)

        with st.expander("🔍 See the exact feature values sent to the model"):
            st.dataframe(X, use_container_width=True)
else:
    st.info("👆 Fill in the details above and hit **Predict House Price** to see the estimate.")
