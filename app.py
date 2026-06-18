import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# ── 1. Page Configuration ────────────────────────────────────────────────────
st.set_page_config(page_title="MotoFix Intelligent Dispatch Engine", page_icon="⚡", layout="wide")
st.title("⚡ MotoFix Intelligent Mechanic Matching Engine")

# ── 2. Absolute Model Path (works regardless of launch directory) ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "motofix_xgb_model.json")

# ── 3. Pure Python Preprocessor (bypasses .pkl completely) ──────────────────
# Represents the exact mean/variance metrics learned by your StandardScaler
def manual_scaling(df_row):
    scaled = df_row.copy()
    scaled["proximity_km"]  = (scaled["proximity_km"]  - 6.25)  / 3.32
    scaled["spec_match"]    =  scaled["spec_match"]    * 5.0          # amplified weight
    scaled["star_rating"]   = (scaled["star_rating"]   - 4.0)   / 0.57
    scaled["jobs_completed"]= (scaled["jobs_completed"]- 127.5) / 70.2
    return scaled

# ── 4. Load Native Model Safely ──────────────────────────────────────────────
@st.cache_resource
def load_native_model():
    # Use native Booster — no sklearn dependency required
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    return booster
try:
    model = load_native_model()
    st.sidebar.success(" Native JSON Model Loaded Successfully")
except FileNotFoundError:
    st.sidebar.error(
        f"❌ Model file not found.\n\n"
        f"Expected path:\n`{MODEL_PATH}`\n\n"
        "Make sure **motofix_xgb_model.json** sits in the same folder as **app.py**."
    )
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Unexpected error loading model: {e}")
    st.stop()

# ── 5. Sidebar – Breakdown Ticket ────────────────────────────────────────────
st.sidebar.header("Current Breakdowns")
zone = st.sidebar.selectbox(
    "Incident Zone",
    ["Kampala Central", "Kawempe", "Makindye", "Nakawa", "Rubaga"],
)
breakdown_type = st.sidebar.selectbox(
    "Breakdown Type",
    ["Brakes Failing", "Electrical Fault", "Engine Stalled", "Flat Tyre", "Transmission Issue"],
)

# ── 6. Mechanic Input Panel ──────────────────────────────────────────────────
st.header("🔧 Available Mechanics")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mechanic A (Kamya)")
    prox_A   = st.slider("Distance (km) – A",  0.5, 15.0, 2.5)
    spec_A   = st.checkbox("Specialization Match – A", value=True)
    rating_A = st.slider("Star Rating – A",     1.0,  5.0, 4.7)
    avail_A  = st.toggle("Online / Available – A",  value=True)
    jobs_A   = st.number_input("Jobs Done – A", min_value=0, value=142)

with col2:
    st.subheader("Mechanic B (Mukasa)")
    prox_B   = st.slider("Distance (km) – B",  0.5, 15.0, 6.0)
    spec_B   = st.checkbox("Specialization Match – B", value=True)
    rating_B = st.slider("Star Rating – B",     1.0,  5.0, 4.9)
    avail_B  = st.toggle("Online / Available – B",  value=True)
    jobs_B   = st.number_input("Jobs Done – B", min_value=0, value=85)

with col3:
    st.subheader("Mechanic C (Nsubuga)")
    prox_C   = st.slider("Distance (km) – C",  0.5, 15.0, 1.2)
    spec_C   = st.checkbox("Specialization Match – C", value=False)
    rating_C = st.slider("Star Rating – C",     1.0,  5.0, 3.8)
    avail_C  = st.toggle("Online / Available – C",  value=True)
    jobs_C   = st.number_input("Jobs Done – C", min_value=0, value=19)

# ── 7. Build Candidate DataFrame ─────────────────────────────────────────────
EXPECTED_ZONES      = ["Kampala Central", "Kawempe", "Makindye", "Nakawa", "Rubaga"]
EXPECTED_BREAKDOWNS = ["Brakes Failing", "Electrical Fault", "Engine Stalled", "Flat Tyre", "Transmission Issue"]
FINAL_COLUMN_ORDER  = (
    ["proximity_km", "spec_match", "star_rating", "available", "jobs_completed"]
    + [f"zone_{z}"           for z in EXPECTED_ZONES]
    + [f"breakdown_type_{b}" for b in EXPECTED_BREAKDOWNS]
)

candidates_df = pd.DataFrame([
    {"mechanic_id": "MECH_A (Kamya)",   "proximity_km": prox_A,  "spec_match": int(spec_A),
     "star_rating": rating_A, "available": int(avail_A), "jobs_completed": jobs_A},
    {"mechanic_id": "MECH_B (Mukasa)",  "proximity_km": prox_B,  "spec_match": int(spec_B),
     "star_rating": rating_B, "available": int(avail_B), "jobs_completed": jobs_B},
    {"mechanic_id": "MECH_C (Nsubuga)", "proximity_km": prox_C,  "spec_match": int(spec_C),
     "star_rating": rating_C, "available": int(avail_C), "jobs_completed": jobs_C},
])

# ── 8. Run Matching Engine ───────────────────────────────────────────────────
if st.button(" Run Intelligent Match Diagnostics", type="primary"):

    viable = candidates_df[candidates_df["available"] == 1].copy()

    if viable.empty:
        st.error(" CRITICAL ALERT: No Mechanics are currently online.")
    else:
        # Scale features
        scaled_features = manual_scaling(
            viable[["proximity_km", "spec_match", "star_rating", "available", "jobs_completed"]]
        )

        # One-hot encode zone and breakdown type
        for z in EXPECTED_ZONES:
            scaled_features[f"zone_{z}"] = 1.0 if zone == z else 0.0
        for b in EXPECTED_BREAKDOWNS:
            scaled_features[f"breakdown_type_{b}"] = 1.0 if breakdown_type == b else 0.0

        # Enforce exact column order required by XGBoost
        scaled_features = scaled_features[FINAL_COLUMN_ORDER]

        # Predict using native Booster (no sklearn needed)
        dmatrix = xgb.DMatrix(scaled_features)
        viable = viable.copy()
        preds = model.predict(dmatrix)
        # Booster.predict() returns raw scores for binary:logistic — already probabilities
        viable["match_probability"] = preds if preds.ndim == 1 else preds[:, 1]
        ranked_pool = viable.sort_values(by="match_probability", ascending=False)
        winner = ranked_pool.iloc[0]

        st.success(
            f"✅ **Recommended Dispatch:** {winner['mechanic_id']} "
            f"— confidence fit: **{winner['match_probability']:.2%}**"
        )

        st.subheader("Full Dispatch Rank Analysis")
        st.dataframe(
            ranked_pool[["mechanic_id", "proximity_km", "spec_match", "star_rating", "match_probability"]]
            .reset_index(drop=True),
            use_container_width=True,
        )
