import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="MotoFix AI Dispatch Engine", page_icon="⚡", layout="wide")
st.title("⚡ MotoFix Intelligent Mechanic Matching Engine")
st.caption("Production Cloud Architecture — Specialization-Prioritized Pipeline")

# Find exact absolute runtime path mapping
BASE_PATH = os.path.dirname(__file__)
PREPROCESSOR_PATH = os.path.join(BASE_PATH, 'dispatch_preprocessor.pkl')
MODEL_PATH = os.path.join(BASE_PATH, 'motofix_match_model.pkl')

# 2. Load ML Assets Safely
@st.cache_resource
def load_ml_assets():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

try:
    preprocessor, model = load_ml_assets()
    st.sidebar.success("✅ ML Model & Preprocessor Mounted Successfully")
except Exception as e:
    st.sidebar.error("❌ Failed to load ML models. Ensure .pkl files are uploaded to GitHub root.")
    st.stop()

# 3. Input Configuration Setup
st.sidebar.header("📋 Current Breakdown Ticket")
zone = st.sidebar.selectbox("Incident Zone", ['Kampala Central', 'Kawempe', 'Makindye', 'Nakawa', 'Rubaga'])
breakdown_type = st.sidebar.selectbox("Breakdown Type", ['Brakes Failing', 'Electrical Fault', 'Engine Stalled', 'Flat Tyre', 'Transmission Issue'])

st.header("🛠️ Available Regional Field Agents")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mechanic A (Kamya)")
    prox_A = st.slider("Distance (km) - A", 0.5, 15.0, 2.5)
    spec_A = st.checkbox("Specialization Match - A", value=True)
    rating_A = st.slider("Star Rating - A", 1.0, 5.0, 4.7)
    avail_A = st.toggle("Online / Available - A", value=True)
    jobs_A = st.number_input("Jobs Done - A", min_value=0, value=142)

with col2:
    st.subheader("Mechanic B (Mukasa)")
    prox_B = st.slider("Distance (km) - B", 0.5, 15.0, 6.0)
    spec_B = st.checkbox("Specialization Match - B", value=True)
    rating_B = st.slider("Star Rating - B", 1.0, 5.0, 4.9)
    avail_B = st.toggle("Online / Available - B", value=True)
    jobs_B = st.number_input("Jobs Done - B", min_value=0, value=85)

with col3:
    st.subheader("Mechanic C (Nsubuga)")
    prox_C = st.slider("Distance (km) - C", 0.5, 15.0, 1.2)
    spec_C = st.checkbox("Specialization Match - C", value=False)
    rating_C = st.slider("Star Rating - C", 1.0, 5.0, 3.8)
    avail_C = st.toggle("Online / Available - C", value=True)
    jobs_C = st.number_input("Jobs Done - C", min_value=0, value=19)

# Build dynamic inputs array
candidates_df = pd.DataFrame([
    {'mechanic_id': 'MECH_A (Kamya)', 'proximity_km': prox_A, 'spec_match': int(spec_A), 'star_rating': rating_A, 'available': int(avail_A), 'jobs_completed': jobs_A, 'zone': zone, 'breakdown_type': breakdown_type},
    {'mechanic_id': 'MECH_B (Mukasa)', 'proximity_km': prox_B, 'spec_match': int(spec_B), 'star_rating': rating_B, 'available': int(avail_B), 'jobs_completed': jobs_B, 'zone': zone, 'breakdown_type': breakdown_type},
    {'mechanic_id': 'MECH_C (Nsubuga)', 'proximity_km': prox_C, 'spec_match': int(spec_C), 'star_rating': rating_C, 'available': int(avail_C), 'jobs_completed': jobs_C, 'zone': zone, 'breakdown_type': breakdown_type},
])

if st.button("🚀 Run Intelligent Match Diagnostics", type="primary"):
    viable = candidates_df[candidates_df['available'] == 1].copy()
    
    if viable.empty:
        st.error("🚨 CRITICAL ALERT: No field agents are currently online.")
    else:
        # Prepare data for the scikit-learn transformer
        features = viable[['proximity_km', 'spec_match', 'star_rating', 'available', 'jobs_completed', 'zone', 'breakdown_type']]
        transformed_features = preprocessor.transform(features)
        
        # Calculate raw probabilities from the ML model
        viable['match_probability'] = model.predict_proba(transformed_features)[:, 1]
        
        # 💡 CRITICAL PRIORITIZATION OVERRIDE:
        # To guarantee specialization rules over proximity without breaking scikit-learn's structure, 
        # we apply a professional operational weight penalty (50% deduction) to mechanics who lack specialization.
        viable['match_probability'] = np.where(
            viable['spec_match'] == 0, 
            viable['match_probability'] * 0.50, 
            viable['match_probability']
        )
        
        # Rank mechanics based on the updated prioritization metrics
        ranked_pool = viable.sort_values(by='match_probability', ascending=False)
        winner = ranked_pool.iloc[0]
        
        st.success(f"🎯 **Recommended Dispatch:** {winner['mechanic_id']} (Priority Match)")
        st.subheader("🎯 Full Dispatch Rank Analysis")
        st.dataframe(ranked_pool[['mechanic_id', 'proximity_km', 'spec_match', 'star_rating', 'match_probability']])
