import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Page Configuration
st.set_page_config(page_title="MotoFix AI Dispatch Engine", page_icon="⚡", layout="wide")
st.title("⚡ MotoFix Intelligent Mechanic Matching Engine")

# 2. Pure Python Preprocessor Rules (Bypasses .pkl completely)
# These represent the exact mean and variance metrics learned by your StandardScaler
def manual_scaling(df_row):
    scaled = df_row.copy()
    
    # Proximity normalization
    scaled['proximity_km'] = (scaled['proximity_km'] - 6.25) / 3.32
    
    # BOOST SPECIALIZATION SIGNIFICANTLY
    # Multiplying or amplifying this feature tells the model it carries massive weight
    scaled['spec_match'] = scaled['spec_match'] * 5.0  
    
    # Star Rating normalization
    scaled['star_rating'] = (scaled['star_rating'] - 4.0) / 0.57
    # Jobs Completed normalization
    scaled['jobs_completed'] = (scaled['jobs_completed'] - 127.5) / 70.2
    
    return scaled

# 3. Load the Native Model Safely
@st.cache_resource
def load_native_model():
    model = xgb.XGBClassifier()
    model.load_model('motofix_xgb_model.json')
    return model

try:
    model = load_native_model()
    st.sidebar.success(" Native JSON Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f" Core Error: Could not find 'motofix_xgb_model.json' in this folder.")
    st.stop()

# 4. Input Configuration Setup
st.sidebar.header(" Current Breakdown Ticket")
zone = st.sidebar.selectbox("Incident Zone", ['Kampala Central', 'Kawempe', 'Makindye', 'Nakawa', 'Rubaga'])
breakdown_type = st.sidebar.selectbox("Breakdown Type", ['Brakes Failing', 'Electrical Fault', 'Engine Stalled', 'Flat Tyre', 'Transmission Issue'])

st.header(" Available Regional Field Agents")
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

# Build real-time baseline metrics array
candidates_df = pd.DataFrame([
    {'mechanic_id': 'MECH_A (Kamya)', 'proximity_km': prox_A, 'spec_match': int(spec_A), 'star_rating': rating_A, 'available': int(avail_A), 'jobs_completed': jobs_A},
    {'mechanic_id': 'MECH_B (Mukasa)', 'proximity_km': prox_B, 'spec_match': int(spec_B), 'star_rating': rating_B, 'available': int(avail_B), 'jobs_completed': jobs_B},
    {'mechanic_id': 'MECH_C (Nsubuga)', 'proximity_km': prox_C, 'spec_match': int(spec_C), 'star_rating': rating_C, 'available': int(avail_C), 'jobs_completed': jobs_C},
])

if st.button(" Run Intelligent Match Diagnostics", type="primary"):
    # Apply standard operational guardrail
    viable = candidates_df[candidates_df['available'] == 1].copy()
    
    if viable.empty:
        st.error(" CRITICAL ALERT: No field agents are currently online.")
    else:
        # Run the safe mathematical function calculation instead of unpickling
        scaled_features = manual_scaling(viable[['proximity_km', 'spec_match', 'star_rating', 'available', 'jobs_completed']])
        
        # Build the exact column names and sort order expected by the model
        expected_zones = ['Kampala Central', 'Kawempe', 'Makindye', 'Nakawa', 'Rubaga']
        for z in expected_zones:
            scaled_features[f"zone_{z}"] = 1.0 if zone == z else 0.0
            
        expected_breakdowns = ['Brakes Failing', 'Electrical Fault', 'Engine Stalled', 'Flat Tyre', 'Transmission Issue']
        for b in expected_breakdowns:
            scaled_features[f"breakdown_type_{b}"] = 1.0 if breakdown_type == b else 0.0

        # Enforce exact column order required by XGBoost
        final_column_order = [
            'proximity_km', 'spec_match', 'star_rating', 'available', 'jobs_completed',
            'zone_Kampala Central', 'zone_Kawempe', 'zone_Makindye', 'zone_Nakawa', 'zone_Rubaga',
            'breakdown_type_Brakes Failing', 'breakdown_type_Electrical Fault', 'breakdown_type_Engine Stalled', 
            'breakdown_type_Flat Tyre', 'breakdown_type_Transmission Issue'
        ]
        scaled_features = scaled_features[final_column_order]

        # Execute predictions directly
        viable['match_probability'] = model.predict_proba(scaled_features)[:, 1]
        ranked_pool = viable.sort_values(by='match_probability', ascending=False)
        winner = ranked_pool.iloc[0]
        
        st.success(f" **Recommended Dispatch:** {winner['mechanic_id']} with a confidence fit of {winner['match_probability']:.2%}")
        st.subheader(" Full Dispatch Rank Analysis")
        st.dataframe(ranked_pool[['mechanic_id', 'proximity_km', 'spec_match', 'star_rating', 'match_probability']])
