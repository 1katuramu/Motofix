MotoFix — Intelligent Mechanic Matching Engine

MotoFix is a Streamlit demo application that recommends which mechanic should be dispatched to a vehicle breakdown. Given an incident's zone and breakdown type, plus live details on up to three candidate mechanics (distance, specialization match, rating, availability, and jobs completed), it scores each candidate with a trained XGBoost model and recommends the best match.

How It Works


Pick the incident context in the sidebar — the zone (e.g. Kawempe, Rubaga, Kampala Central) and the breakdown type (e.g. Electrical Fault, Engine Stalled, Flat Tyre).
Enter candidate mechanic details in the three columns (distance in km, whether their specialization matches the breakdown, star rating, online/available toggle, and jobs completed).
Run the matching engine. The app filters out unavailable mechanics, scales/encodes the remaining candidates' features, and scores them with a pretrained XGBoost model loaded from motofix_xgb_model.json.
View the result — the top-ranked mechanic is highlighted as the recommended dispatch, with a full ranked table showing every viable candidate's match probability.


Project Structure

Motofix-main/
├── app.py                                # Streamlit application (UI + inference)
├── requirements.txt                      # Python dependencies
├── motofix_xgb_model.json                # Trained XGBoost booster (native format, loaded at runtime)
├── motofix_match_model.pkl                # Serialized model artifact (alternate/reference copy)
├── dispatch_preprocessor.pkl              # Serialized preprocessing artifact (alternate/reference copy)
└── motofix_unbiased_matching_data.csv     # Training/reference dataset (5,000 mechanic-candidate rows)

Model & Features

The app loads motofix_xgb_model.json directly with XGBoost's native Booster API (no scikit-learn dependency required at inference time). Before scoring, each candidate mechanic's raw inputs are transformed to match what the model was trained on:


Numeric features (proximity_km, star_rating, jobs_completed) are standardized using fixed mean/scale constants baked into app.py.
spec_match (whether the mechanic's specialization matches the breakdown) is boosted with an amplified weight.
Zone and breakdown type are one-hot encoded to match the exact column order the model expects.


The training data (motofix_unbiased_matching_data.csv) contains historical dispatch records across five Kampala-area zones and five breakdown types, with a best_mechanic_selected label used to train the matching model.


Note: the .pkl files (motofix_match_model.pkl, dispatch_preprocessor.pkl) are not loaded by app.py at runtime — the app re-implements scaling manually and loads the model from the native JSON format instead. They're included as reference/original training artifacts.



Requirements


Python 3.9+
Dependencies listed in requirements.txt:

streamlit
pandas
numpy
xgboost
joblib





Setup & Running

bash# Navigate to the project folder
cd Motofix-main

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

Streamlit will open the app in your browser (default: http://localhost:8501). Make sure motofix_xgb_model.json stays in the same folder as app.py — the app resolves the model path relative to its own location, so it works regardless of the directory you launch it from.

Notes & Limitations


The demo panel is limited to three mechanics (A, B, C) entered manually; it's a proof-of-concept UI rather than a live dispatch integration.
The scaling constants in manual_scaling() are fixed values derived from the original training data rather than being recomputed dynamically, so they should be retrained/updated if the underlying data distribution changes.
Zones and breakdown types are limited to the fixed lists defined in app.py (EXPECTED_ZONES, EXPECTED_BREAKDOWNS); new categories would require retraining the model and updating these lists together.
