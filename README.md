# MotoFix — Intelligent Mechanic Matching Engine

MotoFix is a small Streamlit app that picks the best mechanic to dispatch for a roadside breakdown. Instead of just sending whoever is closest, it scores each available mechanic with a trained XGBoost model that weighs distance, specialization match, rating, and job history together, then ranks them.

It was built around a real problem for roadside assistance dispatch in Kampala: proximity alone is a bad signal, because the nearest mechanic isn't always the right one if they don't specialize in the fault or have a poor track record.

## How it works

You pick the breakdown's zone and type in the sidebar, then fill in up to three candidate mechanics (distance, specialization match, star rating, whether they're online, jobs completed). Hit the button and the app:

1. Filters out anyone not marked available
2. Scales the numeric features to match what the model was trained on
3. One-hot encodes the zone and breakdown type
4. Runs everything through the XGBoost booster
5. Ranks the mechanics by match probability and shows the winner

The three mechanics in the demo (Kamya, Mukasa, Nsubuga) are just placeholders for testing — in a real deployment these would come from a live pool of available mechanics.

## Project files

- `app.py` — the Streamlit app itself
- `motofix_xgb_model.json` — the trained XGBoost model, loaded directly via the native Booster API (no scikit-learn dependency needed at inference time)
- `motofix_match_model.pkl` / `dispatch_preprocessor.pkl` — earlier pickled versions of the model and preprocessing pipeline, kept for reference/retraining
- `motofix_unbiased_matching_data.csv` — the training data: 5,000 requests across 5 zones and 5 breakdown types, with mechanic-level features and a `best_mechanic_selected` label
- `requirements.txt` — Python dependencies

## Running it

You'll need Python 3.9+ installed.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the app in your browser (usually at `http://localhost:8501`). Make sure `motofix_xgb_model.json` stays in the same folder as `app.py` — the app resolves it relative to its own location, so it doesn't matter what directory you launch it from.

## Notes on the model

The scaling used at inference time (`manual_scaling` in `app.py`) hard-codes the mean/variance the original `StandardScaler` learned during training, rather than loading the `.pkl` preprocessor. This was done so the app has no scikit-learn dependency at runtime — only `xgboost` is needed to run predictions. If the model gets retrained on new data, these constants need to be updated to match.

The dataset name says "unbiased," but that just means known confounds (like the model always winning on distance) were deliberately balanced out during data generation — it's synthetic data, not something pulled from live dispatch logs.

## Possible next steps

- Replace the manual sidebar entry with a real mechanic pool (pulled from a database or API)
- Add authentication so this isn't a public-facing dispatch tool
- Retrain periodically as more real dispatch outcomes come in, and swap the synthetic dataset for actual historical data
