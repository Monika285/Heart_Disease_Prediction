from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "heart_model.pkl"
META_PATH = BASE / "models" / "metadata.json"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.caption("Educational ML project — this prediction is not a medical diagnosis.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("Model not trained yet. Add data/heart.csv and run: python src/train.py")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

st.write("Best trained model:", metadata["best_model"])
st.subheader("Enter patient values")

# Defaults are only convenient starting values; use the dataset's units/coding.
defaults = {
    "age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}

values = {}
cols = st.columns(2)
for i, feature in enumerate(metadata["features"]):
    with cols[i % 2]:
        values[feature] = st.number_input(
            feature,
            value=float(defaults.get(feature, 0)),
            step=1.0
        )

if st.button("Predict"):
    sample = pd.DataFrame([[values[f] for f in metadata["features"]]],
                          columns=metadata["features"])
    pred = int(model.predict(sample)[0])
    prob = float(model.predict_proba(sample)[0][1])

    if pred == 1:
        st.error(f"Higher predicted likelihood ({prob:.1%})")
    else:
        st.success(f"Lower predicted likelihood ({prob:.1%})")

    st.info("This tool is for educational purposes only and should not be used to diagnose or treat a medical condition.")
