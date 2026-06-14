from flask import Flask, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE',
    'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
    'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
    'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
    'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_v1.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return {
        "status": "running",
        "endpoints": {
            "/health": "GET",
            "/predict": "POST"
        }
    }, 200


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }, 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return {"error": "JSON body is required"}, 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return {
            "error": "Missing features",
            "missing_features": missing
        }, 400

    input_df = pd.DataFrame([{f: data[f] for f in FEATURES}])

    proba = float(model.predict_proba(input_df)[0][1])
    pred = int(model.predict(input_df)[0])

    return {
        "prediction": pred,
        "probability_default": round(proba, 4),
        "model_version": "v1"
    }, 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
