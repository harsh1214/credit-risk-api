from fastapi import FastAPI
from app.schemas import LoanFeatures
from app.model import SAMPLE_INPUTS, predict_proba

app = FastAPI(title="Credit Risk Prediction API")

@app.get("/samples")
def get_samples():
    return SAMPLE_INPUTS

@app.post("/predict")
def predict(data: LoanFeatures):
    prob = predict_proba(data)
    return {
        "probability_of_default": float(round(prob, 4))
    }

@app.post("/predict/sample/{sample_name}")
def predict_sample(sample_name: str):
    data = SAMPLE_INPUTS[sample_name]
    prob = predict_proba(type("X", (), data))
    return {"probability_of_default": float(round(prob, 4))}