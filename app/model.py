import json
from pathlib import Path
import numpy as np
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

with open(MODEL_DIR / "feature_order.json") as f:
    FEATURE_ORDER = json.load(f)

model = xgb.XGBClassifier()
model.load_model(MODEL_DIR / "xgb_model.json")

with open(MODEL_DIR / "sample_inputs.json") as f:
    SAMPLE_INPUTS = json.load(f)

def predict_proba(features):
    x = np.array(
        [getattr(features, f) for f in FEATURE_ORDER],
        dtype=float
    ).reshape(1, -1)

    return model.predict_proba(x)[0, 1]
