from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common import decode_predictions, load_model_bundle, normalize_input_frame, positive_probability


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "lr": BASE_DIR / "models" / "lr_model.pkl",
    "gbt": BASE_DIR / "models" / "gbt_model.pkl",
    "xgb": BASE_DIR / "models" / "xgb_model.pkl",
}

app = FastAPI(title="Census Income Minimal API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_CACHE = {}


class PredictRequest(BaseModel):
    model: str = "xgb"
    records: list[dict[str, Any]]


def get_model(model_key: str):
    model_key = model_key.lower().strip()
    if model_key not in MODEL_PATHS:
        allowed = ", ".join(sorted(MODEL_PATHS))
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_key}'. Use one of: {allowed}.")

    if model_key not in MODEL_CACHE:
        path = MODEL_PATHS[model_key]
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {path}")
        MODEL_CACHE[model_key] = load_model_bundle(path)
    return MODEL_CACHE[model_key]


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(MODEL_PATHS)}


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="records cannot be empty.")

    bundle = get_model(request.model)
    rows = normalize_input_frame(pd.DataFrame(request.records))
    pipeline = bundle["pipeline"]

    predictions = decode_predictions(bundle, pipeline.predict(rows))
    probabilities = positive_probability(bundle, pipeline.predict_proba(rows))

    output = []
    for prediction, probability in zip(predictions, probabilities):
        output.append(
            {
                "model": request.model.lower(),
                "prediction": str(prediction),
                "probability_gt_80k": round(float(probability), 6),
            }
        )
    return {"count": len(output), "results": output}
