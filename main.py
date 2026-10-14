"""FastAPI prediction service for the Iris classifier."""
from pathlib import Path
from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_pipeline.joblib"
app = FastAPI(title="Iris ML Prediction API", version="1.0.0",
              description="Predict Iris species from four flower measurements.")

class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm")

class Prediction(BaseModel):
    prediction: str
    probabilities: dict[str, float]

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run `python scripts/train_model.py` first.")
    return joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Iris ML API is running", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}

@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    try:
        model = load_model()
        row = np.array([[features.sepal_length, features.sepal_width,
                         features.petal_length, features.petal_width]])
        label = str(model.predict(row)[0])
        probs = model.predict_proba(row)[0]
        return Prediction(prediction=label,
                          probabilities={str(k): round(float(v), 6)
                                         for k, v in zip(model.classes_, probs)})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
