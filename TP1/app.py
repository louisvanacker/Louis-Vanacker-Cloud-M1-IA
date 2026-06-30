"""
app.py

API FastAPI exposant le modèle de prédiction de prix de logement
(California Housing).
"""

from contextlib import asynccontextmanager
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model.joblib"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Dictionnaire global qui contiendra le modèle chargé
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle au démarrage de l'API (une seule fois)
    ml_models["housing_model"] = joblib.load(MODEL_PATH)
    yield
    # Nettoyage éventuel à l'arrêt de l'API
    ml_models.clear()


app = FastAPI(title="Housing Price API", lifespan=lifespan)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Revenu médian")
    HouseAge: float = Field(..., description="Âge moyen des logements")
    AveRooms: float = Field(..., description="Nombre moyen de pièces")
    AveBedrms: float = Field(..., description="Nombre moyen de chambres")
    Population: float = Field(..., description="Population du quartier")
    AveOccup: float = Field(..., description="Occupation moyenne")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3,
                "HouseAge": 41,
                "AveRooms": 6.9,
                "AveBedrms": 1.0,
                "Population": 322,
                "AveOccup": 2.5,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }


class PredictionOutput(BaseModel):
    predicted_house_value: float


def _predict_df(df: pd.DataFrame) -> List[float]:
    model = ml_models.get("housing_model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    preds = model.predict(df[FEATURES])
    return [float(p) for p in preds]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(house: HouseFeatures):
    df = pd.DataFrame([house.dict()])
    prediction = _predict_df(df)[0]
    return {"predicted_house_value": round(prediction, 2)}


# Bonus : prédiction en lot
@app.post("/predict_batch", response_model=List[PredictionOutput])
def predict_batch(houses: List[HouseFeatures]):
    if not houses:
        raise HTTPException(status_code=400, detail="La liste est vide")
    df = pd.DataFrame([h.dict() for h in houses])
    predictions = _predict_df(df)
    return [{"predicted_house_value": round(p, 2)} for p in predictions]
