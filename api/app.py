from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

NAME = "Vaishnav Nigade"
ROLL_NO = "2022BCD0045"

app = FastAPI(title="Wine Quality Prediction API")

MODEL_PATH = "models/model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "name": NAME,
        "roll_no": ROLL_NO
    }


@app.post("/predict")
def predict(data: WineInput):
    if model is None:
        return {
            "error": "Model not found. Train the model first.",
            "name": NAME,
            "roll_no": ROLL_NO
        }

    input_df = pd.DataFrame([{
        "fixed acidity": data.fixed_acidity,
        "volatile acidity": data.volatile_acidity,
        "citric acid": data.citric_acid,
        "residual sugar": data.residual_sugar,
        "chlorides": data.chlorides,
        "free sulfur dioxide": data.free_sulfur_dioxide,
        "total sulfur dioxide": data.total_sulfur_dioxide,
        "density": data.density,
        "pH": data.pH,
        "sulphates": data.sulphates,
        "alcohol": data.alcohol
    }])

    prediction = int(model.predict(input_df)[0])

    return {
        "prediction": prediction,
        "name": NAME,
        "roll_no": ROLL_NO
    }