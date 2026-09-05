import pandas as pd
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

mlflow.set_tracking_uri(
    "http://mlflow:5000"
)

# Charge le modèle qui porte l'alias "champion"
model = mlflow.sklearn.load_model(
    "models:/car-price-model@champion"
)

class CarInput(BaseModel):
    brand: str
    model: str
    year: int
    mileage: int
    horsepower: int


@app.get("/")
def home():
    return {"message": "Car Price Prediction API"}


@app.post("/predict")
def predict(car: CarInput):

    input_data = pd.DataFrame([
        {
            "brand": car.brand,
            "model": car.model,
            "year": car.year,
            "mileage": car.mileage,
            "horsepower": car.horsepower
        }
    ])

    prediction = model.predict(input_data)

    return {
        "predicted_price": float(prediction[0])
    }