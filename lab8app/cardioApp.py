from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(
    title="Cardiovascular Disease Predictor",
    description="Predict cardiovascular disease risk based on patient data.",
    version="0.1",
)


class CardioRequest(BaseModel):
    age: int
    gender: int
    height: int
    weight: float
    ap_hi: int  # Systolic blood pressure
    ap_lo: int  # Diastolic blood pressure
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int


@app.on_event("startup")
def load_artifacts():
    global model
    model = joblib.load("cardio_model.pkl")
    print("Model loaded successfully")


@app.get("/")
def main():
    return {"message": "Cardiovascular Disease Prediction API"}


@app.post("/predict")
def predict(data: CardioRequest):
    # Convert request to dataframe
    input_data = pd.DataFrame([data.dict()])

    # Add derived features
    # Calculate BMI
    height_m = input_data["height"] / 100
    input_data["bmi"] = input_data["weight"] / (height_m**2)

    # Calculate pulse pressure
    input_data["pulse_pressure"] = input_data["ap_hi"] - input_data["ap_lo"]

    model_features = ["age", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure"]
    processed_data = input_data[model_features]

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    return {
        "prediction": int(prediction),  # 0 = No disease, 1 = Disease
        "probability": float(probability),
        "risk_level": "High"
        if probability > 0.7
        else "Medium"
        if probability > 0.4
        else "Low",
    }


if __name__ == "__main__":
    uvicorn.run("cardioApp:app", host="127.0.0.1", port=8000, reload=True)
