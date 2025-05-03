from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("reddit_model_pipeline.joblib")

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)


class RedditComment(BaseModel):
    text: str


class Prediction(BaseModel):
    text: str
    probability_remove: float
    prediction: int


@app.get("/")
def home():
    return {
        "message": "Reddit Comment Classifier API. Use POST /predict to make predictions."
    }


@app.post("/predict", response_model=Prediction)
def predict(comment: RedditComment):
    try:
        # Make prediction
        proba = model.predict_proba(np.array([comment.text]))
        pred_class = 1 if proba[0][1] > 0.5 else 0

        return {
            "text": comment.text,
            "probability_remove": float(proba[0][1]),
            "prediction": pred_class,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
