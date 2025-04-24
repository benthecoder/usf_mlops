import requests
import json

# Sample patient data for testing
test_patient = {
    "age": 55,
    "gender": 1,
    "height": 170,
    "weight": 80.0,
    "ap_hi": 140,  # Systolic blood pressure
    "ap_lo": 90,  # Diastolic blood pressure
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}

url = "http://127.0.0.1:8000/predict"
try:
    response = requests.post(url, json=test_patient)
    response.raise_for_status()

    print("API Response:")
    print(json.dumps(response.json(), indent=2))

    result = response.json()
    prediction = result.get("prediction")
    risk_level = result.get("risk_level")

    print(
        f"\nPrediction: {'Cardiovascular disease likely' if prediction == 1 else 'No cardiovascular disease'}"
    )
    print(f"Risk Level: {risk_level}")

except requests.exceptions.RequestException as e:
    print(f"Error connecting to API: {e}")
    print(
        "Make sure the FastAPI server is running with 'uvicorn cardioApp:app --reload'"
    )
except Exception as e:
    print(f"Error: {e}")
