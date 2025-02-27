from fastapi.testclient import TestClient
from app import app
import warnings

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Churn Prediction API - Use POST /predict for predictions"}

def test_predict_endpoint():
    with warnings.catch_warnings():
        # Use simple warning filter instead of specific class
        warnings.simplefilter("ignore", category=FutureWarning)
        
        test_data = {
            "State": "OH",
            "Account_length": 128,
            "Area_code": "415",
            "International_plan": "No",
            "Voice_mail_plan": "Yes",
            "Number_vmail_messages": 25,
            "Total_day_minutes": 265.1,
            "Total_day_calls": 110,
            "Total_day_charge": 45.07,
            "Total_eve_minutes": 197.4,
            "Total_eve_calls": 99,
            "Total_eve_charge": 16.78,
            "Total_night_minutes": 244.7,
            "Total_night_calls": 91,
            "Total_night_charge": 11.01,
            "Total_intl_minutes": 10.0,
            "Total_intl_calls": 3,
            "Total_intl_charge": 2.7,
            "Customer_service_calls": 1
        }

        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        response_json = response.json()
        assert "prediction" in response_json
        assert "confidence" in response_json
        assert response_json["prediction"] in ["Churn", "No Churn"]
        assert 0 <= response_json["confidence"] <= 1
