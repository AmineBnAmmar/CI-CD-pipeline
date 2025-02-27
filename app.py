# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from model_pipeline import load_model

# Load model and preprocessing artifacts
model, encoder, scaler = load_model("churn_model.pkl")

# Create FastAPI app
app = FastAPI(title="Churn Prediction API")

# Define input data schema using Pydantic
class CustomerData(BaseModel):
    State: str
    Account_length: int
    Area_code: str
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

# Enable CORS if needed
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_churn(data: CustomerData):
    try:
        # Convert input data to DataFrame using Pydantic V2 syntax
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
       

        # Rename columns to match training data format (spaces instead of underscores)
        column_mapping = {
            'International_plan': 'International plan',
            'Voice_mail_plan': 'Voice mail plan',
            'Account_length': 'Account length',
            'Area_code': 'Area code',
            'Customer_service_calls': 'Customer service calls',
            'Number_vmail_messages': 'Number vmail messages',
            'Total_day_calls': 'Total day calls',
            'Total_day_charge': 'Total day charge',
            'Total_day_minutes': 'Total day minutes',
            'Total_eve_calls': 'Total eve calls',
            'Total_eve_charge': 'Total eve charge',
            'Total_eve_minutes': 'Total eve minutes',
            'Total_intl_calls': 'Total intl calls',
            'Total_intl_charge': 'Total intl charge',
            'Total_intl_minutes': 'Total intl minutes',
            'Total_night_calls': 'Total night calls',
            'Total_night_charge': 'Total night charge',
            'Total_night_minutes': 'Total night minutes'
        }
        input_df = input_df.rename(columns=column_mapping)
        
        # Update categorical columns to use spaced names
        categorical_cols = ["State", "International plan", "Voice mail plan"]
        
        # Validate all categorical columns exist
        missing_cols = [col for col in categorical_cols if col not in input_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing categorical columns in input: {missing_cols}"
            )

        # Encode categorical features
        encoded_data = encoder.transform(input_df[categorical_cols])
        numerical_data = input_df.drop(columns=categorical_cols)
        processed_df = pd.concat([numerical_data, encoded_data], axis=1)

        # Scale numerical features
        scaled_data = scaler.transform(processed_df)

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "confidence": float(model.predict_proba(scaled_data)[0][1])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Churn Prediction API - Use POST /predict for predictions"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
