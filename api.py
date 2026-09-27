from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

import database


# -------------------------------------------------
# Create FastAPI application
# -------------------------------------------------

app = FastAPI(
    title="Household Electricity Consumption Forecasting API",
    description=(
        "Predicts household electricity consumption "
        "using environmental conditions, occupancy "
        "and appliance usage. Also stores every prediction "
        "in a database for history and stats."
    ),
    version="1.1.0"
)


# -------------------------------------------------
# Enable CORS
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

model = joblib.load(
    "models/electricity_model.pkl"
)


# -------------------------------------------------
# Initialize database (creates predictions.db + table
# on first run if they don't already exist)
# -------------------------------------------------

database.init_db()


# -------------------------------------------------
# Input data schema
# -------------------------------------------------

class PredictionInput(BaseModel):

    temperature_c: float = Field(
        ge=0,
        le=50
    )

    humidity_pct: float = Field(
        ge=0,
        le=100
    )

    day_of_week: str

    is_weekend: int = Field(
        ge=0,
        le=1
    )

    is_holiday: int = Field(
        ge=0,
        le=1
    )

    people_at_home: int = Field(
        ge=0,
        le=10
    )

    hours_at_home: float = Field(
        ge=0,
        le=24
    )

    ac_hours: float = Field(
        ge=0,
        le=24
    )

    fan_hours: float = Field(
        ge=0,
        le=24
    )

    tv_hours: float = Field(
        ge=0,
        le=24
    )

    washing_machine_used: int = Field(
        ge=0,
        le=1
    )


# -------------------------------------------------
# Home endpoint
# -------------------------------------------------

@app.get("/")
def home():

    return {
        "message":
        "Context-Aware Household Electricity Prediction API is running"
    }


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    # ---------------------------------------------
    # Create input dataframe
    # ---------------------------------------------

    input_data = pd.DataFrame([
        {
            "temperature_c": data.temperature_c,
            "humidity_pct": data.humidity_pct,
            "day_of_week": data.day_of_week,
            "is_weekend": data.is_weekend,
            "is_holiday": data.is_holiday,
            "people_at_home": data.people_at_home,
            "hours_at_home": data.hours_at_home,
            "ac_hours": data.ac_hours,
            "fan_hours": data.fan_hours,
            "tv_hours": data.tv_hours,
            "washing_machine_used":
                data.washing_machine_used
        }
    ])


    # ---------------------------------------------
    # Make prediction
    # ---------------------------------------------

    prediction = model.predict(input_data)[0]
    predicted_kwh = round(float(prediction), 3)


    # ---------------------------------------------
    # Save this prediction to the database
    # ---------------------------------------------

    database.save_prediction(data.dict(), predicted_kwh)


    # ---------------------------------------------
    # Return result
    # ---------------------------------------------

    return {
        "predicted_electricity_consumption_kwh":
            predicted_kwh
    }


# -------------------------------------------------
# History endpoint — view past predictions
# -------------------------------------------------

@app.get("/history")
def history(limit: int = 50):

    return database.get_history(limit)


# -------------------------------------------------
# Stats endpoint — summary across all predictions
# -------------------------------------------------

@app.get("/stats")
def stats():

    return database.get_stats()


# -------------------------------------------------
# Clear history endpoint — wipe stored predictions
# -------------------------------------------------

@app.delete("/history")
def delete_history():

    database.clear_history()

    return {
        "message": "All prediction history cleared"
    }