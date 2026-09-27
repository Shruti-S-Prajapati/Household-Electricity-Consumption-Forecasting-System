# Household Electricity Consumption Forecasting System

A context-aware machine learning system that predicts household electricity
consumption (in kWh) based on weather conditions, occupancy, and appliance
usage patterns.

## Overview

Traditional electricity forecasting models rely mostly on weather data
(temperature, humidity) and calendar features (weekday/weekend). This
project extends that by incorporating **occupancy** and **appliance usage**
as first-class features, so the model can answer scenario-based questions
such as:

- What will consumption look like if no one is home today?
- How does an extra hour of AC usage change predicted consumption?
- Does a holiday (fewer people home / no college) reduce usage?

## Features Used

| Feature | Description |
|---|---|
| `temperature_c` | Ambient temperature (°C) |
| `humidity_pct` | Relative humidity (%) |
| `day_of_week` | Day name (categorical) |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_holiday` | 1 if holiday |
| `people_at_home` | Number of people present |
| `hours_at_home` | Total person-hours at home |
| `ac_hours` | Hours of AC usage |
| `fan_hours` | Hours of fan usage |
| `tv_hours` | Hours of TV usage |
| `washing_machine_used` | 1 if washing machine used |

**Target:** `electricity_consumption_kwh`

## Project Structure

```
Household-Electricity-Consumption-Forecasting-System/
├── data/
│   └── home_electricity_consumption.csv   # Hourly dataset (1 year)
├── outputs/                               # EDA plots (generated)
├── models/
│   └── electricity_model.pkl              # Trained model (generated)
├── data_preprocessing_eda.py              # Data checks + EDA plots
├── train_model.py                         # Trains and saves the model
├── api.py                                 # FastAPI prediction service
├── index.html                             # Simple web frontend
├── requirements.txt
└── README.md
```

## Model

A **Random Forest Regressor** (150 trees, max depth 12) inside a scikit-learn
`Pipeline`, with `day_of_week` one-hot encoded and all other features passed
through directly.

**Performance (on held-out 20% test split):**

| Metric | Value |
|---|---|
| MAE | 0.191 kWh |
| RMSE | 0.291 kWh |
| R² | 0.778 |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Run EDA / preprocessing checks (optional, for analysis/report):**
```bash
python data_preprocessing_eda.py
```

**2. Train the model:**
```bash
python train_model.py
```

**3. Start the API:**
```bash
uvicorn api:app --reload
```
API runs at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

**4. Open the frontend:**
Open `index.html` in a browser (API must be running).

## API Example

`POST /predict`
```json
{
  "temperature_c": 28.5,
  "humidity_pct": 65,
  "day_of_week": "Monday",
  "is_weekend": 0,
  "is_holiday": 0,
  "people_at_home": 2,
  "hours_at_home": 10,
  "ac_hours": 4,
  "fan_hours": 3,
  "tv_hours": 2,
  "washing_machine_used": 1
}
```

Response:
```json
{ "predicted_electricity_consumption_kwh": 1.23 }
```

## Notes

- Dataset is synthetic, generated to reflect realistic household patterns,
  with added noise so no single feature deterministically explains
  consumption (avoids an unrealistic near-perfect R²).
- Outliers (high-AC-usage hours) were kept rather than removed, since they
  represent genuine high-consumption periods rather than data errors.