import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

df = pd.read_csv("data/home_electricity_consumption.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# -------------------------------------------------
# Features and target
# -------------------------------------------------

X = df.drop(
    columns=[
        "electricity_consumption_kwh",
        "timestamp"
    ]
)

y = df["electricity_consumption_kwh"]


# -------------------------------------------------
# Define feature types
# -------------------------------------------------

categorical_features = [
    "day_of_week"
]

numeric_features = [
    "temperature_c",
    "humidity_pct",
    "is_weekend",
    "is_holiday",
    "people_at_home",
    "hours_at_home",
    "ac_hours",
    "fan_hours",
    "tv_hours",
    "washing_machine_used"
]


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# -------------------------------------------------
# Random Forest model
# -------------------------------------------------

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


# -------------------------------------------------
# Complete ML pipeline
# -------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -------------------------------------------------
# Train/Test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training model...")


# -------------------------------------------------
# Train model
# -------------------------------------------------

pipeline.fit(X_train, y_train)


print("Model training completed.")


# -------------------------------------------------
# Make predictions
# -------------------------------------------------

predictions = pipeline.predict(X_test)


# -------------------------------------------------
# Evaluate model
# -------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.3f} kWh")
print(f"RMSE : {rmse:.3f} kWh")
print(f"R²   : {r2:.3f}")


# -------------------------------------------------
# Save trained model
# -------------------------------------------------

joblib.dump(
    pipeline,
    "models/electricity_model.pkl"
)

print("\nModel saved successfully:")
print("models/electricity_model.pkl")
