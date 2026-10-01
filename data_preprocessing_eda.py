
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------

df = pd.read_csv("data/home_electricity_consumption.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)

# -------------------------------------------------
# 2. Missing values check
# -------------------------------------------------

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())
# Decision: dataset has no missing values, so no imputation needed.

# -------------------------------------------------
# 3. Duplicate rows check
# -------------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())

# -------------------------------------------------
# 4. Convert timestamp + extract time features
# -------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
# Decision: hour/month extracted for EDA plots below. Not fed into the
# model currently since day_of_week + is_weekend + is_holiday already
# capture calendar effects; hour can be added later if hour-level
# granularity is needed in the model.

# -------------------------------------------------
# 5. Statistical summary
# -------------------------------------------------

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

# -------------------------------------------------
# 6. Outlier check (IQR method) on target
# -------------------------------------------------

q1 = df["electricity_consumption_kwh"].quantile(0.25)
q3 = df["electricity_consumption_kwh"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df["electricity_consumption_kwh"] < lower) | (df["electricity_consumption_kwh"] > upper)]

print("\n" + "=" * 50)
print("OUTLIER CHECK (IQR method) on target")
print("=" * 50)
print(f"Lower bound: {lower:.3f}, Upper bound: {upper:.3f}")
print(f"Number of outlier rows: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
# Decision: high-consumption hours are genuine (AC-heavy hours), not
# data errors, so outliers are kept rather than removed.

# -------------------------------------------------
# 7. Correlation with target
# -------------------------------------------------

numeric_cols = ["temperature_c", "humidity_pct", "is_weekend", "is_holiday",
                 "people_at_home", "hours_at_home", "ac_hours", "fan_hours",
                 "tv_hours", "washing_machine_used", "electricity_consumption_kwh"]

corr = df[numeric_cols].corr()
print("\n" + "=" * 50)
print("CORRELATION WITH TARGET")
print("=" * 50)
print(corr["electricity_consumption_kwh"].sort_values(ascending=False))

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

# -------------------------------------------------
# 8. Target distribution
# -------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df["electricity_consumption_kwh"], bins=40, kde=True)
plt.title("Distribution of Electricity Consumption (kWh)")
plt.xlabel("kWh")
plt.tight_layout()
plt.savefig("outputs/target_distribution.png", dpi=150)
plt.close()

# -------------------------------------------------
# 9. Consumption by hour of day
# -------------------------------------------------

plt.figure(figsize=(9, 5))
hourly_avg = df.groupby("hour")["electricity_consumption_kwh"].mean()
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker="o")
plt.title("Average Consumption by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Avg kWh")
plt.tight_layout()
plt.savefig("outputs/consumption_by_hour.png", dpi=150)
plt.close()

# -------------------------------------------------
# 10. Consumption: weekday vs weekend, holiday vs non-holiday
# -------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(x="is_weekend", y="electricity_consumption_kwh", data=df, ax=axes[0])
axes[0].set_title("Weekday (0) vs Weekend (1)")
sns.boxplot(x="is_holiday", y="electricity_consumption_kwh", data=df, ax=axes[1])
axes[1].set_title("Non-Holiday (0) vs Holiday (1)")
plt.tight_layout()
plt.savefig("outputs/weekend_holiday_boxplots.png", dpi=150)
plt.close()

# -------------------------------------------------
# 11. Consumption vs AC hours (strongest driver) and temperature
# -------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.scatterplot(x="ac_hours", y="electricity_consumption_kwh", data=df, alpha=0.3, ax=axes[0])
axes[0].set_title("Consumption vs AC Hours")
sns.scatterplot(x="temperature_c", y="electricity_consumption_kwh", data=df, alpha=0.3, ax=axes[1])
axes[1].set_title("Consumption vs Temperature")
plt.tight_layout()
plt.savefig("outputs/consumption_vs_ac_temp.png", dpi=150)
plt.close()

print("\nAll EDA plots saved to outputs/")
