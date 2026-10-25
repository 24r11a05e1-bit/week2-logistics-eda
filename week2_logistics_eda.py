"""
Week 2 Logistics Exploratory Data Analysis (EDA)
Author: Ramidi Bhaavani
College: Geetanjali College of Engineering and Technology

This project generates a simulated logistics dataset, cleans it, produces
summary tables, and saves five visualizations. All data is synthetic.
Install requirements with: pip install pandas numpy matplotlib
Run with: python week2_logistics_eda.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("week2_output")
OUT.mkdir(exist_ok=True)
(OUT / "visualizations").mkdir(exist_ok=True)
(OUT / "data").mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n = 300

routes = rng.choice(
    ["Route A", "Route B", "Route C", "Route D"],
    size=n, p=[0.30, 0.25, 0.25, 0.20]
)
distance_map = {"Route A": 180, "Route B": 420, "Route C": 760, "Route D": 1050}
base_days = {"Route A": 1.3, "Route B": 2.4, "Route C": 3.8, "Route D": 5.0}
distance = np.array([distance_map[r] for r in routes], dtype=float)
base = np.array([base_days[r] for r in routes])
weather = rng.choice(["Normal", "Rain", "Severe"], size=n, p=[0.72, 0.23, 0.05])
weekend = rng.choice(["No", "Yes"], size=n, p=[0.72, 0.28])
volume = rng.integers(1, 31, size=n)

transit = (
    base + distance / 900
    + np.where(weather == "Rain", 0.7, np.where(weather == "Severe", 1.8, 0))
    + np.where(weekend == "Yes", 0.35, 0)
    + rng.normal(0, 0.55, size=n)
)
transit = np.maximum(transit, 0.25)
cost = 250 + distance * 1.45 + volume * 18
cost += np.where(weather == "Severe", 120, 0) + rng.normal(0, 100, size=n)
cost = np.maximum(cost, 100)

raw = pd.DataFrame({
    "shipment_id": [f"SHP{i:04d}" for i in range(1, n + 1)],
    "route": routes,
    "distance_km": distance,
    "weather_condition": weather,
    "weekend_dispatch": weekend,
    "shipment_volume": volume,
    "transit_days": np.round(transit, 2),
    "shipping_cost_inr": np.round(cost, 2),
})

# Add a small number of missing values and long transit-time anomalies for practice.
missing_idx = rng.choice(raw.index, size=9, replace=False)
raw.loc[missing_idx[:4], "transit_days"] = np.nan
raw.loc[missing_idx[4:7], "shipping_cost_inr"] = np.nan
raw.loc[missing_idx[7:], "weather_condition"] = None
outlier_idx = rng.choice(raw.index, size=3, replace=False)
raw.loc[outlier_idx, "transit_days"] = (
    raw.loc[outlier_idx, "transit_days"].fillna(0) * 3 + 8
).round(2)

raw.to_csv(OUT / "data" / "simulated_logistics_data_raw.csv", index=False)

clean = raw.copy()
clean["weather_condition"] = clean["weather_condition"].fillna("Unknown")
for col in ["transit_days", "shipping_cost_inr"]:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean[col] = clean.groupby("route")[col].transform(lambda s: s.fillna(s.median()))

def iqr_flags(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

clean["transit_outlier"] = iqr_flags(clean["transit_days"])
clean["cost_outlier"] = iqr_flags(clean["shipping_cost_inr"])
clean["cost_per_km"] = clean["shipping_cost_inr"] / clean["distance_km"]
clean["delay_over_3_days"] = clean["transit_days"] > 3

clean.to_csv(OUT / "data" / "simulated_logistics_data_cleaned.csv", index=False)
clean.describe(include="all").transpose().to_csv(OUT / "summary_statistics.csv")
route_summary = clean.groupby("route").agg(
    shipments=("shipment_id", "count"),
    avg_transit_days=("transit_days", "mean"),
    median_transit_days=("transit_days", "median"),
    avg_shipping_cost_inr=("shipping_cost_inr", "mean"),
    avg_volume=("shipment_volume", "mean"),
).round(2)
route_summary.to_csv(OUT / "route_summary.csv")

# 1. Histogram
plt.figure(figsize=(7.2, 4.4))
plt.hist(clean["transit_days"], bins=18, edgecolor="black")
plt.title("Distribution of Shipment Transit Times")
plt.xlabel("Transit time (days)")
plt.ylabel("Number of shipments")
plt.tight_layout()
plt.savefig(OUT / "visualizations" / "transit_histogram.png", dpi=160)
plt.close()

# 2. Box plot
plt.figure(figsize=(7.2, 4.4))
plt.boxplot([clean.loc[clean.route == r, "transit_days"] for r in
             ["Route A", "Route B", "Route C", "Route D"]],
            labels=["Route A", "Route B", "Route C", "Route D"])
plt.title("Transit Time by Route")
plt.xlabel("Route")
plt.ylabel("Transit time (days)")
plt.tight_layout()
plt.savefig(OUT / "visualizations" / "route_boxplot.png", dpi=160)
plt.close()

# 3. Scatter plot
plt.figure(figsize=(7.2, 4.4))
for condition in ["Normal", "Rain", "Severe", "Unknown"]:
    subset = clean[clean.weather_condition == condition]
    plt.scatter(subset["distance_km"], subset["transit_days"], label=condition, alpha=0.75)
plt.title("Distance vs Transit Time by Weather")
plt.xlabel("Distance (km)")
plt.ylabel("Transit time (days)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "visualizations" / "distance_transit_scatter.png", dpi=160)
plt.close()

# 4. Bar chart
plt.figure(figsize=(7.2, 4.4))
route_summary["avg_shipping_cost_inr"].plot(kind="bar")
plt.title("Average Shipping Cost by Route")
plt.xlabel("Route")
plt.ylabel("Average shipping cost (INR)")
plt.tight_layout()
plt.savefig(OUT / "visualizations" / "route_cost_bar.png", dpi=160)
plt.close()

# 5. Correlation heatmap
corr = clean[["distance_km", "shipment_volume", "transit_days", "shipping_cost_inr"]].corr()
plt.figure(figsize=(6.4, 4.6))
plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar(label="Correlation coefficient")
plt.title("Correlation Between Logistics Metrics")
plt.tight_layout()
plt.savefig(OUT / "visualizations" / "correlation_heatmap.png", dpi=160)
plt.close()

print("EDA complete. Files saved in:", OUT.resolve())
print("\nRoute summary:\n", route_summary)
print("\nPotential transit-time outliers:", int(clean["transit_outlier"].sum()))
