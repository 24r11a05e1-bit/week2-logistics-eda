# Week 2 Logistics EDA Project

**Author:** Ramidi Bhaavani  
**College:** Geetanjali College of Engineering and Technology

## Overview
This project demonstrates exploratory data analysis (EDA) for a simulated logistics dataset of 300 shipments. It explores route, distance, weather, shipment volume, transit time, and shipping cost.

> **Important:** The dataset is synthetic and does not represent real company performance.

## Requirements
- Python 3
- pandas
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install pandas numpy matplotlib
```

## Run
```bash
python week2_logistics_eda.py
```

The script creates a `week2_output` folder containing raw and cleaned CSV files, summary tables, and five visualizations.

## Methods
- Generate repeatable synthetic data with NumPy.
- Handle missing values with route-level medians and an `Unknown` category.
- Flag potential outliers using the interquartile range (IQR).
- Calculate route summaries and cost per kilometre.
- Create a histogram, box plot, scatter plot, bar chart, and correlation heatmap.

## Limitations
The patterns are partly built into the simulation. They should be treated as practice results, not real operational findings.
