# Predictive Insight Engine

Residential real estate price prediction and market valuation analysis using supervised machine learning.

## Introduction
This project uses a residential real estate dataset designed for property price prediction and market valuation analysis.

- **Dataset Size:** 4,200 records (rows) and 12 features (columns) 
- **Data Quality:** Clean dataset with no missing (null) values 
- **Target Variable:** `house_price_inr` 

## Problem Statement
The goal is to build a predictive system that estimates house prices based on property attributes such as size, number of rooms, location score, and property age . This involves designing, implementing, analyzing, and evaluating supervised learning models to predict house prices accurately and understand differences in model performance .

## Dataset Features
The dataset contains the following columns :

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `house_id` | `int64` | Unique property identifier  |
| `area_sqft` | `int64` | Interior living area in square feet  |
| `bedrooms` | `int64` | Number of bedrooms  |
| `bathrooms` | `int64` | Number of bathrooms  |
| `location_score` | `float64` | Location rating score  |
| `age_years` | `int64` | Age of the property in years  |
| `distance_city_km` | `float64` | Distance from the city center in kilometers  |
| `lot_size_sqft` | `int64` | Total lot size in square feet  |
| `has_garage` | `int64` | Garage availability indicator (`1` = yes, `0` = no)  |
| `has_pool` | `int64` | Swimming pool availability indicator (`1` = yes, `0` = no)  |
| `renovation_years_ago` | `int64` | Years elapsed since last renovation  |
| `house_price_inr` | `int64` | Target variable: property price in INR  |

## Workflow & Implementation
1. **Import Libraries:** Load core data science and modeling libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`) .
2. **Data Acquisition:** Load the CSV dataset into a pandas DataFrame and inspect the first few rows . *(Note: Update the file path in the notebook to a relative path such as `data/RealEstate_HousePrice_Dataset_4200.csv` before sharing on GitHub .)*
3. **Data Pre-Profiling:** Check dataset dimensions `(4200, 12)`, column data types, and verify zero missing values .
4. **Feature & Target Identification:** Compute numeric correlations and heatmaps to examine relationships with `house_price_inr`, identifying key related features such as `area_sqft`, `bedrooms`, `bathrooms`, `location_score`, and `lot_size_sqft` .
5. **Train-Test Split:** Split the data into training and testing sets (`test_size=0.2`, `random_state=42`) .
6. **Simple Linear Regression:** Train a `LinearRegression` model using `area_sqft` to predict `house_price_inr` on the test set and compare actual vs. predicted values .
7. **Regression Line & Parameters:** Plot the scatter plot and fitted regression line, and inspect the learned slope and intercept .
8. **Assumption Checks:** Compute residuals (`y1_test - y_pred`) and inspect plots for linearity, error independence/autocorrelation, and homoscedasticity .

## Getting Started

### Prerequisites
Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
