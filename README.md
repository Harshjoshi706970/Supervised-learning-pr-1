🏠 Predictive Insight Engine — House Price Prediction & Regression Analysis

A comprehensive supervised machine learning project designed to estimate residential real estate prices (house_price_inr) and analyze model behavior across Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, and custom Gradient Descent optimizers.

📋 Table of Contents

Overview & Problem Statement

Dataset Description

Project Workflow

Model Comparison & Results

Gradient Descent Optimization Analysis

Key Findings & Business Insights

Getting Started

Repository Structure

🎯 Overview & Problem Statement

Accurately valuing residential properties requires understanding how structural and locational characteristics drive market prices. The objective of this project is to:

Explore and visualize correlations between property features and house prices.

Design, implement, and validate regression models (Simple, Multiple, and Polynomial Regression).

Implement Batch, Stochastic (SGD), and Mini-Batch Gradient Descent from scratch to compare convergence speed and training efficiency.

Diagnose bias-variance trade-offs, underfitting, and overfitting to recommend the optimal production model.

📊 Dataset Description

Source File: RealEstate_HousePrice_Dataset_4200.csv

Dimensions: 4,200 rows $\times$ 12 columns

Data Quality: Clean dataset with 0 missing (null) values

Train / Test Split: 80% Training (3,360 samples) / 20% Testing (840 samples) with random_state=42

Feature                Data Type    Description
house_id               int64        Unique property listing identifier
area_sqft              int64        Total living floor area in square feet
bedrooms               int64        Number of bedrooms
bathrooms              int64        Number of bathrooms
location_score         float64      Neighborhood/location desirability rating
age_years              int64        Age of the property in years
distance_city_km       float64      Distance from city center in kilometers
lot_size_sqft          int64        Total land/lot area in square feet
has_garage             int64        Binary indicator (1 = Yes, 0 = No)
has_pool               int64        Binary indicator (1 = Yes, 0 = No)
renovation_years_ago   int64        Years since last renovation
house_price_inr        int64        Target Variable: Property price in Indian Rupees (INR)

🔄 Project Workflow

Data Acquisition & Pre-Profiling: Verified dataset shape (4200, 12), data types, and confirmed zero missing values.

Exploratory Data Analysis (EDA): Generated correlation heatmaps identifying primary positive price drivers (area_sqft, location_score, bathrooms, bedrooms, and lot_size_sqft).

Simple Linear Regression (area_sqft $\rightarrow$ house_price_inr):

Extracted learned parameters: Slope = ₹14,788.31 / sq. ft., Intercept = -₹1,163,519.18.

Validated core regression assumptions: Linearity, Independence of Errors, Homoscedasticity, Normality of Residuals, and Multicollinearity.

Multiple Linear Regression: Trained on key relevant features (area_sqft, bedrooms, bathrooms, location_score, lot_size_sqft) and full feature sets to eliminate omitted variable bias.

Polynomial Regression (Degrees 2, 3, and 5): Evaluated non-linear curve fitting on both non-linear synthetic benchmarks ($R^2$ jumped from -0.008 to 0.946) and area_sqft to test for non-linearity vs. underfitting.

Gradient Descent From Scratch: Implemented and benchmarked Batch GD, Stochastic GD, and Mini-Batch GD (batch_size=16).

Bias-Variance & Complexity Analysis: Compared training vs. testing $R^2$ scores to identify the optimal bias-variance balance.

📈 Model Comparison & Results

1. Real Estate Valuation Models (house_price_inr)

Metric

Simple Linear Regression (area_sqft)

Multiple Linear Regression (5 Core Features)

Multiple Linear Regression (All 10 Features)

MSE

$6.70 \times 10^{13}$

$1.36 \times 10^{13}$

—

MAE

₹62,94,593.70 (~₹62.95 L)

₹27,76,989.67 (~₹27.77 L)

—

RMSE

₹81,84,696.70 (~₹81.85 L)

₹36,91,486.26 (~₹36.91 L)

—

Test $R^2$ Score

0.5625 (56.25%)

0.9110 (91.10%)

0.9180 (91.80%)

Adjusted $R^2$

0.5620

0.9105

—

Fit Diagnosis

High Bias / Underfitting

Low Bias, Low Variance

Optimal Fit

2. Polynomial Regression Analysis

On area_sqft (Degrees 2, 3, 5): Train $R^2$ (0.572–0.573) and Test $R^2$ (0.563) remained virtually unchanged compared to Simple Linear Regression, proving that the area_sqft vs. house_price_inr relationship is fundamentally linear and underfitting was caused by omitted features rather than polynomial degree.

On Non-Linear Data (df2): Polynomial Regression (Degree 2) improved $R^2$ from -0.0081 (Linear) to 0.9456, reducing RMSE from 2.574 to 0.598.

⚡ Gradient Descent Optimization Analysis

Custom implementations optimizing the intercept parameter ($b$) on the training split (3,360 rows) with learning_rate = 0.0001:

Optimization Method

Samples per Update

Epochs

Execution Time

Final Intercept ($b$)

Test $R^2$

Scikit-Learn OLS

Closed-Form

—

—

-1,163,519.18

0.562520

Batch Gradient Descent

All 3,360 rows

20 (converged)

0.0014 s

-1,163,519.18

0.562520

Stochastic GD (SGD)

1 row

50

0.4003 s

-1,193,898.36

0.562220

Mini-Batch GD

16 rows

200

0.3369 s

-1,163,233.63

0.562523

💡 Key Findings & Business Insights

Best-Performing Model: Multiple Linear Regression provides the best bias-variance balance (Train R² = 0.925, Test R² = 0.918, train-test gap of only 0.007), cutting MAE by ~55.9% compared to Simple Linear Regression.

Underfitting vs. Overfitting: Relying on floor area alone underfits the market (Test R² = 0.563), systematically undervaluing luxury/prime-location homes and overvaluing smaller/distant listings.

Practical Business Value: While living area adds a baseline of ~₹14,788 per sq. ft., accurate automated appraisals require factoring in location quality (location_score), interior capacity (bedrooms, bathrooms), and lot size.

🚀 Getting Started

Prerequisites

Ensure you have Python 3.8+ and the following libraries installed:

pip install numpy pandas matplotlib seaborn scikit-learn jupyter


Running the Notebook

Clone this repository:

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


Place RealEstate_HousePrice_Dataset_4200.csv in your working directory and update the file path in the data acquisition cell of SL_pr1.ipynb if needed.

Launch Jupyter Notebook:

jupyter notebook SL_pr1.ipynb


📁 Repository Structure

├── SL_pr1.ipynb                              # Main Jupyter Notebook with EDA, models, GD & analysis
├── RealEstate_HousePrice_Dataset_4200.csv    # Dataset (4,200 real estate records, 12 features)
└── README.md                                 # Project documentation
