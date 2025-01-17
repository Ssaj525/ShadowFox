# Car Selling Price Prediction Project

This project focuses on building a machine learning model to estimate the selling price of cars based on various attributes. The dataset includes information such as fuel type, age of the car, showroom price, mileage, seller details, transmission type, and ownership history.

## Problem Overview

The goal is to develop a predictive model that accurately estimates the selling price of a car using its features. This system aims to assist users in determining a fair market value for their vehicles.

## Dataset Features

- **Selling_Price**: The target variable representing the car's selling price (in lakhs).
- **Present_Price**: The original showroom price (in lakhs).
- **Kms_Driven**: Total kilometers the car has been driven.
- **Fuel_Type**: Type of fuel the car uses (e.g., Petrol, Diesel, CNG).
- **Seller_Type**: Indicates if the seller is a dealer or an individual.
- **Transmission**: The transmission type (Manual or Automatic).
- **Owner**: Number of previous owners.
- **Years_Used**: Calculated as the difference between the current year and the car’s year of manufacture.

## Tools and Technologies

- **Python 3.8+**
- Core Libraries:
  - **pandas**: For data manipulation.
  - **numpy**: For numerical computations.
  - **matplotlib & seaborn**: For data visualization.
  - **scikit-learn**: For building the machine learning model.
  - **joblib**: For model persistence.
  - **gunicorn**: For deploying the application.
  - **flask**: For creating the web application interface.

## Example Prediction

1. **Input Features**:

   - Fuel Type: Petrol
   - Present Price: ₹12.5 lakhs
   - Kilometers Driven: 15,000 km
   - Car Year: 2022
   - Seller Type: Individual
   - Transmission: Manual
   - Owner: 0 (First Owner)

2. **Predicted Output**:
   - Estimated Selling Price: **₹6.06 lakhs**
