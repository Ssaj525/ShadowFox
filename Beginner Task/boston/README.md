# Boston Housing Price Prediction

This project aims to predict housing prices in Boston using a regression model. The dataset includes various features such as crime rate, number of rooms, and proximity to amenities. By preprocessing the data, exploring feature relationships, and building a predictive model, the goal is to provide accurate predictions and insights into the factors that influence Boston's housing market.

---

## **Key Components**

- **Data Preprocessing**:

  - Address missing values with imputation techniques.
  - Standardize the dataset to enhance model performance.

- **Exploratory Data Analysis**:

  - Utilize correlation heatmaps and visualizations to identify patterns and trends.

- **Model Development**:

  - Implement Linear Regression to forecast housing prices.
  - Evaluate the model using metrics such as Mean Squared Error (MSE) and R² Score.

- **Visualization**:
  - Generate scatter plots for actual vs. predicted prices.
  - Create heatmaps to illustrate feature correlations.

---

## **Dataset Overview**

The dataset consists of 13 predictor variables and 1 target variable (`MEDV` - Median value of homes).

| **Feature** | **Description**                                        |
| ----------- | ------------------------------------------------------ |
| CRIM        | Per capita crime rate                                  |
| ZN          | Proportion of residential land zoned for large lots    |
| RM          | Average number of rooms per dwelling                   |
| DIS         | Weighted distances to employment centers               |
| TAX         | Full-value property tax rate                           |
| MEDV        | Median value of owner-occupied homes (Target Variable) |

---

## **Outputs**

- Heatmap showcasing feature relationships.
- Evaluation metrics (MSE, RMSE, R² Score).
- Scatter plot comparing actual and predicted prices.

## **Results Summary**

The model's performance metrics include:

- **Mean Squared Error (MSE):** computed value
- **Root Mean Squared Error (RMSE):** computed value
- **R² Score:** computed value

## **Key Visualizations**

### **Correlation Heatmap**

Illustrates the relationship between different features and the target variable (MEDV).

### **Actual vs. Predicted Prices**

Graphically compares predicted prices with actual values to assess model accuracy.

## **Technologies Utilized**

- Python 3.8+
- Key Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib
  - gunicorn
  - flask
