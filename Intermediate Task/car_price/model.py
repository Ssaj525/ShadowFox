import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data
url = "./car.csv"
df = pd.read_csv(url)

# Preprocess the data
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df['Year'] = df['Year'].dt.year
df['Selling_Price'] = df['Selling_Price'].astype(float)
df['Present_Price'] = df['Present_Price'].astype(float)
df['Kms_Driven'] = df['Kms_Driven'].astype(float)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Prepare features and target
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'car_price_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model trained and saved successfully.")