import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the data
url = "./boston.csv"
df = pd.read_csv(url)

# Preprocess the data
df = df.astype(float)

# Prepare features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

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
joblib.dump(model, 'boston_house_price_model.joblib')
joblib.dump(scaler, 'boston_scaler.joblib')

print("Model trained and saved successfully.")