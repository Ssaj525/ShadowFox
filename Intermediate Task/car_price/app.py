from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('car_price_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Prepare the input data
    input_data = pd.DataFrame({
        'Year': [2024 - int(data['year'])],  # Calculate car age
        'Present_Price': [float(data['presentPrice'])],
        'Kms_Driven': [float(data['kmsDriven'])],
        'Owner': [int(data['owner'])],
        'Fuel_Type_Diesel': [1 if data['fuelType'] == 'Diesel' else 0],
        'Fuel_Type_Petrol': [1 if data['fuelType'] == 'Petrol' else 0],
        'Seller_Type_Individual': [1 if data['sellerType'] == 'Individual' else 0],
        'Transmission_Manual': [1 if data['transmission'] == 'Manual' else 0]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return jsonify({'prediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)