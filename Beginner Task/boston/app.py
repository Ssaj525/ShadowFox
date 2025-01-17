from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('boston_house_price_model.joblib')
scaler = joblib.load('boston_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Prepare the input data
    input_data = pd.DataFrame({
        'CRIM': [float(data['crim'])],
        'ZN': [float(data['zn'])],
        'INDUS': [float(data['indus'])],
        'CHAS': [float(data['chas'])],
        'NOX': [float(data['nox'])],
        'RM': [float(data['rm'])],
        'AGE': [float(data['age'])],
        'DIS': [float(data['dis'])],
        'RAD': [float(data['rad'])],
        'TAX': [float(data['tax'])],
        'PTRATIO': [float(data['ptratio'])],
        'B': [float(data['b'])],
        'LSTAT': [float(data['lstat'])]
    })
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Convert the prediction to thousands of dollars
    prediction_thousands = prediction * 1000
    
    return jsonify({'prediction': round(prediction_thousands, 2)})

if __name__ == '__main__':
    app.run(debug=True)

