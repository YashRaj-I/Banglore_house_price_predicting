from flask import Flask, request, jsonify
from flask import render_template

import pickle
import pandas as pd
import json  # Add this import for JSON handling

# Load the trained model and columns
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    columns = json.load(f)  # Now json is defined, it will work
    data_columns = columns['data_columns']

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Get input from the user (JSON format)
        data = request.get_json()

        location = data['location']
        sqft = data['sqft']
        bath = data['bath']
        bhk = data['bhk']

        # Prepare input data for the model
        x = [0] * len(data_columns)
        x[data_columns.index(location)] = 1  # One-hot encoding for location
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # Make prediction
        prediction = model.predict([x])[0]

        return jsonify({'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
