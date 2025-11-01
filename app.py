# app.py - COMPLETE WORKING VERSION
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)


class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.train_model()

    def train_model(self):
        # Create realistic training data
        np.random.seed(42)

        # Generate 200 sample houses
        size = np.random.randint(800, 4000, 200)
        bedrooms = np.random.randint(1, 6, 200)
        age = np.random.randint(0, 50, 200)
        location = np.random.choice([1, 2, 3], 200)  # 1=Urban, 2=Suburban, 3=Rural

        # Create realistic prices based on features
        base_price = 50000
        price = (base_price +
                 size * 150 +
                 bedrooms * 30000 -
                 age * 1000 +
                 location * 20000 +
                 np.random.normal(0, 20000, 200))

        # Ensure prices are realistic
        price = np.maximum(price, 50000)

        X = np.column_stack([size, bedrooms, age, location])
        y = price

        # Train the model
        self.model.fit(X, y)
        print("✅ ML Model trained successfully!")


# Initialize the predictor
predictor = HousePricePredictor()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])
        location = int(request.form['location'])

        # Validate inputs
        if not (500 <= size <= 10000):
            return jsonify({'success': False, 'error': 'Size must be between 500-10000 sq ft'})
        if not (1 <= bedrooms <= 10):
            return jsonify({'success': False, 'error': 'Bedrooms must be 1-10'})
        if not (0 <= age <= 100):
            return jsonify({'success': False, 'error': 'Age must be 0-100 years'})
        if location not in [1, 2, 3]:
            return jsonify({'success': False, 'error': 'Location must be 1, 2, or 3'})

        # Make prediction
        features = np.array([[size, bedrooms, age, location]])
        predicted_price = predictor.model.predict(features)[0]

        # Format location name
        location_names = {1: "Urban ", 2: "Suburban ", 3: "Rural "}

        return jsonify({
            'success': True,
            'predicted_price': f"${predicted_price:,.2f}",
            'details': f"• Size: {size:,} sq ft\n• Bedrooms: {bedrooms}\n• Age: {age} years\n• Location: {location_names[location]}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Please check your inputs: {str(e)}'})


if __name__ == '__main__':
    print("House Price Predictor Web App Started!")
    print("ML Model loaded and ready to use")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True)