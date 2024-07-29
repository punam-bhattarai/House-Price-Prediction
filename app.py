# Importing required libraries
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained ANN model
ann_model = load_model('best_model_ann.h5', custom_objects={'mse': 'mean_squared_error'})

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # List all the columns/features that the user needs to input
    input_features = [
        'brokered_by', 'status', 'bed', 'bath', 'acre_lot', 
        'street', 'city', 'state', 'zip_code', 'house_size', 
        'price_per_sqft', 'total_rooms', 'house_age'
    ]
    # Collecting the data from the form
    data = [float(request.form[feature]) for feature in input_features]
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    prediction = ann_model.predict(data)
    output = prediction[0][0]  # Adjusting for the shape of ANN prediction output
    return render_template('index.html', prediction_text=f'Predicted House Price: ${output:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
