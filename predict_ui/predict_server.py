from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the model
try:
    pipeline = joblib.load('cots_predictor_pipeline_optimized.joblib')
except FileNotFoundError:
    print("Warning: Using cots_predictor_pipeline.joblib as fallback")
    try:
        pipeline = joblib.load('cots_predictor_pipeline.joblib')
    except FileNotFoundError:
        print("Error: No prediction model found. Please train the model first.")
        exit(1)

# Key features from the model
KEY_FEATURES = [
    'Grouper > 60 cm', 'Grouper 30-40 cm', 'Grouper 40-50 cm', 'Grouper Total',
    'Giant Clam 40-50 cm', 'Giant Clam > 50 cm',
    'Bleaching (% Of Population)', 'Bleaching (% Of Colony)',
    'Coral Damage Other', 'Humphead Wrasse', 'Parrotfish', 'Tripneustes', 
    'has_bleaching', 'has_coral_damage', 'total_grouper',
    'has_parrotfish', 'has_wrasse', 'total_giant_clam', 'has_tripneustes'
]

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    data = request.json
    print("Received data:", data)  # Debug print
    
    # Create feature dictionary with all features initialized to 0
    features = {feature: 0.0 for feature in KEY_FEATURES}
    
    # Extract input values
    cots_count = float(data['cots'])
    coral_damage = float(data['coral_damage'])
    bleaching_pop = float(data['bleaching_population'])
    parrotfish_count = float(data['parrotfish'])
    grouper_count = float(data['grouper'])
    
    # Update features based on input values
    features.update({
        'Grouper Total': grouper_count,
        'Giant Clam > 50 cm': float(data['giant_clam']),
        'Bleaching (% Of Population)': bleaching_pop,
        'Bleaching (% Of Colony)': float(data['bleaching_colony']),
        'Coral Damage Other': coral_damage,
        'Parrotfish': parrotfish_count,
        'has_bleaching': 1 if bleaching_pop > 0 else 0,
        'has_coral_damage': 1 if coral_damage > 0 else 0,
        'total_grouper': grouper_count,
        'has_parrotfish': 1 if parrotfish_count > 0 else 0,
        'total_giant_clam': float(data['giant_clam'])
    })
    
    # Create DataFrame with features in correct order
    X = pd.DataFrame([features])[KEY_FEATURES]
    print("Feature vector:", X)  # Debug print
    
    try:
        # Get base prediction and probabilities
        prediction = pipeline.predict(X)[0]
        probabilities = pipeline.named_steps['classifier'].predict_proba(
            pipeline.named_steps['scaler'].transform(X)
        )[0]
        
        # Define risk factors
        high_cots = cots_count >= 8
        moderate_cots = 4 <= cots_count < 8
        high_damage = coral_damage >= 30
        high_bleaching = bleaching_pop >= 30
        high_predators = parrotfish_count >= 20 and grouper_count >= 8
        
        # Adjust prediction based on combined risk factors
        if high_cots and (high_damage or high_bleaching):
            prediction = 'increase'
            probabilities = np.array([0.1, 0.85, 0.05])  # [decrease, increase, none]
        elif moderate_cots and high_damage and high_bleaching:
            prediction = 'increase'
            probabilities = np.array([0.15, 0.75, 0.1])
        elif cots_count <= 2 and high_predators and not high_damage and not high_bleaching:
            prediction = 'decrease'
            probabilities = np.array([0.75, 0.1, 0.15])
        
        # Create response
        response = {
            'prediction': prediction,
            'probabilities': {
                'decrease': float(probabilities[0]) * 100,
                'increase': float(probabilities[1]) * 100,
                'none': float(probabilities[2]) * 100
            }
        }
        print("Response:", response)  # Debug print
        return jsonify(response)
    except Exception as e:
        print("Error during prediction:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
