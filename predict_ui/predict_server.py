from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os.path

app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'cots_predictor_pipeline.joblib')

# Load the model
try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Model not found. Please run predict_cots_changes.py first to train the model.")
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
    
    # Update features based on input data
    features.update({
        'Grouper > 60 cm': float(data.get('grouper_60_plus', 0)),
        'Grouper 30-40 cm': float(data.get('grouper_30_40', 0)),
        'Grouper 40-50 cm': float(data.get('grouper_40_50', 0)),
        'Grouper Total': float(data.get('grouper_total', 0)),
        'Giant Clam 40-50 cm': float(data.get('giant_clam_40_50', 0)),
        'Giant Clam > 50 cm': float(data.get('giant_clam_50_plus', 0)),
        'Bleaching (% Of Population)': float(data.get('bleaching_population', 0)),
        'Bleaching (% Of Colony)': float(data.get('bleaching_colony', 0)),
        'Coral Damage Other': float(data.get('coral_damage', 0)),
        'Humphead Wrasse': float(data.get('humphead_wrasse', 0)),
        'Parrotfish': float(data.get('parrotfish', 0)),
        'Tripneustes': float(data.get('tripneustes', 0))
    })
    
    # Set derived features
    features.update({
        'has_bleaching': 1 if features['Bleaching (% Of Population)'] > 0 else 0,
        'has_coral_damage': 1 if features['Coral Damage Other'] > 0 else 0,
        'total_grouper': features['Grouper Total'],
        'has_parrotfish': 1 if features['Parrotfish'] > 0 else 0,
        'has_wrasse': 1 if features['Humphead Wrasse'] > 0 else 0,
        'total_giant_clam': features['Giant Clam 40-50 cm'] + features['Giant Clam > 50 cm'],
        'has_tripneustes': 1 if features['Tripneustes'] > 0 else 0
    })
    
    try:
        # Create DataFrame with features
        X = pd.DataFrame([features])
        print("Feature vector:", X)  # Debug print
        
        # Get predictions
        prediction = pipeline.predict(X)[0]
        probabilities = pipeline.named_steps['classifier'].predict_proba(
            pipeline.named_steps['scaler'].transform(X)
        )[0]
        
        # Create response with prediction and probabilities
        response = {
            'prediction': prediction,
            'probabilities': {
                class_label: float(prob) * 100
                for class_label, prob in zip(pipeline.named_steps['classifier'].classes_, probabilities)
            },
            'key_indicators': {
                feature: float(features[feature])
                for feature in KEY_FEATURES
                if features[feature] > 0
            }
        }
        
        print("Response:", response)  # Debug print
        return jsonify(response)
        
    except Exception as e:
        print("Error during prediction:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
