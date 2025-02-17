from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os.path
import joblib
import pandas as pd
import os.path
import sys
import os
# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.features import KEY_FEATURES

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app)

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'cots_predictor.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# Get the project root directory (two levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'cots_predictor_pipeline.joblib')

# Load the model
try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Model not found. Please run predict_cots_changes.py first to train the model.")
    exit(1)

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
        'has_parrotfish': 1 if features['Parrotfish'] > 0 else 0,
        'has_wrasse': 1 if features['Humphead Wrasse'] > 0 else 0,
        'total_giant_clam': float(data.get('total_giant_clam', 0)),
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
