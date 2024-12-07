import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import os.path
from .features import KEY_FEATURES, prepare_features
from .test_cots_predictions import test_known_events

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'reefcheck.db')

def train_model():
    print("Preparing features...")
    df = prepare_features()
    
    # Prepare features and target
    X = df.drop(['survey_id', 'site_id', 'target'], axis=1)
    y = df['target']
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Create pipeline with SMOTE and RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,  # Increased depth to capture more complex patterns
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Train the model
    print("\nTraining model with SMOTE oversampling...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Performance:")
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Save the pipeline
    model_path = os.path.join(SCRIPT_DIR, 'cots_predictor_pipeline.joblib')
    joblib.dump(pipeline, model_path)
    print("\nModel pipeline saved to disk.")
    
    return pipeline

def predict_site(site_id, pipeline=None):
    """Predict COTS change probability for a specific site"""
    if pipeline is None:
        model_path = os.path.join(SCRIPT_DIR, 'cots_predictor_pipeline.joblib')
        pipeline = joblib.load(model_path)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get the latest survey for the site
    latest_survey = pd.read_sql_query("""
        SELECT survey_id, date
        FROM belt
        WHERE site_id = ?
        GROUP BY survey_id
        ORDER BY date DESC
        LIMIT 1
    """, conn, params=[site_id])
    
    if latest_survey.empty:
        print(f"No surveys found for site {site_id}")
        return None
    
    # Get features for the latest survey
    df = prepare_features()
    site_data = df[df['site_id'] == site_id].copy()
    
    if site_data.empty:
        print(f"Insufficient data for site {site_id}")
        return None
    
    # Prepare features
    X = site_data.drop(['survey_id', 'site_id', 'target'], axis=1)
    
    # Get predictions and probabilities
    prediction = pipeline.predict(X)
    probabilities = pipeline.named_steps['classifier'].predict_proba(
        pipeline.named_steps['scaler'].transform(X)
    )
    
    print(f"\nPrediction for site {site_id}:")
    print(f"Latest survey date: {latest_survey['date'].iloc[0]}")
    print(f"Predicted outcome: {prediction[0]}")
    print("\nProbabilities:")
    for class_label, prob in zip(pipeline.named_steps['classifier'].classes_, probabilities[0]):
        print(f"{class_label}: {prob:.2%}")
        
    # Print key indicators if probability of change is high
    if prediction[0] != 'none' or max(probabilities[0]) < 0.8:
        print("\nKey Indicators:")
        # Use all features from KEY_FEATURES list
        for feature in KEY_FEATURES:
            # if feature in X.columns and X[feature].iloc[0] > 0:
            print(f"{feature}: {X[feature].iloc[0]:.2f}")
    
    conn.close()
    return prediction[0], probabilities[0]

if __name__ == "__main__":
    print("Training COTS change prediction model...")
    pipeline = train_model()
    
    print("\nTesting model against known COTS changes...")
    test_known_events()
