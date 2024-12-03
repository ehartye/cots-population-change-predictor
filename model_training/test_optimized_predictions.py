import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def test_optimized_model():
    # Load the optimized model
    try:
        pipeline = joblib.load('cots_predictor_pipeline_optimized.joblib')
    except FileNotFoundError:
        print("Optimized model not found. Please run optimize_cots_model.py first.")
        return

    conn = sqlite3.connect('reefcheck.db')
    
    # Get all COTS change events
    print("Testing optimized model on known COTS changes...")
    cots_changes = pd.read_sql_query("""
        SELECT 
            site_id,
            survey_id_1,
            survey_id_2,
            change,
            total_cots_1,
            total_cots_2,
            date_1,
            date_2,
            CASE 
                WHEN change > 0 THEN 'increase'
                WHEN change < 0 THEN 'decrease'
            END as actual_outcome
        FROM cots_changes
        ORDER BY date_1
    """, conn)
    
    # Get site names for better reporting
    sites = pd.read_sql_query("""
        SELECT DISTINCT s.site_id, s.reef_name 
        FROM site_description s
        JOIN belt b ON b.site_id = s.site_id
    """, conn)
    
    # Prepare features for all surveys
    from predict_cots_changes import prepare_features
    all_features = prepare_features()
    
    print("\n=== Testing Optimized Model Against Known COTS Changes ===\n")
    
    correct_predictions = 0
    total_predictions = 0
    all_true = []
    all_pred = []
    
    for _, event in cots_changes.iterrows():
        # Get the survey data before the change
        survey_data = all_features[all_features['survey_id'] == event['survey_id_1']].copy()
        
        if not survey_data.empty:
            # Prepare features
            X = survey_data.drop(['survey_id', 'site_id', 'target'], axis=1)
            
            # Get predictions
            prediction = pipeline.predict(X)[0]
            probabilities = pipeline.named_steps['classifier'].predict_proba(
                pipeline.named_steps['scaler'].transform(X)
            )[0]
            
            # Get site name
            site_name = sites[sites['site_id'] == event['site_id']]['reef_name'].iloc[0] \
                if not sites[sites['site_id'] == event['site_id']].empty else "Unknown Site"
            
            print(f"\nSite: {site_name}")
            print(f"Date range: {event['date_1']} to {event['date_2']}")
            print(f"COTS change: {event['total_cots_1']} → {event['total_cots_2']} (Change: {event['change']})")
            print(f"Actual outcome: {event['actual_outcome']}")
            print(f"Model prediction: {prediction}")
            print("\nPrediction probabilities:")
            for class_label, prob in zip(pipeline.named_steps['classifier'].classes_, probabilities):
                print(f"{class_label}: {prob:.2%}")
            
            # Show key indicators
            print("\nKey indicators at time of prediction:")
            important_features = [
                'COTS', 'COTS_presence', 'total_giant_clam',
                'Coral Damage Other', 'Bleaching (% Of Population)',
                'Bleaching (% Of Colony)', 'Parrotfish', 'total_grouper'
            ]
            for feature in important_features:
                if feature in X.columns:
                    value = X[feature].iloc[0]
                    if value > 0:
                        print(f"{feature}: {value:.2f}")
            
            # Track accuracy
            if prediction == event['actual_outcome']:
                correct_predictions += 1
                print("\nResult: ✓ Correct prediction")
            else:
                print("\nResult: ✗ Incorrect prediction")
            
            total_predictions += 1
            all_true.append(event['actual_outcome'])
            all_pred.append(prediction)
            
            print("-" * 80)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total events tested: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    print(f"Accuracy on known events: {accuracy:.1f}%")
    
    # Get unique classes that actually appear in the data
    unique_classes = sorted(list(set(all_true)))
    
    print("\nDetailed Performance Metrics:")
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred, 
                              labels=unique_classes,  # Only include classes that appear in the data
                              zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_pred, labels=unique_classes))
    
    conn.close()

if __name__ == "__main__":
    test_optimized_model()
