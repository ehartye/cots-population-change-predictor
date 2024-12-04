import sqlite3
import pandas as pd
import joblib
import os.path
from model_training.predict_cots_changes import prepare_features, KEY_FEATURES

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'reefcheck.db')

def predict_recent_changes():
    # Load the optimized model
    try:
        model_path = os.path.join(SCRIPT_DIR, 'cots_predictor_pipeline_optimized.joblib')
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print("Optimized model not found. Please run optimize_cots_model.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # Get all 2021-2023 surveys using the year field
    print("Analyzing 2021-2023 surveys for potential COTS changes...")
    recent_surveys = pd.read_sql_query("""
        SELECT DISTINCT b.survey_id, b.site_id, b.date, b.year, s.reef_name
        FROM belt b
        JOIN site_description s ON b.site_id = s.site_id
        WHERE b.year IN (2021, 2022, 2023)
        ORDER BY b.date
    """, conn)
    
    # Get historical COTS changes
    historical_changes = pd.read_sql_query("""
        SELECT DISTINCT site_id, 
               MIN(date_1) as first_change,
               MAX(date_2) as last_change,
               COUNT(*) as num_changes
        FROM cots_changes
        GROUP BY site_id
    """, conn)
    
    if recent_surveys.empty:
        print("No surveys found for 2021-2023")
        return
    
    # Prepare features for all surveys
    all_features = prepare_features()
    
    surveys_2021 = recent_surveys[recent_surveys['year'] == 2021]
    surveys_2022 = recent_surveys[recent_surveys['year'] == 2022]
    surveys_2023 = recent_surveys[recent_surveys['year'] == 2023]
    
    print(f"\nFound {len(surveys_2021)} surveys from 2021")
    print(f"Found {len(surveys_2022)} surveys from 2022")
    print(f"Found {len(surveys_2023)} surveys from 2023")
    
    def analyze_year_surveys(year_surveys, year):
        print(f"\n=== Analyzing Risk of COTS Changes for {year} ===\n")
        
        predictions = []
        
        for _, survey in year_surveys.iterrows():
            # Get the survey data
            survey_data = all_features[all_features['survey_id'] == survey['survey_id']].copy()
            
            if not survey_data.empty:
                # Prepare features
                X = survey_data.drop(['survey_id', 'site_id', 'target'], axis=1)
                
                # Get predictions
                prediction = pipeline.predict(X)[0]
                probabilities = pipeline.named_steps['classifier'].predict_proba(
                    pipeline.named_steps['scaler'].transform(X)
                )[0]
                
                # Get key indicators
                indicators = {}
                for feature in KEY_FEATURES:
                    if feature in X.columns:
                        value = X[feature].iloc[0]
                        if value > 0:
                            indicators[feature] = value
                
                # Check historical changes
                site_history = historical_changes[historical_changes['site_id'] == survey['site_id']]
                has_history = not site_history.empty
                
                # Store prediction info
                pred_info = {
                    'site_name': survey['reef_name'],
                    'date': survey['date'],
                    'prediction': prediction,
                    'confidence': max(probabilities) * 100,
                    'probabilities': {
                        class_label: prob * 100 
                        for class_label, prob in zip(pipeline.named_steps['classifier'].classes_, probabilities)
                    },
                    'indicators': indicators,
                    'has_history': has_history
                }
                if has_history:
                    pred_info['history'] = {
                        'first_change': site_history['first_change'].iloc[0],
                        'last_change': site_history['last_change'].iloc[0],
                        'num_changes': site_history['num_changes'].iloc[0]
                    }
                predictions.append(pred_info)
        
        # Sort predictions by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Group by predicted outcome
        increases = [p for p in predictions if p['prediction'] == 'increase']
        decreases = [p for p in predictions if p['prediction'] == 'decrease']
        stable = [p for p in predictions if p['prediction'] == 'none']
        
        # Print all predicted changes
        if increases:
            print(f"\nPredicted COTS Increases ({year}):")
            print("-" * 80)
            for pred in increases:
                print(f"\nSite: {pred['site_name']}")
                print(f"Survey Date: {pred['date']}")
                print(f"Confidence: {pred['confidence']:.1f}%")
                if pred['has_history']:
                    print("\nHistorical COTS Changes:")
                    print(f"First recorded change: {pred['history']['first_change']}")
                    print(f"Most recent change: {pred['history']['last_change']}")
                    print(f"Total number of changes: {pred['history']['num_changes']}")
                print("\nProbabilities:")
                for outcome, prob in pred['probabilities'].items():
                    print(f"{outcome}: {prob:.1f}%")
                print("\nKey Indicators:")
                for indicator, value in pred['indicators'].items():
                    print(f"{indicator}: {value:.2f}")
                print("-" * 80)
        else:
            print(f"\nNo predicted COTS increases for {year}")
        
        if decreases:
            print(f"\nPredicted COTS Decreases ({year}):")
            print("-" * 80)
            for pred in decreases:
                print(f"\nSite: {pred['site_name']}")
                print(f"Survey Date: {pred['date']}")
                print(f"Confidence: {pred['confidence']:.1f}%")
                if pred['has_history']:
                    print("\nHistorical COTS Changes:")
                    print(f"First recorded change: {pred['history']['first_change']}")
                    print(f"Most recent change: {pred['history']['last_change']}")
                    print(f"Total number of changes: {pred['history']['num_changes']}")
                print("\nProbabilities:")
                for outcome, prob in pred['probabilities'].items():
                    print(f"{outcome}: {prob:.1f}%")
                print("\nKey Indicators:")
                for indicator, value in pred['indicators'].items():
                    print(f"{indicator}: {value:.2f}")
                print("-" * 80)
        else:
            print(f"\nNo predicted COTS decreases for {year}")
        
        # Print summary for the year
        print(f"\n=== Summary for {year} ===")
        print(f"Total surveys analyzed: {len(predictions)}")
        print(f"Predicted increases: {len(increases)} ({len([p for p in increases if p['confidence'] > 50])} high confidence)")
        print(f"Predicted decreases: {len(decreases)} ({len([p for p in decreases if p['confidence'] > 50])} high confidence)")
        print(f"Predicted stable: {len(stable)}")
        sites_with_history = len([p for p in predictions if p['has_history']])
        print(f"Sites with previous COTS changes: {sites_with_history}")
    
    # Analyze each year
    analyze_year_surveys(surveys_2021, 2021)
    analyze_year_surveys(surveys_2022, 2022)
    analyze_year_surveys(surveys_2023, 2023)
    
    conn.close()

if __name__ == "__main__":
    predict_recent_changes()
