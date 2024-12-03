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

# Define key features at module level
KEY_FEATURES = [
    'Grouper > 60 cm', 'Grouper 30-40 cm', 'Grouper 40-50 cm', 'Grouper Total',
    'Giant Clam 40-50 cm', 'Giant Clam > 50 cm',
    'Bleaching (% Of Population)', 'Bleaching (% Of Colony)',
    'Coral Damage Other', 'Humphead Wrasse', 'Parrotfish', 'Tripneustes', 'has_bleaching', 'has_coral_damage', 'total_grouper',
    'has_parrotfish', 'has_wrasse', 'total_giant_clam', 'has_tripneustes'
]

def prepare_features():
    conn = sqlite3.connect('../reefcheck.db')
    
    # Get all surveys with their COTS change status
    cots_changes = pd.read_sql_query("""
        SELECT DISTINCT survey_id_1 as survey_id, 
               CASE 
                   WHEN change > 0 THEN 'increase'
                   WHEN change < 0 THEN 'decrease'
               END as change_type
        FROM cots_changes
    """, conn)
    
    # Get all surveys
    all_surveys = pd.read_sql_query("""
        SELECT DISTINCT survey_id, site_id, date
        FROM belt
        ORDER BY date
    """, conn)
    
    # Function to get organism counts for a survey
    def get_survey_features(survey_id):
        query = """
            SELECT 
                organism_code,
                segment_1_count,
                segment_2_count,
                segment_3_count,
                segment_4_count
            FROM belt
            WHERE survey_id = ?
            AND organism_code IS NOT NULL 
            AND organism_code != ''
        """
        df = pd.read_sql_query(query, conn, params=[survey_id])
        
        features = {}
        count_columns = ['segment_1_count', 'segment_2_count', 'segment_3_count', 'segment_4_count']
        
        for _, row in df.iterrows():
            valid_counts = []
            for col in count_columns:
                count = row[col]
                if pd.isna(count) or count == '':
                    continue
                try:
                    count_float = float(count)
                    valid_counts.append(count_float)
                except (ValueError, TypeError):
                    continue
            
            if len(valid_counts) >= 2:  # Only include if at least 2 valid segments
                organism = row['organism_code']
                avg_count = sum(valid_counts) / len(valid_counts)
                features[organism] = avg_count
                
                # Add derived features for key indicators
                if 'Bleaching' in organism:
                    features['has_bleaching'] = 1 if avg_count > 0 else 0
                elif 'Coral Damage' in organism:
                    features['has_coral_damage'] = 1 if avg_count > 0 else 0
                elif organism.startswith('Grouper'):
                    features['total_grouper'] = features.get('total_grouper', 0) + avg_count
                elif organism == 'Parrotfish':
                    features['has_parrotfish'] = 1 if avg_count > 0 else 0
                elif organism == 'Humphead Wrasse':
                    features['has_wrasse'] = 1 if avg_count > 0 else 0
                elif organism.startswith('Giant Clam'):
                    features['total_giant_clam'] = features.get('total_giant_clam', 0) + avg_count
                elif organism == 'Tripneustes':
                    features['has_tripneustes'] = 1 if avg_count > 0 else 0
        
        return features
    
    # Prepare dataset
    dataset = []
    
    for _, survey in all_surveys.iterrows():
        features = get_survey_features(survey['survey_id'])
        
        # Only include surveys with sufficient data
        if len(features) >= 5:  # Require at least 5 features to be present
            row = {'survey_id': survey['survey_id'], 'site_id': survey['site_id']}
            
            # Add features
            for feature in KEY_FEATURES:
                row[feature] = features.get(feature, 0)  # Use 0 if feature not present
            
            # Add target variable
            change_record = cots_changes[cots_changes['survey_id'] == survey['survey_id']]
            if not change_record.empty:
                row['target'] = change_record.iloc[0]['change_type']
            else:
                row['target'] = 'none'
            
            dataset.append(row)
    
    conn.close()
    return pd.DataFrame(dataset)

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
    joblib.dump(pipeline, 'cots_predictor_pipeline.joblib')
    print("\nModel pipeline saved to disk.")
    
    return pipeline

def predict_site(site_id, pipeline=None):
    """Predict COTS change probability for a specific site"""
    if pipeline is None:
        pipeline = joblib.load('cots_predictor_pipeline.joblib')
    
    conn = sqlite3.connect('../reefcheck.db')
    
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
            if feature in X.columns and X[feature].iloc[0] > 0:
                print(f"{feature}: {X[feature].iloc[0]:.2f}")
    
    conn.close()
    return prediction[0], probabilities[0]

if __name__ == "__main__":
    print("Training COTS change prediction model...")
    pipeline = train_model()
    
    # Get a list of sites with sufficient data
    conn = sqlite3.connect('../reefcheck.db')
    sites = pd.read_sql_query("""
        SELECT DISTINCT site_id 
        FROM belt 
        GROUP BY site_id 
        HAVING COUNT(DISTINCT survey_id) >= 2
    """, conn)
    conn.close()
    
    if not sites.empty:
        print("\nExample predictions:")
        # Try to predict for first 3 sites with sufficient data
        for site_id in sites['site_id'].head(3):
            predict_site(site_id, pipeline)
