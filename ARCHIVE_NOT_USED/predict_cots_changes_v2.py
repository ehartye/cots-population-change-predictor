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

# Define significant organisms and their control means
SIGNIFICANT_ORGANISMS = {
    'Grouper > 60 cm': 0.05,
    'Grouper Total': 0.76,
    'Giant Clam 40-50 cm': 0.03,
    'Grouper 40-50 cm': 0.12,
    'Grouper 30-40 cm': 0.54,
    'Giant Clam > 50 cm': 0.08,
    'Bleaching (% Of Population)': 20.69,
    'Coral Damage Other': 3.81,
    'Bleaching (% Of Colony)': 81.87,
    'Humphead Wrasse': 0.04,
    'Parrotfish': 3.40,
    'Tripneustes': 0.55
}

def prepare_features():
    conn = sqlite3.connect('reefcheck.db')
    
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
    
    def get_survey_features(survey_id):
        # Build query to only select significant organisms
        significant_organisms = list(SIGNIFICANT_ORGANISMS.keys())
        placeholders = ','.join(['?' for _ in significant_organisms])
        
        query = f"""
            SELECT 
                organism_code,
                segment_1_count,
                segment_2_count,
                segment_3_count,
                segment_4_count
            FROM belt
            WHERE survey_id = ?
            AND organism_code IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=[survey_id] + significant_organisms)
        
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
            
            if len(valid_counts) >= 1:
                organism = row['organism_code']
                total_count = sum(valid_counts)
                features[organism] = total_count
        
        return features
    
    # Prepare dataset
    dataset = []
    
    for _, survey in all_surveys.iterrows():
        features = get_survey_features(survey['survey_id'])
        
        # Only include surveys with sufficient data
        if len(features) >= 3:  # Require at least 3 significant features
            row = {'survey_id': survey['survey_id'], 'site_id': survey['site_id']}
            
            # Initialize all possible features to 0
            for organism in SIGNIFICANT_ORGANISMS.keys():
                row[organism] = 0
            
            # Update with actual values
            row.update(features)
            
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
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=7,               # Balanced depth
            min_samples_split=4,       # Moderate split requirement
            min_samples_leaf=2,        # Moderate leaf requirement
            class_weight={             # Custom class weights
                'decrease': 2.0,       # Higher weight for rare events
                'increase': 2.0,
                'none': 1.0
            },
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
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the pipeline
    joblib.dump(pipeline, 'cots_predictor_pipeline_v2.joblib')
    print("\nModel pipeline saved to disk.")
    
    return pipeline

def predict_site(site_id, pipeline=None):
    """Predict COTS change probability for a specific site"""
    if pipeline is None:
        pipeline = joblib.load('cots_predictor_pipeline_v2.joblib')
    
    conn = sqlite3.connect('reefcheck.db')
    
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
        # Show non-zero features sorted by value
        features_with_values = {col: X[col].iloc[0] for col in X.columns if X[col].iloc[0] > 0}
        sorted_features = dict(sorted(features_with_values.items(), key=lambda x: x[1], reverse=True))
        for feature, value in sorted_features.items():
            control_mean = SIGNIFICANT_ORGANISMS[feature]
            relative_diff = ((value - control_mean) / control_mean * 100) if control_mean > 0 else value * 100
            print(f"{feature}: {value:.2f} ({relative_diff:+.1f}% vs control)")
    
    conn.close()
    return prediction[0], probabilities[0]

if __name__ == "__main__":
    print("Training COTS change prediction model...")
    pipeline = train_model()
    
    # Get a list of sites with sufficient data
    conn = sqlite3.connect('reefcheck.db')
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
