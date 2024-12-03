import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def prepare_features(use_derived=True):
    conn = sqlite3.connect('reefcheck.db')
    
    # Get all COTS change events
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
            
            if len(valid_counts) >= 2:
                organism = row['organism_code']
                avg_count = sum(valid_counts) / len(valid_counts)
                features[organism] = avg_count
                
                if use_derived:
                    # Add derived features
                    if organism == 'COTS':
                        features['COTS_presence'] = 1 if avg_count > 0 else 0
                    elif 'Bleaching' in organism:
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
    
    # Basic features that always exist
    base_features = [
        'Grouper > 60 cm', 'Grouper 30-40 cm', 'Grouper 40-50 cm', 'Grouper Total',
        'Giant Clam 40-50 cm', 'Giant Clam > 50 cm',
        'Bleaching (% Of Population)', 'Bleaching (% Of Colony)',
        'Coral Damage Other', 'Humphead Wrasse', 'Parrotfish', 'Tripneustes', 'COTS'
    ]
    
    # Derived features that are optional
    derived_features = [
        'COTS_presence', 'has_bleaching', 'has_coral_damage', 'total_grouper',
        'has_parrotfish', 'has_wrasse', 'total_giant_clam', 'has_tripneustes'
    ]
    
    key_features = base_features + (derived_features if use_derived else [])
    
    dataset = []
    for _, survey in all_surveys.iterrows():
        features = get_survey_features(survey['survey_id'])
        
        if len(features) >= 5:
            row = {'survey_id': survey['survey_id'], 'site_id': survey['site_id']}
            
            for feature in key_features:
                row[feature] = features.get(feature, 0)
            
            change_record = cots_changes[cots_changes['survey_id'] == survey['survey_id']]
            if not change_record.empty:
                row['target'] = change_record.iloc[0]['change_type']
            else:
                row['target'] = 'none'
            
            dataset.append(row)
    
    conn.close()
    return pd.DataFrame(dataset)

def compare_models():
    print("=== Testing Model with Only Base Features ===")
    df_base = prepare_features(use_derived=False)
    X_base = df_base.drop(['survey_id', 'site_id', 'target'], axis=1)
    y_base = df_base['target']
    
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    
    pipeline_base = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline_base.fit(X_train_base, y_train_base)
    y_pred_base = pipeline_base.predict(X_test_base)
    
    print("\nBase Features Only Performance:")
    print(classification_report(y_test_base, y_pred_base, zero_division=0))
    
    feature_importance_base = pd.DataFrame({
        'feature': X_base.columns,
        'importance': pipeline_base.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Base Features:")
    print(feature_importance_base.head(10))
    
    print("\n=== Testing Model with All Features ===")
    df_all = prepare_features(use_derived=True)
    X_all = df_all.drop(['survey_id', 'site_id', 'target'], axis=1)
    y_all = df_all['target']
    
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    pipeline_all = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline_all.fit(X_train_all, y_train_all)
    y_pred_all = pipeline_all.predict(X_test_all)
    
    print("\nAll Features Performance:")
    print(classification_report(y_test_all, y_pred_all, zero_division=0))
    
    feature_importance_all = pd.DataFrame({
        'feature': X_all.columns,
        'importance': pipeline_all.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Features Including Derived:")
    print(feature_importance_all.head(10))

if __name__ == "__main__":
    compare_models()
