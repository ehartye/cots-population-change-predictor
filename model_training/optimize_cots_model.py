import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline
import joblib
from predict_cots_changes import prepare_features, KEY_FEATURES

def add_interaction_features(X):
    """Add interaction features between key indicators"""
    # Add interactions between bleaching and damage
    X['bleaching_damage'] = X['has_bleaching'] * X['has_coral_damage']
    
    # Add interactions between predator presence
    X['predator_diversity'] = (X['has_parrotfish'] + X['has_wrasse'] + 
                             (X['total_grouper'] > 0).astype(int))
    
    # Ratio features
    X['clam_to_grouper'] = np.where(X['total_grouper'] > 0, 
                                   X['total_giant_clam'] / (X['total_grouper'] + 1e-6), 
                                   0)
    
    # Threshold-based features
    X['high_bleaching'] = (X['Bleaching (% Of Population)'] > 30).astype(int)
    X['large_predator_presence'] = ((X['Grouper > 60 cm'] > 0) | 
                                  (X['Giant Clam > 50 cm'] > 0)).astype(int)
    
    return X

def optimize_model():
    # Monkey patch the sqlite connection in prepare_features
    import types
    def prepare_features_local():
        conn = sqlite3.connect('reefcheck.db')  # Use local path
        
        # Get all surveys with their COTS change status
        cots_changes = pd.read_sql_query("""
            SELECT DISTINCT survey_id_1 as survey_id, 
                   CASE 
                       WHEN change > 0 THEN 'increase'
                       WHEN change < 0 THEN 'decrease'
                   END as change_type
            FROM cots_changes
        """, conn)
        
        # Rest of the original prepare_features function
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
    
    print("Preparing features...")
    df = prepare_features_local()
    
    # Prepare features and target
    X = df.drop(['survey_id', 'site_id', 'target'], axis=1)
    y = df['target']
    
    # Add interaction features
    X = add_interaction_features(X)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Try both Random Forest and Gradient Boosting
    classifiers = {
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier()
    }
    
    best_score = 0
    best_model = None
    best_params = None
    best_classifier = None
    
    for clf_name, clf in classifiers.items():
        print(f"\nOptimizing {clf_name}...")
        
        # Define the pipeline with feature selection
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_select', SelectFromModel(estimator=RandomForestClassifier(n_estimators=100))),
            ('sampler', SMOTE()),
            ('classifier', clf)
        ])
        
        # Define expanded parameter grid
        if clf_name == 'rf':
            param_grid = {
                'feature_select__threshold': ['mean', '0.5*mean', '1.5*mean'],
                'classifier__n_estimators': [200, 300, 400],
                'classifier__max_depth': [10, 15, 20, None],
                'classifier__min_samples_split': [2, 4, 6],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced', 'balanced_subsample']
            }
            
            # Add SMOTE-specific parameters
            smote_grid = {
                'sampler': [SMOTE()],
                'sampler__k_neighbors': [3, 5, 7]
            }
            param_grid.update(smote_grid)
            
            # Try a separate grid with ADASYN
            adasyn_grid = {
                'sampler': [ADASYN()],
                'sampler__n_neighbors': [3, 5, 7]
            }
            
            # Run grid search for SMOTE first
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            
            # Update best model if current one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_classifier = clf_name
            
            # Update param grid for ADASYN
            param_grid.update(adasyn_grid)
            del param_grid['sampler__k_neighbors']
            
            # Run grid search for ADASYN
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            
            # Update best model if current one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_classifier = clf_name
                
        else:  # Gradient Boosting
            param_grid = {
                'feature_select__threshold': ['mean', '0.5*mean', '1.5*mean'],
                'classifier__n_estimators': [200, 300, 400],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 4, 5],
                'classifier__min_samples_split': [2, 4, 6],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
            
            # Add SMOTE-specific parameters
            smote_grid = {
                'sampler': [SMOTE()],
                'sampler__k_neighbors': [3, 5, 7]
            }
            param_grid.update(smote_grid)
            
            # Run grid search for SMOTE
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            
            # Update best model if current one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_classifier = clf_name
            
            # Try ADASYN
            adasyn_grid = {
                'sampler': [ADASYN()],
                'sampler__n_neighbors': [3, 5, 7]
            }
            param_grid.update(adasyn_grid)
            del param_grid['sampler__k_neighbors']
            
            # Run grid search for ADASYN
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            
            # Update best model if current one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_classifier = clf_name
    
    print(f"\nBest classifier: {best_classifier}")
    print("\nBest parameters:")
    print(best_params)
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    
    print("\nBest Model Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Get selected features
    selected_features_mask = best_model.named_steps['feature_select'].get_support()
    selected_features = X.columns[selected_features_mask].tolist()
    
    print("\nSelected Features:")
    print(selected_features)
    
    # Feature importance for the final classifier
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
    
    # Save the best model
    joblib.dump(best_model, 'cots_predictor_pipeline_optimized.joblib')
    print("\nOptimized model saved to disk.")
    
    # Test model stability
    print("\nTesting model stability with different random seeds...")
    seeds = [42, 123, 456, 789, 101112]
    results = []
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=seed,
            stratify=y
        )
        
        # Create pipeline with best parameters
        test_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_select', SelectFromModel(
                estimator=RandomForestClassifier(n_estimators=100),
                threshold=best_params['feature_select__threshold']
            )),
            ('sampler', best_model.named_steps['sampler']),  # Use the same sampler type as best model
            ('classifier', best_model.named_steps['classifier'])
        ])
        
        test_pipeline.fit(X_train, y_train)
        y_pred = test_pipeline.predict(X_test)
        score = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            'seed': seed,
            'accuracy': score['accuracy'],
            'macro_f1': score['macro avg']['f1-score']
        })
    
    print("\nModel Stability Results:")
    results_df = pd.DataFrame(results)
    print("\nAccuracy across seeds:")
    print(f"Mean: {results_df['accuracy'].mean():.3f}")
    print(f"Std: {results_df['accuracy'].std():.3f}")
    print("\nMacro F1 across seeds:")
    print(f"Mean: {results_df['macro_f1'].mean():.3f}")
    print(f"Std: {results_df['macro_f1'].std():.3f}")
    
    return best_model

if __name__ == "__main__":
    print("Optimizing COTS prediction model...")
    best_model = optimize_model()
