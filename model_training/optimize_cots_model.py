import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from predict_cots_changes import prepare_features

def optimize_model():
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
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE()),
        ('classifier', RandomForestClassifier())
    ])
    
    # Define parameter grid
    param_grid = {
        'smote__k_neighbors': [3, 5, 7],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 15, 20],
        'classifier__min_samples_split': [2, 4, 6],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Perform grid search with cross-validation
    print("\nPerforming grid search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("\nBest parameters:")
    print(grid_search.best_params_)
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nBest Model Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Save the best model
    joblib.dump(best_model, 'cots_predictor_pipeline_optimized.joblib')
    print("\nOptimized model saved to disk.")
    
    # Compare with different random seeds
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
        best_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(
                random_state=seed,
                k_neighbors=grid_search.best_params_['smote__k_neighbors']
            )),
            ('classifier', RandomForestClassifier(
                random_state=seed,
                n_estimators=grid_search.best_params_['classifier__n_estimators'],
                max_depth=grid_search.best_params_['classifier__max_depth'],
                min_samples_split=grid_search.best_params_['classifier__min_samples_split'],
                class_weight=grid_search.best_params_['classifier__class_weight']
            ))
        ])
        
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)
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
