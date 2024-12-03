import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from ARCHIVE_NOT_USED.predict_cots_changes_v2 import prepare_features
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib

def custom_f1(y_true, y_pred):
    """Custom F1 score that weights minority classes more heavily"""
    f1_decrease = f1_score(y_true, y_pred, average=None, labels=['decrease'])[0]
    f1_increase = f1_score(y_true, y_pred, average=None, labels=['increase'])[0]
    f1_none = f1_score(y_true, y_pred, average=None, labels=['none'])[0]
    
    # Weight minority classes more heavily
    return (3 * f1_decrease + 3 * f1_increase + f1_none) / 7

def optimize_model():
    print("Preparing features...")
    df = prepare_features()
    
    # Prepare features and target
    X = df.drop(['survey_id', 'site_id', 'target'], axis=1)
    y = df['target']
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'smote__k_neighbors': [3, 4, 5],
        'classifier__n_estimators': [150, 200, 250],
        'classifier__max_depth': [6, 7, 8],
        'classifier__min_samples_split': [3, 4, 5],
        'classifier__min_samples_leaf': [1, 2, 3],
        'classifier__class_weight': [
            {
                'decrease': 2.0,
                'increase': 2.0,
                'none': 1.0
            },
            {
                'decrease': 2.5,
                'increase': 2.0,
                'none': 1.0
            },
            {
                'decrease': 2.0,
                'increase': 2.5,
                'none': 1.0
            }
        ]
    }
    
    # Define scoring metrics
    scoring = {
        'f1_custom': make_scorer(custom_f1),
        'precision_decrease': make_scorer(precision_score, labels=['decrease'], average=None, zero_division=0),
        'recall_decrease': make_scorer(recall_score, labels=['decrease'], average=None, zero_division=0),
        'precision_increase': make_scorer(precision_score, labels=['increase'], average=None, zero_division=0),
        'recall_increase': make_scorer(recall_score, labels=['increase'], average=None, zero_division=0),
        'precision_none': make_scorer(precision_score, labels=['none'], average=None, zero_division=0),
        'recall_none': make_scorer(recall_score, labels=['none'], average=None, zero_division=0)
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='f1_custom',
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    print("\nPerforming grid search...")
    grid_search.fit(X, y)
    
    print("\nBest parameters:")
    print(grid_search.best_params_)
    
    print("\nBest scores for each metric:")
    for metric in scoring.keys():
        best_score_idx = grid_search.cv_results_[f'rank_test_{metric}'].argmin()
        best_score = grid_search.cv_results_[f'mean_test_{metric}'][best_score_idx]
        best_params = {
            k.split('__')[1]: v 
            for k, v in grid_search.cv_results_['params'][best_score_idx].items()
        }
        print(f"\n{metric}:")
        print(f"Score: {best_score:.3f}")
        print("Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    
    # Save best model
    print("\nSaving best model...")
    joblib.dump(grid_search.best_estimator_, 'cots_predictor_pipeline_v2_optimized.joblib')
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    print("Optimizing COTS prediction model...")
    best_model = optimize_model()
