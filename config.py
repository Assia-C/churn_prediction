import os

# File paths
FILE_PATH = 'Vodafone_Customer_Churn_Sample_Dataset.csv'
SCALER_PATH = 'robustscaler.pkl'
RESULTS_DIR = 'results'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'Churn'
TARGET_MAPPING = {'Yes': 1, 'No': 0}

# SMOTE parameters
SMOTE_RANDOM_STATE = RANDOM_STATE

# Model file names
MODEL_FILES = {
    'logistic_regression': 'logistic_regression_model.pkl',
    'random_forest': 'random_forest_model.pkl',
    'xgboost': 'xgboost_model.pkl',
    'best_xgboost': 'best_xgboost_model.pkl'
}

# Results file names
RESULT_FILES = {
    'model_results': 'model_results.txt',
    'xgb_tuning_results': 'xgb_tuning_results.txt',
    'best_params': 'best_xgb_params.txt'
}

# Hyperparameter grid for XGBoost
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
