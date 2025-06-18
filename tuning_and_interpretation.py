from data_cleaning import load_data, clean_data, encode_categorical, scale_numerical, ensure_float_features, split_and_balance
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib


def preprocess_for_tuning(file_path, target_column='Churn'):
    """
    Full preprocessing pipeline for tuning: load, clean, encode, scale, and split data.
    Returns balanced train/test splits.
    """
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    cleaned_data[target_column] = cleaned_data[target_column].map({'Yes': 1, 'No': 0})
    encoded_data = encode_categorical(cleaned_data, target_column=target_column)
    encoded_and_scaled_data = scale_numerical(encoded_data, exclude_cols=[target_column])
    encoded_and_scaled_data = ensure_float_features(encoded_and_scaled_data, exclude_cols=[target_column])
    X_train_res, X_test, y_train_res, y_test = split_and_balance(encoded_and_scaled_data, target_column)
    return X_train_res, X_test, y_train_res, y_test


def tune_xgboost(X_train, y_train, param_grid=None, scoring='f1', cv=3, n_jobs=-1):
    """
    Perform hyperparameter tuning for XGBoost using GridSearchCV.
    Returns the best estimator.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            # 'subsample': [0.7, 0.8, 1.0],
            # 'colsample_bytree': [0.7, 0.8, 1.0],
            # 'scale_pos_weight': [1, 2, 3, 5, 10]
        }
    print("Starting hyperparameter tuning for XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    return grid.best_estimator_


def save_best_params(params, path='best_xgb_params.txt'):
    """
    Save the best hyperparameters to a text file.
    """
    with open(path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Best parameters saved to {path}")


def evaluate_and_save(model, X_test, y_test, log_path="xgb_eval_results_results.txt"):
    """
    Evaluate the model on the test set and save the classification report to a file.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    with open(log_path, "w") as log_file:
        log_file.write(report)
    print("Model evaluation saved to", log_path)


if __name__ == "__main__":
    file_path = 'Vodafone_Customer_Churn_Sample_Dataset.csv'
    target_column = 'Churn'
    X_train_res, X_test, y_train_res, y_test = preprocess_for_tuning(file_path, target_column)
    best_xgb = tune_xgboost(X_train_res, y_train_res)
    save_best_params(best_xgb.get_params())
    best_xgb.fit(X_train_res, y_train_res)
    # Save the best model
    joblib.dump(best_xgb, 'best_xgb_model.pkl')
    print("Best XGBoost model saved as 'best_xgb_model.pkl'")
    evaluate_and_save(best_xgb, X_test, y_test)

