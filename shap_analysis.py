import joblib
import pandas as pd
from data_cleaning import load_data, clean_data, encode_categorical, scale_numerical, ensure_float_features, split_and_balance
import shap


def preprocess_for_shap(file_path, target_column='Churn'):
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    cleaned_data[target_column] = cleaned_data[target_column].map({'Yes': 1, 'No': 0})
    encoded_data = encode_categorical(cleaned_data, target_column=target_column)
    encoded_and_scaled_data = scale_numerical(encoded_data, exclude_cols=[target_column])
    encoded_and_scaled_data = ensure_float_features(encoded_and_scaled_data, exclude_cols=[target_column])
    _, X_test, _, y_test = split_and_balance(encoded_and_scaled_data, target_column)
    return X_test, y_test


if __name__ == "__main__":
    file_path = 'Vodafone_Customer_Churn_Sample_Dataset.csv'
    target_column = 'Churn'
    # Load model and test data
    best_xgb = joblib.load("best_xgb_model.pkl")
    X_test, y_test = preprocess_for_shap(file_path, target_column)
    # Run SHAP
    sample_X = X_test.iloc[:10]
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(sample_X, approximate=True)
    shap.summary_plot(shap_values, sample_X)