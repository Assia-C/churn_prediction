from data_cleaning import load_data, clean_data, encode_categorical, scale_numerical, ensure_float_features, split_and_balance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


def evaluate_models(models, X_train, y_train, X_test, y_test, log_path="model_results.txt", print_report=True):
    """
    Train and evaluate multiple models, print and log classification reports.
    """
    with open(log_path, "w") as log_file:
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            result = f"\n{name} Results:\n" + classification_report(y_test, y_pred)
            if print_report:
                print(result)
            log_file.write(result)


def preprocess_for_modeling(file_path, target_column='Churn'):
    """
    Full preprocessing pipeline: load, clean, encode, scale, and split data.
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


if __name__ == "__main__":
    # Set file path and target column
    file_path = 'Vodafone_Customer_Churn_Sample_Dataset.csv'
    target_column = 'Churn'
    X_train_res, X_test, y_train_res, y_test = preprocess_for_modeling(file_path, target_column)
    # Define models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    # Evaluate and log results
    evaluate_models(models, X_train_res, y_train_res, X_test, y_test, print_report=True)
