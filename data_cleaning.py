import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """Load CSV data"""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(data):
    """
    Convert TotalCharges to float and drop rows with missing values.
    """
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna(subset=['TotalCharges']).copy()
    data['TotalCharges'] = data['TotalCharges'].astype(float)
    print("Data cleaning completed.")
    return data


def encode_categorical(data, target_column):
    """
    One-hot encode categorical columns, excluding the target column.
    """
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return data_encoded


def scale_numerical(data, exclude_cols=None):
    """
    Scale numerical columns (excluding those in exclude_cols) using RobustScaler.
    """
    if exclude_cols is None:
        exclude_cols = []
    numerical_cols = data.select_dtypes(include=['float64', 'int64', 'bool']).columns.difference(exclude_cols)
    scaler = RobustScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data


def ensure_float_features(data, exclude_cols=None):
    """
    Ensure all columns (except those in exclude_cols) are float type.
    """
    if exclude_cols is None:
        exclude_cols = []
    cols_to_convert = data.columns.difference(exclude_cols)
    data[cols_to_convert] = data[cols_to_convert].astype(float)
    return data


def split_and_balance(data_encoded, target_column, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and balance the training set using SMOTE.
    """
    X = data_encoded.drop(columns=[target_column])
    y = data_encoded[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print("Training class distribution before SMOTE:")
    print(pd.Series(y_train).value_counts())
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE, training class distribution:")
    print(pd.Series(y_train_res).value_counts())
    return X_train_res, X_test, y_train_res, y_test
