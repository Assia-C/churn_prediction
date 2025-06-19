import os
import pandas as pd
import joblib
from typing import Tuple, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data with error handling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Shape: {data.shape}")
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling TotalCharges and removing unnecessary columns.
    """
    data = data.copy()

    # Convert TotalCharges to numeric, handling errors
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Count and remove rows with missing TotalCharges
    missing_count = data['TotalCharges'].isna().sum()
    if missing_count > 0:
        print(f"  Removing {missing_count} rows with missing TotalCharges")
        data = data.dropna(subset=['TotalCharges'])

    # Remove customerID if present (not useful for modeling)
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
        print("  Removed customerID column")

    print(f"Data cleaning completed. Final shape: {data.shape}")
    return data


def encode_target(data: pd.DataFrame, target_column: str, 
                  mapping: dict = None) -> pd.DataFrame:
    """
    Encode target variable.
    """
    if mapping is None:
        mapping = config.TARGET_MAPPING

    data = data.copy()
    data[target_column] = data[target_column].map(mapping)
    print(f"Target variable encoded: {dict(data[target_column].value_counts())}")
    return data


def encode_categorical(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    One-hot encode categorical columns (excluding target).
    """
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    if categorical_cols:
        print(f"  Encoding categorical columns: {categorical_cols}")
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        print(f"Categorical encoding completed. Shape: {data_encoded.shape}")
        return data_encoded
    else:
        print("No categorical columns to encode")
        return data


def prepare_features(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Ensure all feature columns are float type.
    """
    data = data.copy()
    feature_cols = data.columns.difference([target_column])
    data[feature_cols] = data[feature_cols].astype(float)
    print("All features converted to float type")
    return data


def split_data(data: pd.DataFrame, target_column: str, 
               test_size: float = None, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("Data split completed:")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train class distribution: {dict(y_train.value_counts())}")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   scaler_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using RobustScaler fitted on training data.
    """
    if scaler_path is None:
        scaler_path = config.SCALER_PATH

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"Features scaled and scaler saved to {scaler_path}")
    return X_train_scaled, X_test_scaled


def balance_data(X_train: pd.DataFrame, y_train: pd.Series, 
                random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance training data using SMOTE.
    """
    if random_state is None:
        random_state = config.SMOTE_RANDOM_STATE
    print(f"  Class distribution before SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Data balanced with SMOTE: {dict(pd.Series(y_train_balanced).value_counts())}")

    return X_train_balanced, y_train_balanced


def preprocess_data(file_path: str = None, target_column: str = None,
                    for_inference: bool = False) -> Tuple:
    """
    Complete preprocessing pipeline.
    """
    if file_path is None:
        file_path = config.FILE_PATH
    if target_column is None:
        target_column = config.TARGET_COLUMN

    print("Starting preprocessing pipeline...")
    # Load and clean
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    # Encode target and categorical variables
    target_encoded_data = encode_target(cleaned_data, target_column)
    categorical_encoded_data = encode_categorical(target_encoded_data, target_column)
    # Prepare features
    processed_data = prepare_features(categorical_encoded_data, target_column)
    # Split data
    X_train, X_test, y_train, y_test = split_data(processed_data, target_column)
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    if for_inference:
        print("Preprocessing completed (inference mode - no balancing)")
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        # Balance training data
        X_train_balanced, y_train_balanced = balance_data(X_train_scaled, y_train)
        print("Preprocessing completed (training mode - with balancing)")
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
