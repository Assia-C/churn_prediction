import pandas as pd


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(f"Data contains {data.shape[0]} rows and {data.shape[1]} columns.")
        print("First 5 rows of the data:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def data_report(data):
    if data is not None:
        print("\n--- Data Summary ---")
        print(data.describe())
        print("\n--- Data Types and Missing Values ---")
        print(data.info())
        print("\n--- Cardinality of Each Column ---")
        print(data.nunique())
        print("\n--- Duplicate Rows ---")
        print(f"Number of duplicate rows: {data.duplicated().sum()}")
        print("\n--- NaN Values Per Column ---")
        print(data.isna().sum())
    else:
        print("No data to analyse.")


def check_balance(data, target_column):
    if target_column in data.columns:
        print(data[target_column].value_counts())
        print("\nClass distribution (percentage):")
        print(data[target_column].value_counts(normalize=True) * 100)
    else:
        print(f"Column '{target_column}' not found in the data.")


if __name__ == "__main__":
    file_path = 'Vodafone_Customer_Churn_Sample_Dataset.csv'
    data = load_data(file_path)
    data_report(data)
    check_balance(data, 'Churn')
