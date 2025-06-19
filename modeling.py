import os
import joblib
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import config
from data_cleaning import preprocess_data


def get_baseline_models() -> Dict[str, Any]:
    """
    Define baseline models for comparison.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_estimators=100
        ),
        "XGBoost": XGBClassifier(
            eval_metric='logloss', 
            random_state=config.RANDOM_STATE,
            n_estimators=100
        )
    }
    return models


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str) -> str:
    """
    Evaluate a single model and return formatted results.
    """
    y_pred = model.predict(X_test)
    # Classification report
    report = classification_report(y_test, y_pred)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    result = f"\n{'='*50}\n{model_name} Results:\n{'='*50}\n"
    result += f"Classification Report:\n{report}\n"
    result += f"Confusion Matrix:\n{cm}\n"

    return result


def train_and_evaluate_models(models: Dict[str, Any], X_train: pd.DataFrame,
                              y_train: pd.Series, X_test: pd.DataFrame,
                              y_test: pd.Series, save_models: bool = True) -> None:
    """
    Train and evaluate multiple models.

    Args:
        models: Dictionary of model name -> model instance
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        save_models: Whether to save trained models
    """
    results_path = os.path.join(config.RESULTS_DIR, config.RESULT_FILES['model_results'])

    print("Training and evaluating models...")

    with open(results_path, "w") as log_file:
        log_file.write("VODAFONE CUSTOMER CHURN PREDICTION - MODEL COMPARISON\n")
        log_file.write(f"Training samples: {X_train.shape[0]}\n")
        log_file.write(f"Test samples: {X_test.shape[0]}\n")
        log_file.write(f"Features: {X_train.shape[1]}\n\n")

        for name, model in models.items():
            print(f"  Training {name}...")
            # Train model
            model.fit(X_train, y_train)
            # Evaluate model
            result = evaluate_model(model, X_test, y_test, name)
            # Print and log results
            print(result)
            log_file.write(result)
            # Save model if requested
            if save_models:
                model_filename = config.MODEL_FILES.get(
                    name.lower().replace(' ', '_'),
                    f"{name.lower().replace(' ', '_')}_model.pkl"
                )
                model_path = os.path.join(config.RESULTS_DIR, model_filename)
                joblib.dump(model, model_path)
                print(f"{name} model saved to {model_path}")

    print(f"All results saved to {results_path}")


def main():
    """Main execution function."""
    print("VODAFONE CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*55)
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()
    # Get baseline models
    models = get_baseline_models()

    # Train and evaluate models
    train_and_evaluate_models(models, X_train, y_train, X_test, y_test)
    print("Model training and evaluation completed!")
    print(f"Check results in: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
