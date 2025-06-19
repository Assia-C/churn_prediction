import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import config
from data_cleaning import preprocess_data


def get_param_grid(custom_grid: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get hyperparameter grid for XGBoost tuning.
    """
    if custom_grid is not None:
        return custom_grid

    return config.XGB_PARAM_GRID


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                 param_grid: Optional[Dict] = None, scoring: str = 'f1',
                 cv: int = 3, n_jobs: int = -1, verbose: bool = True) -> Any:
    """
    Perform hyperparameter tuning for XGBoost using GridSearchCV.
    """
    if param_grid is None:
        param_grid = get_param_grid()

    if verbose:
        print("Starting XGBoost hyperparameter tuning...")
        print(f"  Parameter grid: {param_grid}")
        print(f"  Scoring metric: {scoring}")
        print(f"  CV folds: {cv}")

    # Create base model
    xgb_base = XGBClassifier(
        eval_metric='logloss', 
        random_state=config.RANDOM_STATE
    )

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1 if verbose else 0
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    if verbose:
        print("Tuning completed!")
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_


def save_best_params(params: Dict[str, Any], filepath: Optional[str] = None) -> None:
    """
    Save best hyperparameters to file.
    """
    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, config.RESULT_FILES['best_params'])

    with open(filepath, 'w') as f:
        f.write("BEST XGBOOST HYPERPARAMETERS\n")
        f.write("="*30 + "\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Best parameters saved to {filepath}")


def evaluate_tuned_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                         filepath: Optional[str] = None) -> None:
    """
    Evaluate the tuned model and save results.
    """
    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, config.RESULT_FILES['xgb_tuning_results'])

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate report
    report = classification_report(y_test, y_pred)

    # Save results
    with open(filepath, "w") as f:
        f.write("TUNED XGBOOST MODEL EVALUATION\n")
        f.write("="*35 + "\n\n")
        f.write("Model parameters:\n")
        for key, value in model.get_params().items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nClassification Report:\n{report}")

    print(f"Model evaluation saved to {filepath}")
    print(f"\nTuned XGBoost Results:\n{report}")


def save_tuned_model(model: Any, filepath: Optional[str] = None) -> None:
    """
    Save the tuned model.
    """
    if filepath is None:
        filepath = os.path.join(config.RESULTS_DIR, config.MODEL_FILES['best_xgboost'])

    joblib.dump(model, filepath)
    print(f"Best XGBoost model saved to {filepath}")


def main():
    """Main execution function."""
    print("VODAFONE CUSTOMER CHURN PREDICTION - XGBOOST TUNING")
    print("="*55)

    # Preprocess data
    print("\n1. Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()

    # Tune XGBoost
    print("\n2. Tuning XGBoost hyperparameters...")
    best_xgb = tune_xgboost(X_train, y_train)

    # Save best parameters
    print("\n3. Saving best parameters...")
    save_best_params(best_xgb.get_params())

    # Train final model with best parameters
    print("\n4. Training final model...")
    best_xgb.fit(X_train, y_train)

    # Save tuned model
    print("\n5. Saving tuned model...")
    save_tuned_model(best_xgb)

    # Evaluate tuned model
    print("\n6. Evaluating tuned model...")
    evaluate_tuned_model(best_xgb, X_test, y_test)
    print("\nXGBoost tuning completed!")
    print(f"Check results in: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
