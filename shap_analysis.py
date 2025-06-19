"""SHAP analysis for model interpretability."""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Any
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    print("SHAP not installed. Install with: pip install shap")
    shap = None

import config
from data_cleaning import preprocess_data


def load_model(model_path: Optional[str] = None) -> Any:
    """Load trained model for SHAP analysis."""
    if model_path is None:
        best_path = os.path.join(config.RESULTS_DIR, config.MODEL_FILES['best_xgboost'])
        regular_path = os.path.join(config.RESULTS_DIR, config.MODEL_FILES['xgboost'])
        if os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(regular_path):
            model_path = regular_path
        else:
            raise FileNotFoundError("No XGBoost model found. Run modeling.py or tuning.py first.")

    return joblib.load(model_path)


def run_shap_analysis(model: Any, X_test: pd.DataFrame, sample_size: int = 50) -> tuple:
    """Run complete SHAP analysis and return values and sample data."""
    if shap is None:
        raise ImportError("SHAP is not installed")
    # Sample data if needed
    if len(X_test) > sample_size:
        indices = np.random.choice(X_test.index, size=sample_size, replace=False)
        X_sample = X_test.loc[indices]
    else:
        X_sample = X_test
    # Create explainer
    if hasattr(model, 'get_booster'):  # XGBoost
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X_sample, min(100, len(X_sample)))
        explainer = shap.Explainer(model, background)
    # Generate SHAP values
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):  # Multi-class
        shap_values = shap_values[1]  # Positive class

    return shap_values, X_sample


def create_plots(shap_values: np.ndarray, X_sample: pd.DataFrame,
                 save_plots: bool = True, top_n: int = 10):
    """Create SHAP summary plot"""
    plots_dir = os.path.join(config.RESULTS_DIR, "shap_plot")
    os.makedirs(plots_dir, exist_ok=True)

    # Summary plot only
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
    if save_plots:
        plt.savefig(os.path.join(plots_dir, "shap_summary_plot.png"),
                    bbox_inches='tight', dpi=200)
        print(f"Summary plot saved to {plots_dir}/shap_summary_plot.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    _, X_test, _, _ = preprocess_data()
    model = load_model()
    shap_values, X_sample = run_shap_analysis(model, X_test)
    top_features = create_plots(shap_values, X_sample)
    print("SHAP analysis complete!")
