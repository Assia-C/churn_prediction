# Vodafone Customer Churn Prediction

A machine learning solution for predicting customer churn.

## 🎯 Project Overview

This project develops a predictive model to identify customers at risk of churning.

### Key Features
- **Churn Prediction Model**: 78% accuracy using XGBoost
- **Feature Importance Analysis**: SHAP-based interpretability (extendable)
- **Automated Pipeline**: End-to-end ML workflow
- **Performance Monitoring**: Comprehensive model evaluation

## 📊 Model Performance

**XGBoost Results:**
```
Overall Accuracy: 78%

Class Performance:
- Retain (0): Precision 0.86, Recall 0.84, F1 0.85
- Churn (1):  Precision 0.58, Recall 0.61, F1 0.60

Weighted Average: F1-Score 0.78
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Assia-C/vodafone-churn-prediction.git
cd vodafone-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Pipeline

#### Full Pipeline (Recommended)
```bash
python src/pipeline.py
```

#### Quick Run (Skip Tuning and SHAP)
```bash
python src/pipeline.py --quick
```

#### Custom Options
```bash
python src/pipeline.py --skip-tuning    # Skip hyperparameter tuning
python src/pipeline.py --skip-shap      # Skip SHAP analysis
```

### Individual Components

#### Data Preprocessing
```bash
python src/data_cleaning.py
```

#### Model Training
```bash
python src/modeling.py
```

#### Hyperparameter Tuning
```bash
python src/tuning.py
```

#### SHAP Analysis
```bash
python src/shap_analysis.py
```

## 📈 Results Summary

### Model Comparison
| Algorithm | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score |
|-----------|----------|-------------------|----------------|----------|
| XGBoost   | 78%      | 58%               | 61%            | 0.60     |
| Random Forest    | 76%      | 55%               | 58%            | 0.57     |
| Logistic Regression| 74%      | 52%               | 55%            | 0.53     |

## 🔧 Configuration

### Key Settings (`config.py`)

# Model parameters
MODEL_FILES = {
    'xgboost': 'xgboost_model.pkl',
    'best_xgboost': 'best_xgboost_model.pkl'
}

### Customisation
- Modify `config.py` for different data paths
- Adjust model parameters in `modeling.py`

### 📊 Output Files

### Model Artifacts
- `xgboost_model.pkl` - Base XGBoost model (outperforms tuned model)
- `best_xgboost_model.pkl` - Tuned model

## 📝 Next Steps

### Phase 2 Enhancements
- [ ] Improve prediction on minority class
- [ ] Advanced feature engineering
- [ ] Deep learning model exploration
- [ ] Customer segmentation refinement
