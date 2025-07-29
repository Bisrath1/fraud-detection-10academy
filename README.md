# Fraud Detection Challenge - 10 Academy Week 8 & 9

## Overview
This repository contains the solution for the 10 Academy Week 8 & 9 Fraud Detection Challenge. The goal is to develop a machine learning model to detect fraudulent transactions using two datasets: `Fraud_Data.csv` and `creditcard.csv`. The project is structured into three tasks:
1. **Data Preprocessing**: Cleaned datasets, merged geolocation data, performed exploratory data analysis (EDA), and engineered features.
2. **Model Building**: Trained and evaluated Logistic Regression and XGBoost models, with XGBoost selected for superior performance.
3. **Model Explainability**: Used SHAP to interpret model predictions, identifying key fraud drivers and providing business recommendations.

## Repository Structure
```
fraud-detection-10academy/
├── data/
│   ├── processed/              # Processed datasets (linked externally due to size)
│   └── raw/                    # Raw datasets (linked externally due to size)
├── notebooks/                  # Jupyter notebooks for EDA
├── output/                     # Output plots and results
│   ├── purchase_value_dist.png
│   ├── fraud_by_browser.png
│   ├── cm_logreg_fraud.png
│   ├── cm_xgboost_fraud.png
│   ├── cm_logreg_creditcard.png
│   ├── cm_xgboost_creditcard.png
│   ├── shap_summary_fraud.png
│   ├── shap_force_fraud.png
│   ├── shap_summary_creditcard.png
│   ├── shap_force_creditcard.png
│   ├── model_metrics.csv
│   ├── model_selection.txt
│   └── shap_interpretation.txt
├── src/
│   ├── task1_preprocessing.py   # Data cleaning and feature engineering
│   ├── task2_modeling.py       # Model training and evaluation
│   ├── task3_explainability.py # SHAP analysis for model interpretability
│   └── preprocess_creditcard.py # Preprocessing for creditcard.csv
├── .gitignore                  # Excludes venv/, large datasets, etc.
├── README.md                   # This file
├── report.md                   # Final project report
└── requirements.txt            # Python dependencies
```

## Datasets
Due to GitHub’s file size limits (50 MB recommended, 100 MB hard limit), large datasets are hosted on Google Drive:
- `Fraud_Data.csv`: [Google Drive Link]
- `creditcard.csv`: [Google Drive Link]
- `processed_fraud_data.csv`: [Google Drive Link]
- `fraud_data_with_country.csv`: [Google Drive Link]
- `processed_creditcard_data.csv`: [Google Drive Link]

**Note**: `fraud_features.csv` was initially used as a proxy for `creditcard.csv` due to a dataset mismatch. After confirming the availability of `creditcard.csv`, it was preprocessed and used for Tasks 2 and 3.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bisrath1/fraud-detection-10academy.git
   cd fraud-detection-10academy
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate     # On Linux/Mac
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Datasets**:
   - Download the datasets from the Google Drive links above.
   - Place them in `data/raw/` (`Fraud_Data.csv`, `creditcard.csv`) and `data/processed/` (`fraud_data_with_country.csv`, `processed_creditcard_data.csv`).

## Running the Code
1. **Task 1: Data Preprocessing**:
   ```bash
   python src/task1_preprocessing.py
   python src/preprocess_creditcard.py
   ```
   - Outputs: `data/processed/fraud_data_with_country.csv`, `data/processed/processed_creditcard_data.csv`, `output/purchase_value_dist.png`, `output/fraud_by_browser.png`.

2. **Task 2: Model Building**:
   ```bash
   python src/task2_modeling.py
   ```
   - Outputs: `output/cm_*.png`, `output/model_metrics.csv`, `output/model_selection.txt`.

3. **Task 3: Model Explainability**:
   ```bash
   python src/task3_explainability.py
   ```
   - Outputs: `output/shap_summary_fraud.png`, `output/shap_force_fraud.png`, `output/shap_summary_creditcard.png`, `output/shap_force_creditcard.png`, `output/shap_interpretation.txt`.

## Key Findings
- **Task 1**: Engineered features like `signup_to_purchase_sec`, `user_txn_count`, and `is_country_top10` for `Fraud_Data.csv`. Scaled `Time`, `Amount`, and `V1-V28` for `creditcard.csv`. EDA revealed patterns in purchase values and browser usage.
- **Task 2**: XGBoost outperformed Logistic Regression with AUC-PR=0.604 and F1-Score=0.679 for `Fraud_Data.csv`, and AUC-PR=0.808 and F1-Score=0.760 for `creditcard.csv`.
- **Task 3**: SHAP analysis identified key fraud drivers:
  - `Fraud_Data.csv`: `signup_to_purchase_sec`, `purchase_value`, `user_txn_count`, `is_country_top10`.
  - `creditcard.csv`: `V1-V28` (PCA features), `Amount`, `Time`.
  - Recommendations: Monitor short signup-to-purchase times, extreme purchase values, high transaction counts, high-risk geolocations (`Fraud_Data`), and unusual `Amount`/`Time` patterns (`creditcard`).

## Submission Notes
- **Repository**: `https://github.com/Bisrath1/fraud-detection-10academy`
- **Dataset Handling**: Large datasets were removed from the repository due to GitHub size limits and are linked in Google Drive. The `report.md` details the dataset mismatch issue (initial use of `fraud_features.csv`) and its resolution with `creditcard.csv`.
- **Environment**: Tested on Windows with Python 3.11.9. Dependencies are listed in `requirements.txt`.

## Requirements
See `requirements.txt` for the full list. Key dependencies include:
- `pandas>=2.2.2`
- `numpy>=1.26.4`
- `scikit-learn==1.5.1`
- `imbalanced-learn==0.12.3`
- `xgboost>=2.1.1`
- `matplotlib>=3.9.2`
- `seaborn>=0.13.2`
- `shap==0.46.0`

