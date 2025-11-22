# E-Commerce Fraud Detection (BT4012 Fraud Analytics: Group 15)

A comprehensive machine learning pipeline for detecting fraudulent e-commerce transactions using ensemble methods, featuring advanced hyperparameter tuning, adversarial robustness testing, and explainable AI capabilities.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements a complete fraud detection system for e-commerce transactions with:
- **Class Imbalance Handling**: Only 5.17% of transactions are fraudulent
- **Ensemble Learning**: Combines Neural Networks, LightGBM, and XGBoost
- **Adversarial Testing**: Simulates sophisticated fraud evasion tactics
- **Explainable AI**: Generates human-readable fraud case reports

## ✨ Features

- **Advanced EDA**: Temporal patterns, categorical analysis, Benford's Law validation
- **Multiple ML Approaches**: 
  - Supervised: Logistic Regression, XGBoost, LightGBM, Neural Networks
  - Unsupervised: Isolation Forest
- **Hyperparameter Optimization**: Optuna-based tuning for all models
- **Metric Strategy**: PR-AUC optimization (not ROC-AUC) due to severe class imbalance
- **Ensemble Optimization**: Grid search for optimal model weights and thresholds
- **Adversarial Robustness**: "Chameleon Attack" simulation
- **Human-in-the-Loop**: Automated fraud case report generation

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster neural network training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/briannoelkesuma/BT4012-Group-15-Ecommerce-Fraud.git
cd BT4012-Group-15-Ecommerce-Fraud
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate the virtual environment**
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Data Source

**Dataset**: [Fraudulent E-Commerce Transactions](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions) (Kaggle)

**Features**:
- Transaction details (amount, date, time, quantity)
- Customer information (age, account age, location)
- Payment and device information
- Product categories
- Shipping/billing addresses

**Target Variable**: `is_fraudulent` (0 = legitimate, 1 = fraud)

## 📁 Project Structure

```
BT4012-Group-15-Ecommerce-Fraud/
│
├── venv/                                          # Virtual environment (create this)
├── saved_models/                                  # Trained model checkpoints
├── .gitattributes
├── .gitignore
├── ecommerce_fraud_detection_pipeline.html        # Readability for Results visualization
├── ecommerce_fraud_detection_pipeline.ipynb       # Main Jupyter notebook for full end-to-end fraud analytics pipeline
├── Fraudulent_E-Commerce_Transaction_Data_2.csv   # Dataset
├── README.md                                      # This file: Documentation for the project
└── requirements.txt                               # Python dependencies
```

## 🔬 Methodology

### 1. Exploratory Data Analysis
- **Target Distribution**: Highly imbalanced (94.83% legitimate, 5.17% fraud)
- **Key Findings**:
  - Fraudulent transactions have higher amounts and more outliers
  - New accounts (low `account_age_days`) are high-risk
  - Temporal patterns show weak but measurable signals
  - Categorical features have uniform fraud rates (~5% across all categories)

### 2. Feature Engineering
```python
# Engineered features
- hour_of_day, day_of_week
- is_night_transaction
- shipping_billing_mismatch
- amount_to_account_age_ratio
```

### 3. Model Training Strategy
- **Train/Test Split**: 80/20 with stratification
- **Scaling**: StandardScaler for numerical features
- **Cross-Validation**: 5-Fold Stratified K-Fold
- **Optimization Metric**: **PR-AUC** (not ROC-AUC)
  - Why? PR-AUC ignores True Negatives and focuses on minority class performance
- **Decision Metric**: **F2-Score** (weighs Recall > Precision)
  - Business logic: Missing fraud is costlier than false alarms

### 4. Hyperparameter Tuning
All models use **Optuna** with TPE (Tree-structured Parzen Estimator) sampler:
- 50 trials per model
- Early stopping for Neural Networks
- Weighted loss functions for class imbalance

## 🤖 Models Implemented

| Model | PR-AUC | ROC-AUC | Recall | Precision | Notes |
|-------|--------|---------|--------|-----------|-------|
| **Neural Network** | **0.6097** | 0.8484 | 0.38 | 0.99 | 4-layer MLP with PReLU, BatchNorm, Dropout |
| **LightGBM** | **0.5998** | 0.8485 | 0.69 | 0.27 | Best tree-based model |
| **XGBoost** | **0.5946** | 0.8544 | 0.72 | 0.27 | Highest single-model Recall |
| **Logistic Regression** | 0.5504 | 0.8100 | 0.35 | 0.99 | Baseline with L1 regularization |
| **Isolation Forest** | 0.2436 | 0.7168 | 0.14 | 0.63 | Unsupervised anomaly detection |
| **Ensemble (Final)** | **0.6046** | 0.8491 | **0.57** | **0.42** | Weighted: 30% NN + 70% LGBM |

### Ensemble Configuration
```python
Best Weights: NN=0.3, XGBoost=0.0, LightGBM=0.7
Optimal Threshold: 0.49
F2-Score: 0.6132
```

## 📈 Results

### Top Feature Importances (LightGBM)
1. `transaction_amount`
2. `account_age_days` (negative correlation with fraud)
3. `amount_to_account_age_ratio`
4. `transaction_hour`
5. `shipping_billing_mismatch`

### Adversarial Robustness Test
**Chameleon Attack Simulation**:
- **Attack Strategy**: Halve transaction amounts + shift to daytime
- **Result**: Only 7.04% evasion success rate (10/142 frauds escaped)
- **Conclusion**: ✅ Model is robust to simple feature manipulation

### Explainable AI Output
The system generates automated fraud case reports for human review:
```
🚨 RISK SCORE: 100%
🔍 KEY RISK DRIVERS:
  • HIGH VALUE: $2311.00 significantly higher than average
  • NEW ACCOUNT: Created only 17 days ago
  • SUSPICIOUS BEHAVIOR: Large amount on newer account
```

## 💻 Usage

### Run each cell in the Jupyter notebook which has the end-to-end Ecommerce fraud detection pipeline

### Google Colab Setup (Optional)
Uncomment the Colab Environment Setup section in the notebook in the **FIRST CODE CELL*:

## 🔑 Key Insights

### 1. Metric Selection is Critical
- **Accuracy is meaningless** (94.83% by always predicting "Not Fraud")
- **ROC-AUC is misleading** (inflated by massive True Negative count)
- **PR-AUC is the truth-teller** for imbalanced data

### 2. Feature Engineering Beats Model Complexity
- `amount_to_account_age_ratio` is the strongest fraud signal
- Simple ratios outperform raw features
- Temporal features provide weak but additive signals

### 3. Ensemble > Individual Models
- Neural Network alone: 60.97% PR-AUC + 84.84% ROC-AUC
- Ensemble: 60.46% PR-AUC + 84.91% ROC-AUC + better Recall/Precision balance
- Diversity matters: NN (non-linear) + LGBM (tree-based) complement each other

### 4. Real-World Considerations
- **Threshold tuning** is as important as model selection
- **F2-Score** aligns with business reality (fraud cost > false alarm cost)
- **Adversarial robustness** testing reveals true model quality

## 🚧 Future Improvements

1. **Graph Neural Networks**: Model customer networks/fraud rings
2. **Temporal Models**: LSTMs for sequential transaction patterns
3. **Real-Time Deployment**: MLOps pipeline with A/B testing
4. **Cost-Sensitive Learning**: Incorporate actual fraud loss amounts
5. **Active Learning**: Human feedback loop to improve model
6. **Synthetic Data**: SMOTE/ADASYN for better minority class representation

## 📝 License

This project is for educational purposes as part of NUS BT4012 course work.

## 👥 Contributors

Group 15 - BT4012 Fraud Analytics (Brian, Calvin, Aloysius, Leslie)

**Note**: Ensure GPU availability for optimal Neural Network training speed. The pipeline automatically detects CUDA and falls back to CPU if unavailable.
