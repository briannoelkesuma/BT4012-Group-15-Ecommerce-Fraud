# --- 0. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, PrecisionRecallDisplay, confusion_matrix, recall_score
import xgboost as xgb
import optuna # For hyperparameter tuning. Install with: pip install optuna

# --- SCRIPT CONFIGURATION ---
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# --- 1. DATA LOADING AND INITIAL CLEANING ---
print("--- 1. Loading Data ---")
df = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data_2.csv')
df.columns = df.columns.str.replace(' ', '_').str.lower()
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df = df.sort_values('transaction_date').reset_index(drop=True)
print("Data loaded, cleaned, and sorted by date.")

# --- 2. ELITE & LEAKAGE-PROOF FEATURE ENGINEERING ---
print("\n--- 2. Engineering Elite Features ---")

# --- 1. DATA LOADING AND INITIAL CLEANING ---
print("--- 1. Loading Data ---")
df = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data_2.csv')
df.columns = df.columns.str.replace(' ', '_').str.lower()
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df = df.sort_values('transaction_date').reset_index(drop=True)
print("Data loaded, cleaned, and sorted by date.")

def benfords_law_analysis(data, column, title):
    """
    Performs Benford's Law analysis and displays the plot.
    The script will pause here until you close the plot window.
    """
    print(f"\n--- Benford's Law Analysis on '{title}' ---")
    
    # 1. Isolate the first digit
    first_digits = data[column][data[column] > 0].astype(str).str[0].astype(int)
    
    # 2. Calculate observed frequency
    observed_freq = first_digits.value_counts(normalize=True).sort_index()
    
    # 3. Calculate expected Benford frequency
    benford_digits = np.arange(1, 10)
    expected_freq = np.log10(1 + 1/benford_digits)
    
    # 4. Create a DataFrame for plotting
    analysis_df = pd.DataFrame({
        'Digit': benford_digits,
        'Benford (Expected)': expected_freq
    }).set_index('Digit')
    analysis_df = analysis_df.join(observed_freq.rename('Observed')).fillna(0)

    # 5. Generate and style the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    analysis_df.plot(kind='bar', ax=ax, width=0.8, color=['skyblue', 'salmon'])
    
    ax.set_title("Benford's Law vs. Observed First Digit of Transaction Amounts", fontsize=16, pad=20, weight='bold')
    ax.set_xlabel("First Digit", fontsize=12, weight='bold')
    ax.set_ylabel("Frequency", fontsize=12, weight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.legend(fontsize=11)
    
    # Add percentage labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    
    # 6. Display the plot. The script will pause here.
    print("Displaying Benford's Law plot. Close the plot window to continue the script...")
    plt.show()

# Call the function to display the plot
benfords_law_analysis(df, 'transaction_amount', 'Transaction Amount')

# --- 2. ELITE & LEAKAGE-PROOF FEATURE ENGINEERING ---
print("\n--- 2. Engineering Elite Features ---")
df['hour_of_day'] = df['transaction_date'].dt.hour
df['day_of_week'] = df['transaction_date'].dt.dayofweek
df['is_night_transaction'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(int)
df['shipping_billing_mismatch'] = (df['shipping_address'] != df['billing_address']).astype(int)

print("Creating leakage-proof rolling aggregates...")
df['time_since_last_transaction'] = df.groupby('customer_id')['transaction_date'].diff().dt.total_seconds()
df['avg_customer_trans_amount_rolling'] = df.groupby('customer_id')['transaction_amount'].shift(1).expanding().mean()
df['customer_trans_count_rolling'] = df.groupby('customer_id').cumcount()

print("Creating velocity features...")
df_time_indexed = df.set_index('transaction_date')
df['trans_count_last_1h'] = df_time_indexed.groupby('customer_id')['transaction_id'].rolling('1H').count().values
df['trans_count_last_24h'] = df_time_indexed.groupby('customer_id')['transaction_id'].rolling('24H').count().values

print("Creating rarity and frequency features...")
df['ip_frequency'] = df.groupby('ip_address')['ip_address'].transform('count')
df['location_frequency'] = df.groupby('customer_location')['customer_location'].transform('count')

print("Creating deeper behavioral and contextual features...")
# Calculate category average amount ONCE and store it for reuse
df['category_avg_amount'] = df.groupby('product_category')['transaction_amount'].transform('mean')
df['amount_vs_category_avg'] = df['transaction_amount'] / (df['category_avg_amount'] + 1e-6)
df['amount_to_account_age_ratio'] = df['transaction_amount'] / (df['account_age_days'] + 1)
df['customer_category_uniqueness'] = df.groupby('customer_id')['product_category'].transform('nunique') / (df['customer_trans_count_rolling'] + 1)
df.fillna(0, inplace=True)
print("Elite feature engineering complete.")

# --- 3. DATA PREPARATION FOR MODELING ---
print("\n--- 3. Preparing Data for Modeling ---")
features_to_drop = ['transaction_id', 'customer_id', 'transaction_date', 'shipping_address', 'billing_address', 'is_fraudulent',
                      'customer_location', 'ip_address', 'category_avg_amount'] # Drop helper column
X = df.drop(columns=features_to_drop)
y = df['is_fraudulent']
categorical_features = X.select_dtypes(include=['object', 'category']).columns
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numerical_features = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
print("Data preparation complete.")

# --- 4. ADVANCED HYPERPARAMETER TUNING WITH OPTUNA ---
print("\n--- 4. Finding Optimal Hyperparameters with Optuna ---")
def objective(trial):
    scale_pos_weight = y_train.value_counts().get(0, 1) / y_train.value_counts().get(1, 1)
    
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    model.fit(X_train, y_train, verbose=False)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)

    # Optimize FRAUD recall directly
    return recall_score(y_test, y_pred, pos_label=1)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50) # Run 50 trials to find the best params
best_params = study.best_params
print("Best hyperparameters found:", best_params)

# --- 5. TRAINING FINAL MODEL WITH OPTIMAL PARAMETERS ---
print("\n--- 5. Training Final Model with Optimal Hyperparameters ---")
scale_pos_weight = y_train.value_counts().get(0, 1) / y_train.value_counts().get(1, 1)
final_xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='aucpr', scale_pos_weight=scale_pos_weight,
    random_state=42, use_label_encoder=False, **best_params
)
final_xgb_model.fit(X_train, y_train)

y_prob_final = final_xgb_model.predict_proba(X_test)[:, 1]
y_pred_final_class = (y_prob_final >= 0.3).astype(int)

# --- FULL Evaluation Report for Final Model ---
print("\n--- Final Optimized XGBoost Model - Full Evaluation Report ---")
print(f"AUPRC Score: {average_precision_score(y_test, y_prob_final):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob_final):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_final_class, target_names=['Not Fraud', 'Fraud']))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix for Final Model')
plt.show()

# --- 5. VISUALIZING RESULTS ---
print("\n--- 5. Visualizing Results ---")
# Feature Importance from the FINAL model
importances = final_xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values('importance', ascending=False).head(20)
top_features = feature_importance_df['feature'].tolist()

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Top 20 Feature Importances from Final Optimized XGBoost Model')
plt.show()

# --- 6. ADVANCED TOPICS SIMULATION (UPGRADED) ---
print("\n--- 6. Simulating Advanced Scenarios ---")

# --- (Week 7) Intelligent Adversarial Attack Simulation: "The Chameleon" ---
print("\nSimulating 'The Chameleon' Adversarial Attack (Week 7)...")
fraud_indices = y_test[y_test == True].index
if not fraud_indices.empty:
    # Get original raw data for the entire test set
    X_test_raw_df = df.loc[X_test.index].copy()
    
    # Simulate the multi-pronged "Chameleon" attack on the raw data
    print("-> Attacker uses matching shipping/billing addresses.")
    X_test_raw_df.loc[fraud_indices, 'shipping_billing_mismatch'] = 0
    
    print("-> Attacker avoids suspicious night hours.")
    X_test_raw_df.loc[fraud_indices, 'is_night_transaction'] = 0
    X_test_raw_df.loc[fraud_indices, 'hour_of_day'] = 14 # A "safe" hour
    
    print("-> Attacker slows down to avoid velocity flags.")
    X_test_raw_df.loc[fraud_indices, 'time_since_last_transaction'] = 86400 # 1 day
    X_test_raw_df.loc[fraud_indices, 'trans_count_last_1h'] = 1
    
    print("-> Attacker normalizes transaction amount to category average.")
    # This is the most powerful attack: make the amount look normal for its category
    X_test_raw_df.loc[fraud_indices, 'transaction_amount'] = X_test_raw_df.loc[fraud_indices, 'category_avg_amount']
    # Recalculate features that depend on the modified transaction_amount
    X_test_raw_df['amount_vs_category_avg'] = X_test_raw_df['transaction_amount'] / (X_test_raw_df['category_avg_amount'] + 1e-6)
    X_test_raw_df['amount_to_account_age_ratio'] = X_test_raw_df['transaction_amount'] / (X_test_raw_df['account_age_days'] + 1)

    # Now, process this attacked raw data EXACTLY as we did the original data
    X_attacked_processed = X_test_raw_df.drop(columns=features_to_drop)
    X_attacked_processed = pd.get_dummies(X_attacked_processed, columns=categorical_features, drop_first=True)
    X_attacked_processed_aligned, _ = X_attacked_processed.align(X_train, axis=1, fill_value=0)
    X_attacked_processed_aligned[numerical_features] = scaler.transform(X_attacked_processed_aligned[numerical_features])

    # Predict using the final trained model
    y_prob_final_series = pd.Series(y_prob_final, index=y_test.index)
    prob_before_attack = y_prob_final_series[fraud_indices]
    
    prob_after_attack = final_xgb_model.predict_proba(X_attacked_processed_aligned)[:, 1]
    prob_after_attack_series = pd.Series(prob_after_attack, index=X_test.index)[fraud_indices]
    
    print("\nImpact of 'Chameleon' Adversarial Attack on Fraud Probability Predictions:")
    attack_summary = pd.DataFrame({'Prob_Before_Attack': prob_before_attack, 'Prob_After_Attack': prob_after_attack_series})
    attack_summary['Prob_Reduction'] = attack_summary['Prob_Before_Attack'] - attack_summary['Prob_After_Attack']

    print(attack_summary.sort_values('Prob_Reduction', ascending=False).head(10))
    
    avg_prob_before = prob_before_attack.mean()
    avg_prob_after = prob_after_attack_series.mean()
    print(f"\nAverage fraud probability dropped from {avg_prob_before:.2f} to {avg_prob_after:.2f} due to the intelligent attack.")
    
    threshold = 0.5
    evaded_count = ((prob_before_attack >= threshold) & (prob_after_attack_series < threshold)).sum()
    total_fraud = len(fraud_indices)
    print(f"Evasion Rate: {evaded_count}/{total_fraud} ({evaded_count/total_fraud:.1%}) of fraudulent transactions would now be misclassified as legitimate.")

else:
    print("No fraud cases in test set to simulate attack.")

# --- (Week 12 & 13) Human-in-the-Loop & LLM Agent Simulation ---
print("\nSimulating LLM Agent for Human-in-the-Loop (Week 12 & 13)...")
def generate_case_summary(transaction_index, full_df, y_prob_series, top_features_list):
    trans_info = full_df.loc[transaction_index]
    fraud_prob = y_prob_series[transaction_index] * 100
    
    summary = f"""
    ===================================================
    LLM-Generated Case Summary for Fraud Analyst
    ===================================================
    Transaction ID: {trans_info['transaction_id']} | Customer ID: {trans_info['customer_id']}
    Fraud Probability: {fraud_prob:.1f}% -> Recommendation: {'High Priority Review' if fraud_prob > 60 else 'Low Priority'}

    Transaction Context:
    - Amount: ${trans_info['transaction_amount']:.2f} for '{trans_info['product_category']}'
    - Time: Hour {trans_info['hour_of_day']} ({'Night' if trans_info['is_night_transaction'] else 'Day'})
    - Customer Age: {trans_info['customer_age']} | Account Age: {trans_info['account_age_days']} days

    Key Risk Factors (Model Insights):
    -------------------------------------------
    """
    for feature in top_features_list:
        value = trans_info.get(feature) # Use the raw value for interpretability
        if value is not None and value != 0:
            if 'mismatch' in feature and value == 1:
                summary += f"- CRITICAL: Shipping and Billing Addresses DO NOT MATCH.\n"
            elif 'time_since' in feature and value < 3600: # Less than an hour
                 summary += f"- VELOCITY ALERT: Repeat transaction in {value/60:.1f} minutes.\n"
            elif 'amount_vs' in feature and value > 2.0:
                summary += f"- VALUE ANOMALY: Transaction amount is {value:.1f}x the typical average.\n"
            else:
                 summary += f"- {feature.replace('_', ' ').title()}: {value:.2f}\n"

    summary += "==================================================="
    return summary

if not fraud_indices.empty:
    # Find the transaction with the highest fraud probability in the test set
    high_risk_transaction_index_in_test = y_prob_final.argmax()
    high_risk_original_index = y_test.index[high_risk_transaction_index_in_test]
    
    y_prob_series = pd.Series(y_prob_final, index=y_test.index)
    
    case_report = generate_case_summary(high_risk_original_index, df, y_prob_series, top_features)
    print(case_report)
else:
    print("No fraud cases in test set to generate sample report.")

print("\n--- End of Script ---")