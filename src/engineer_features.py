import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Common utils
RANDOM_SEED = 42

def engineer_features(raw_df: pd.DataFrame, use_linear_model: bool = False, use_smote: bool = False) -> tuple:
    """
    Transforms the raw e-commerce transaction data into a model-ready feature set
    and returns a standard train/test split.

    This function performs the following steps:
    1.  Fills missing values.
    2.  Handles dtypes conversion.
    3.  Engineers simple and advanced behavioral features.
    4.  Performs one-hot encoding.
    5.  Selects final features and drops raw columns.
    6.  **ALWAYS** calls the splitting function to return 4 arrays.
    
    Args:
        raw_df (pd.DataFrame): The raw DataFrame loaded from the CSV.
        use_linear_model (bool): 
            - If False: Keeps 'Transaction Amount' (for trees).
            - If True: Keeps 'log_transaction_amount' (for linear models).
        use_smote (bool):
            - If False: Returns a standard train/test split.
            - If True: Returns a train/test split where the training
                       data (X_train, y_train) has been resampled using SMOTE.

    Returns:
        tuple: A tuple containing four DataFrames/Series:
               (X_train, X_test, y_train, y_test)
               (or X_train_resampled, X_test, y_train_resampled, y_test if use_smote=True)
    """
    # Work on a copy to avoid modifying the original DataFrame
    df = raw_df.copy()

    # --- Column Definitions ---
    numeric_cols = [
        'Transaction Amount',
        'Quantity',
        'Customer Age',
        'Account Age Days',
        'Transaction Hour'
    ]
    categorical_cols_to_impute = [
        'Transaction ID',
        'Customer ID',
        'Payment Method',
        'Product Category',
        'Customer Location',
        'Device Used',
        'IP Address',
        'Shipping Address',
        'Billing Address'
    ]
    
    # --- Fill Missing Values (IMPUTATION MUST OCCUR BEFORE CATEGORICAL CONVERSION) ---
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # FIX: Force to 'object' (string) type and then Impute NaNs with 'Unknown'
    for col in categorical_cols_to_impute:
        df[col] = df[col].astype('object')
        df[col] = df[col].fillna('Unknown')

    # --- Convert dtypes and create temporal features from date ---
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    
    df['Payment Method'] = df['Payment Method'].astype('category')
    df['Product Category'] = df['Product Category'].astype('category')
    df['Device Used'] = df['Device Used'].astype('category')

    # --- Sorting ---
    df = df.sort_values(by=['Customer ID', 'Transaction Date'])

    # --- Simple Feature Creation ---
    df['log_transaction_amount'] = np.log1p(df['Transaction Amount'])
    df['address_mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)
    df['is_new_account'] = (df['Account Age Days'] < 30).astype(int)

    # --- Temporal Features ---
    df['Transaction Hour'] = df['Transaction Date'].dt.hour
    df['Transaction Weekday'] = df['Transaction Date'].dt.weekday  # 0=Mon, 6=Sun
    df['is_weekend'] = df['Transaction Weekday'].isin([5,6]).astype(int)
    df['Transaction Day'] = df['Transaction Date'].dt.day
    df['Transaction Month'] = df['Transaction Date'].dt.month
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['Hour_Bin'] = pd.cut(df['Transaction Hour'], bins=bins, labels=labels, right=False).astype('category')

    # --- Advanced Behavioral Feature Engineering ---
    df['customer_avg_spend_before_tx'] = df.groupby('Customer ID')['Transaction Amount'] \
                                            .expanding(min_periods=1).mean() \
                                            .reset_index(level=0, drop=True) \
                                            .shift(1)
    df['customer_avg_spend_before_tx'].fillna(0, inplace=True)
    df['amount_deviation'] = (df['Transaction Amount'] - df['customer_avg_spend_before_tx']) / (df['customer_avg_spend_before_tx'] + 1)

    # --- Categorical Variable Encoding ---
    df = pd.get_dummies(df, columns=['Payment Method', 'Product Category', 'Device Used', 'Hour_Bin'], drop_first=True)

    # --- Final Feature Selection and Cleanup ---
    columns_to_drop = [
        'Transaction ID', 
        'Customer ID', 
        'Transaction Date', 
        'Customer Location', 
        'IP Address', 
        'Shipping Address', 
        'Billing Address',
        'customer_avg_spend_before_tx'
    ]
    
    if use_linear_model:
        print("Strategy: Using LOG-TRANSFORMED amount. Dropping original 'Transaction Amount'.")
        columns_to_drop.append('Transaction Amount')
    else:
        print("Strategy: Using ORIGINAL amount. Dropping 'log_transaction_amount'.")
        columns_to_drop.append('log_transaction_amount')

    df.drop(columns=columns_to_drop, inplace=True)
    
    print("Feature engineering complete.")

    # --- MODIFIED LOGIC ---
    # Always call the splitting and balancing function.
    # This function now handles the logic for 'use_smote'
    # and will always return 4 arrays.
    return split_and_handle_imbalance(
        df, 
        use_smote=use_smote, 
        random_state=RANDOM_SEED
    )

def split_and_handle_imbalance(df, use_smote: bool, target_col='Is Fraudulent', test_size=0.2, random_state=RANDOM_SEED):
    """
    Splits the fully engineered data into train and test sets.
    Optionally applies SMOTE to the training set.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split (stratify to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if use_smote:
        # Apply SMOTE only to the training data
        print("Applying SMOTE to the training set...")
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print("\nBefore SMOTE (Training Set):")
        print(y_train.value_counts())
        print("\nAfter SMOTE (Training Set):")
        print(pd.Series(y_train_resampled).value_counts())
        
        return X_train_resampled, X_test, y_train_resampled, y_test
    
    else:
        # Return the original, non-resampled split
        print("Skipping SMOTE. Returning standard train/test split.")
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("Loading raw data...")
    try:
        raw_data = pd.read_csv('../Fraudulent_E-Commerce_Transaction_Data_2.csv')
    except FileNotFoundError:
        print("Error: 'Fraudulent_E-Commerce_Transaction_Data_2.csv' not found.")
        print("Please make sure the data file is in the correct path relative to the script.")
        exit()

    print("\n" + "="*60)
    # --- DEMONSTRATION 1: For a Non Linear Model (e.g., Tree-Based Model) ---
    print("Generating data for Non Linear Models (No SMOTE)...")
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = engineer_features(
        raw_data, 
        use_linear_model=False, 
        use_smote=False
    )
    print("\n--- Final shapes for Tree Model ---")
    print(f"X_train: {X_train_tree.shape}, y_train: {y_train_tree.shape}")
    print(f"X_test: {X_test_tree.shape}, y_test: {y_test_tree.shape}")


    print("\n" + "="*60)
    # --- DEMONSTRATION 2: For a Linear Model (e.g., Logistic Regression) ---
    print("Generating data for Linear Models (With SMOTE)...")
    X_train_lin, X_test_lin, y_train_lin, y_test_lin = engineer_features(
        raw_data, 
        use_linear_model=True, 
        use_smote=True
    )
    print("\n--- Final shapes for Linear Model ---")
    print(f"X_train: {X_train_lin.shape}, y_train: {y_train_lin.shape}")
    print(f"X_test: {X_test_lin.shape}, y_test: {y_test_lin.shape}")
    
    print("\n" + "="*60)
    print("Feature engineering script demonstrations complete.")