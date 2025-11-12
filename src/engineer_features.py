import pandas as pd
import numpy as np

def engineer_features(raw_df: pd.DataFrame, use_linear_model: bool = False) -> pd.DataFrame:
    """
    Transforms the raw e-commerce transaction data into a model-ready feature set.
    
    This function performs the following steps based on EDA insights:
    1.  Handles datetime conversion.
    2.  Creates simple features: log_transaction_amount, address_mismatch, is_new_account.
    3.  Engineers a powerful behavioral feature: deviation from the customer's average spend.
    4.  Performs one-hot encoding for categorical variables.
    5.  Selects the final set of features and drops unnecessary/raw columns.

    Args:
        raw_df (pd.DataFrame): The raw DataFrame loaded from the CSV.
        use_linear_model (bool): 
            - If False (default): Keeps 'Transaction Amount' and drops the log version. 
                                  Recommended for non-linear models like trees.
            - If True: Keeps 'log_transaction_amount' and drops the original. 
                       Recommended for linear models.

    Returns:
        pd.DataFrame: A processed DataFrame ready for machine learning.
    """
    # Work on a copy to avoid modifying the original DataFrame
    df = raw_df.copy()

    # --- 1. Datetime and Sorting ---
    # Convert to datetime for proper sorting and potential time-based features
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    # Sort data to correctly calculate historical features
    df = df.sort_values(by=['Customer ID', 'Transaction Date'])


    # --- 2. Simple Feature Creation ---
    # Log-transform the skewed 'Transaction Amount'
    # Using log1p to gracefully handle potential zero values
    df['log_transaction_amount'] = np.log1p(df['Transaction Amount'])

    # Create a binary flag for address mismatch, a common fraud indicator
    df['address_mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)

    # Create a binary flag for new accounts, as EDA showed they are higher risk
    # A threshold of 30 days is a reasonable starting point.
    df['is_new_account'] = (df['Account Age Days'] < 30).astype(int)

    # --- 3. Advanced Behavioral Feature Engineering ---
    # Calculate the customer's average transaction amount *before* the current transaction.
    # This is a powerful feature to detect anomalous spending.
    # .shift(1) is crucial to prevent data leakage from the current transaction.
    df['customer_avg_spend_before_tx'] = df.groupby('Customer ID')['Transaction Amount'] \
                                           .expanding(min_periods=1).mean() \
                                           .reset_index(level=0, drop=True) \
                                           .shift(1)
    
    # For a customer's very first transaction, the historical average is NaN. Fill with 0.
    df['customer_avg_spend_before_tx'].fillna(0, inplace=True)
    
    # Calculate how much the current transaction deviates from the customer's historical average.
    # We add 1 to the denominator to avoid division by zero for the first transaction.
    df['amount_deviation'] = (df['Transaction Amount'] - df['customer_avg_spend_before_tx']) / (df['customer_avg_spend_before_tx'] + 1)

    # --- 4. Categorical Variable Encoding ---
    # Convert categorical columns into numerical format using one-hot encoding.
    # drop_first=True helps reduce multicollinearity.
    df = pd.get_dummies(df, columns=['Payment Method', 'Product Category', 'Device Used'], drop_first=True)

    # --- 5. Final Feature Selection and Cleanup ---
    # Define all columns that are no longer needed for modeling.
    # These include identifiers, high-cardinality text, raw date, and intermediate columns.
    columns_to_drop = [
        'Transaction ID', 
        'Customer ID', 
        'Transaction Date', 
        'Customer Location', 
        'IP Address', 
        'Shipping Address', 
        'Billing Address',
        'customer_avg_spend_before_tx' # This was an intermediate step, the deviation is the final feature
    ]
    
    # Based on the function argument, choose which amount feature to drop.
    if use_linear_model:
        print("Strategy: Using LOG-TRANSFORMED amount. Dropping original 'Transaction Amount'.")
        columns_to_drop.append('Transaction Amount')
    else:
        print("Strategy: Using ORIGINAL amount. Dropping 'log_transaction_amount'.")
        columns_to_drop.append('log_transaction_amount')

    df.drop(columns=columns_to_drop, inplace=True)
    
    print("Feature engineering complete.")
    return df


if __name__ == '__main__':
    print("Loading raw data...")
    try:
        raw_data = pd.read_csv('../Fraudulent_E-Commerce_Transaction_Data_2.csv')
    except FileNotFoundError:
        print("Error: 'Fraudulent_E-Commerce_Transaction_Data_2.csv' not found.")
        print("Please make sure the data file is in the same directory as the script.")
        exit()

    print("\n" + "="*60)
    # --- DEMONSTRATION 1: For a Non Linear Model (e.g., Tree-Based Model (e.g., RandomForest, XGBoost) ---
    print("Generating data with ORIGINAL Transaction Amount (for Non Linear Models)...")
    model_ready_data_tree = engineer_features(raw_data, use_linear_model=False)
    print("\n--- Final columns for Tree Model ---")
    print(model_ready_data_tree.columns)

    print("\n" + "="*60)
    # --- DEMONSTRATION 2: For a Linear Model (e.g., Logistic Regression) ---
    print("Generating data with LOG Transaction Amount (for Linear Models)...")
    model_ready_data_linear = engineer_features(raw_data, use_linear_model=True)
    print("\n--- Final columns for Linear Model ---")
    print(model_ready_data_linear.columns)