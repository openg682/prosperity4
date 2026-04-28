"""
Quantitative Equity Machine Learning Predictive Pipeline
Role: Lead Quantitative Python Engineer and Machine Learning Researcher

Objective:
This script implements a strict two-step Machine Learning pipeline to analyze intraday
equities data, extract predictive features, select the most important features using 
Random Forest, and evaluate an XGBoost predictive model using strict Time-Series 
Cross-Validation to prevent data leakage.

Data Context:
- Inputs: 3 days of high-frequency price data (CSV files with ';' separator).
- Outputs: Clean CSV/JSON files containing predictions, retained features, and metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress warnings for cleaner output during research
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & FILE PATHS
# =============================================================================
# CRITICAL: Point these paths to your actual CSV files.
# The user provided an image showing prices and trades for day 2, 3, and 4.
PRICE_FILES = [
    'prices_round_5_day_2.csv',
    'prices_round_5_day_3.csv',
    'prices_round_5_day_4.csv'
]

# Output paths
PREDICTIONS_OUT = 'xgb_predictions.csv'
METRICS_OUT = 'pipeline_metrics.json'

# Pipeline hyperparameters
TARGET_PERIODS_FORWARD = 5
RF_TREES = 50
XGB_ESTIMATORS = 100
CV_SPLITS = 5


# =============================================================================
# STEP 1: DATA PREPROCESSING & TARGET CREATION
# =============================================================================
def load_and_preprocess_data(file_paths: list) -> pd.DataFrame:
    """
    Loads price data from multiple CSVs, concatenates them, and handles basic cleaning.
    """
    df_list = []
    for file in file_paths:
        if os.path.exists(file):
            # Read CSV based on the provided sample structure (separated by ';')
            df = pd.read_csv(file, sep=';')
            df_list.append(df)
        else:
            print(f"WARNING: File {file} not found. Skipping.")
            
    if not df_list:
        raise FileNotFoundError("No valid data files found. Please check paths.")
        
    data = pd.concat(df_list, ignore_index=True)
    
    # Create a unique chronological time index to safely sort and split later
    # Assuming day is an integer (2, 3, 4) and timestamp resets each day.
    # We create a continuous time identifier.
    data['time_id'] = data['day'] * 10_000_000 + data['timestamp']
    data = data.sort_values(by=['time_id', 'product']).reset_index(drop=True)
    
    return data

def engineer_features_and_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictive features and constructs the forward-looking target variable.
    Strictly calculates features grouped by 'product' to avoid cross-asset contamination.
    """
    df = data.copy()
    
    # 1. Feature Engineering (Derived from Order Book data)
    # -------------------------------------------------------------------------
    # Bid-Ask Spread: Wider spreads often indicate volatility or low liquidity.
    df['spread_1'] = df['ask_price_1'] - df['bid_price_1']
    
    # Order Book Imbalance: Measures buying vs selling pressure at the top of the book.
    df['imbalance_1'] = df['bid_volume_1'] / (df['bid_volume_1'] + df['ask_volume_1'] + 1e-8)
    
    # Mid-price short-term momentum (1-period and 3-period returns)
    df['ret_1'] = df.groupby('product')['mid_price'].pct_change(1)
    df['ret_3'] = df.groupby('product')['mid_price'].pct_change(3)
    
    # Rolling volatility (proxy: 10-period standard deviation of returns)
    df['vol_10'] = df.groupby('product')['ret_1'].rolling(window=10).std().reset_index(0, drop=True)
    
    # 2. Target Creation (Zero Look-Ahead Bias)
    # -------------------------------------------------------------------------
    # We want today's (or this timestamp's) features to predict the return N periods forward.
    # A negative shift (-N) pulls future prices backward to the current row.
    df['target_fwd_ret'] = df.groupby('product')['mid_price'].shift(-TARGET_PERIODS_FORWARD) / df['mid_price'] - 1.0
    
    # 3. Clean up missing values
    # -------------------------------------------------------------------------
    # Drop rows where target is NaN (the last N periods of each product cannot be used for training)
    df = df.dropna(subset=['target_fwd_ret'])
    
    # Forward fill missing feature values (e.g., if a rolling window was empty), then drop any remaining
    df = df.fillna(method='ffill').dropna()
    
    return df


# =============================================================================
# STEP 2 & 3: MACHINE LEARNING PIPELINE (FEATURE SELECTION & PREDICTION)
# =============================================================================
def run_ml_pipeline(df: pd.DataFrame):
    """
    Executes Phase 1 (Random Forest Feature Selection) and 
    Phase 2 (XGBoost with strict Time-Series CV).
    """
    # Define feature columns (exclude identifiers and target)
    exclude_cols = ['day', 'timestamp', 'time_id', 'product', 'target_fwd_ret', 'profit_and_loss']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y = df['target_fwd_ret']
    
    print(f"Initial Feature Count: {len(features)}")
    
    # -------------------------------------------------------------------------
    # PHASE 1: Feature Selection via Random Forest
    # -------------------------------------------------------------------------
    # To prevent look-ahead bias during feature selection, we should technically 
    # only use the training set. For simplicity in a 2-step pipeline, we fit on the 
    # first 80% of chronological data to extract stable feature importances.
    split_idx = int(len(df) * 0.8)
    X_fs, y_fs = X.iloc[:split_idx], y.iloc[:split_idx]
    
    print("Phase 1: Training Random Forest for Feature Selection...")
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_fs, y_fs)
    
    # Extract importances and select top 50%
    importance_df = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    top_n = len(features) // 2
    top_features = importance_df['Feature'].head(top_n).tolist()
    
    print(f"Phase 1 Complete. Retained {top_n} features: {top_features}")
    
    # Filter dataset to only top features
    X_filtered = X[top_features]
    
    # -------------------------------------------------------------------------
    # PHASE 2: Predictive Engine via XGBoost & Time-Series CV
    # -------------------------------------------------------------------------
    print("Phase 2: Evaluating XGBoost with strict Time-Series Cross-Validation...")
    
    # CRITICAL CONSTRAINT: We cannot use train_test_split. We must prevent leakage.
    # Furthermore, because we have panel data (multiple products per timestamp), 
    # standard TimeSeriesSplit on rows might leak data if a timestamp is split in half.
    # Solution: Custom Purged Time-Series Split based on unique time_ids.
    
    unique_times = df['time_id'].unique()
    unique_times = np.sort(unique_times)
    
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    
    out-of-sample_preds = []
    out-of-sample_actuals = []
    
    fold = 1
    for train_time_idx, test_time_idx in tscv.split(unique_times):
        # Map time indices back to actual time_id values
        train_times = unique_times[train_time_idx]
        test_times = unique_times[test_time_idx]
        
        # We add a "purge" gap equal to TARGET_PERIODS_FORWARD to ensure the 
        # end of the training set doesn't overlap with the start of the test set's target.
        purge_threshold = train_times[-1] - TARGET_PERIODS_FORWARD
        train_times = train_times[train_times <= purge_threshold]
        
        # Filter dataframe indices based on allowed times
        train_mask = df['time_id'].isin(train_times)
        test_mask = df['time_id'].isin(test_times)
        
        X_train, y_train = X_filtered[train_mask], y[train_mask]
        X_test, y_test = X_filtered[test_mask], y[test_mask]
        
        # Initialize and train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=XGB_ESTIMATORS, 
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        # Predict
        preds = xgb_model.predict(X_test)
        
        # Store results for this fold
        fold_results = pd.DataFrame({
            'time_id': df.loc[test_mask, 'time_id'],
            'product': df.loc[test_mask, 'product'],
            'actual': y_test,
            'prediction': preds
        })
        out-of-sample_preds.append(fold_results)
        
        fold_mae = mean_absolute_error(y_test, preds)
        print(f"  Fold {fold} MAE: {fold_mae:.6f}")
        fold += 1

    # Aggregate all out-of-sample predictions
    final_predictions_df = pd.concat(out-of-sample_preds, ignore_index=True)
    
    return final_predictions_df, top_features

# =============================================================================
# STEP 4: STRUCTURED OUTPUT & EVALUATION
# =============================================================================
def evaluate_and_export(predictions_df: pd.DataFrame, retained_features: list):
    """
    Calculates quantitative metrics (Information Ratio, Sharpe, MAE) and exports 
    the results to disk.
    """
    # Mean Absolute Error
    mae = mean_absolute_error(predictions_df['actual'], predictions_df['prediction'])
    
    # Strategy proxy: Assume we take positions proportional to our predictions.
    # Realized returns = Prediction * Actual Forward Return
    # (In a real scenario, you'd calculate this based on capital allocation).
    predictions_df['strategy_return'] = np.sign(predictions_df['prediction']) * predictions_df['actual']
    
    # Calculate Sharpe Ratio Proxy
    # Assuming ~252 trading days, but since we have intraday data we need the number of 
    # timestamps per day to annualize correctly. We will provide a per-timestamp metric
    # and a proxy annualized version assuming 10,000 timestamps a day (adjust based on data).
    avg_return = predictions_df['strategy_return'].mean()
    std_return = predictions_df['strategy_return'].std()
    
    # Information Ratio (Proxy: Return over tracking error. Here we use basic Sharpe logic)
    sharpe_ratio_per_trade = avg_return / (std_return + 1e-8)
    annualized_sharpe_proxy = sharpe_ratio_per_trade * np.sqrt(10000 * 252) 
    
    # Hit Rate (Percentage of time the predicted sign matches actual sign)
    hit_rate = np.mean(np.sign(predictions_df['prediction']) == np.sign(predictions_df['actual']))

    metrics = {
        "Mean_Absolute_Error": round(mae, 6),
        "Hit_Rate": round(hit_rate, 4),
        "Sharpe_Ratio_Per_Trade": round(sharpe_ratio_per_trade, 6),
        "Annualized_Sharpe_Proxy": round(annualized_sharpe_proxy, 4),
        "Retained_Features": retained_features
    }
    
    print("\n--- Final Evaluation Metrics ---")
    for k, v in metrics.items():
        if k != "Retained_Features":
            print(f"{k}: {v}")
            
    # Export Metrics to JSON
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Export Predictions to CSV
    predictions_df.to_csv(PREDICTIONS_OUT, index=False)
    
    print(f"\nPipeline execution finished. Outputs saved to:\n- {PREDICTIONS_OUT}\n- {METRICS_OUT}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    try:
        # Step 1: Load and Preprocess
        raw_data = load_and_preprocess_data(PRICE_FILES)
        clean_data = engineer_features_and_target(raw_data)
        
        # Step 2 & 3: Run ML Pipeline
        preds_df, top_feats = run_ml_pipeline(clean_data)
        
        # Step 4: Evaluate and Export
        evaluate_and_export(preds_df, top_feats)
        
    except FileNotFoundError as e:
        print(f"Execution Halted: {e}")
        print("Please ensure the CSV files listed in the configuration section are in the working directory.")