import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json
import sys
import gc
from datetime import datetime
from contextlib import redirect_stdout

def validate_data(df, name):
    """Validate input data, handling different column names."""
    # Check for timestamp column
    if 'timestamp' in df.columns:
        ts_col = 'timestamp'
    elif 'ds' in df.columns:
        ts_col = 'ds'
        df = df.rename(columns={'ds': 'timestamp'})
    else:
        raise ValueError(f"Missing timestamp column ('timestamp' or 'ds') in {name} dataset")
        
    # Check for pod count column
    if 'pod_count' in df.columns:
        pod_col = 'pod_count'
    elif 'y' in df.columns:
        pod_col = 'y'
        df = df.rename(columns={'y': 'pod_count'})
    else:
         raise ValueError(f"Missing pod count column ('pod_count' or 'y') in {name} dataset")

    required_columns = ['timestamp', 'pod_count']
    
    # Check for null values in the essential columns
    null_columns = df[required_columns].columns[df[required_columns].isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Found null values in columns {null_columns} in {name} dataset after potential renaming")
    
    # Check for negative values in pod_count
    if (df['pod_count'] < 0).any():
        raise ValueError(f"Found negative values in pod_count column in {name} dataset")
    
    # Convert timestamp just to be sure it's datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df # Return potentially modified DataFrame

def create_features(data):
    """Create time series features from timestamp"""
    result = data.copy()
    
    # Convert timestamp to datetime if it's not already
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    # Extract time features
    result['hour'] = result['timestamp'].dt.hour
    result['dayofweek'] = result['timestamp'].dt.dayofweek
    result['month'] = result['timestamp'].dt.month
    result['day'] = result['timestamp'].dt.day
    result['is_weekend'] = result['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Create cyclical features for hour and day of week
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['day_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)
    
    # Create lag features
    for lag in [1, 2, 3, 4]:
        result[f'pod_count_lag_{lag}'] = result['pod_count'].shift(lag)
    
    # Create rolling mean features
    for window in [3, 6, 12]:
        result[f'pod_count_rolling_mean_{window}'] = result['pod_count'].rolling(window=window).mean()
        result[f'pod_count_rolling_std_{window}'] = result['pod_count'].rolling(window=window).std()
    
    # Drop rows with NaN values from lag features
    result = result.dropna()
    
    return result

def evaluate_forecast(y_true, y_pred):
    """Evaluate forecast with multiple metrics"""
    if len(y_true) != len(y_pred):
        raise ValueError("Length of actual and predicted values must match")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    return rmse, mae, mape, under_provision, over_provision

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-file', required=True)
        parser.add_argument('--test-file', required=True)
        parser.add_argument('--run-id', required=True)
        args = parser.parse_args()

        # Create output directory
        output_dir = os.path.join('./train/models/xgboost_model/runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Load and validate data
        train_data = pd.read_csv(args.train_file)
        test_data = pd.read_csv(args.test_file)
        
        # Use the validated (and potentially renamed) dataframes
        train_data = validate_data(train_data, "training")
        test_data = validate_data(test_data, "testing")

        # Create features
        train_data = create_features(train_data)
        test_data = create_features(test_data)

        # Define features
        exclude_cols = ['timestamp', 'pod_count']
        features = [col for col in train_data.columns if col not in exclude_cols]
        X_train = train_data[features]
        y_train = train_data['pod_count']
        X_test = test_data[features]
        y_test = test_data['pod_count']

        # Train model with optimized parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'tree_method': 'hist',  # Faster training
            'early_stopping_rounds': 20,  # Add early stopping here
            'random_state': 42
        }

        model = xgb.XGBRegressor(**params)
        start_time = time.time()
        
        # Create DMatrix for evaluation
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Setup evaluation list
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        
        # Train using lower-level API for better control
        model_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'tree_method': 'hist'
        }
        
        # Redirect training progress to stderr and train
        num_round = 200
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            bst = xgb.train(
                model_params,
                dtrain,
                num_round,
                evallist,
                early_stopping_rounds=20,
                verbose_eval=True
            )
        
        training_time = time.time() - start_time

        # Make predictions using the trained model
        y_pred = bst.predict(dtest)
        y_pred = np.round(y_pred).clip(min=1)  # Ensure minimum of 1 pod

        # Calculate metrics
        rmse, mae, mape, under_prov, over_prov = evaluate_forecast(y_test, y_pred)

        # Save metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'under_provision': float(under_prov),
            'over_provision': float(over_prov),
            'training_time': float(training_time),
            'model_params': params,
            'features_used': features
        }

        # Save predictions
        predictions_df = pd.DataFrame({
            'timestamp': test_data['timestamp'].astype(str),
            'actual': y_test.tolist(),
            'predicted': y_pred.tolist()
        })
        
        # Save results
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create and save plot
        plt.figure(figsize=(15, 7))
        plt.plot(test_data['timestamp'], y_test, label='Actual Pod Count', alpha=0.7)
        plt.plot(test_data['timestamp'], y_pred, label='Predicted Pod Count', alpha=0.7)
        plt.title('XGBoost Model - Pod Count Prediction')
        plt.xlabel('Time')
        plt.ylabel('Pod Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot.png'), dpi=300)
        plt.close()

        # Print metrics for API
        print(json.dumps(metrics))
        sys.exit(0)

    except Exception as e:
        error_output = {
            'status': 'error',
            'message': str(e),
            'modelRunId': args.run_id if 'args' in locals() else None
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
