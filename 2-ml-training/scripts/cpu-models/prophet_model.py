#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import time
import json
import os
import logging
import warnings
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

def analyze_timeseries(df):
    """Analyze time series characteristics"""
    analysis = {}
    
    # Basic statistics
    y_values = df['y'].values.astype(float)  # Convert to float
    analysis['mean'] = float(np.mean(y_values))
    analysis['std'] = float(np.std(y_values))
    analysis['cv'] = float(analysis['std'] / max(analysis['mean'], 1e-5))  # Coefficient of variation with safety
    
    # Detect bursts
    diff = np.diff(y_values)
    analysis['has_bursts'] = str(bool(np.any(diff > 2 * analysis['std'])))  # Convert to string
    analysis['burst_frequency'] = float(np.sum(diff > 2 * analysis['std']) / max(len(diff), 1))
    
    # Analyze seasonality
    hourly_means = df.groupby(df['ds'].dt.hour)['y'].mean()
    daily_means = df.groupby(df['ds'].dt.dayofweek)['y'].mean()
    
    analysis['hourly_variation'] = float(hourly_means.std() / max(hourly_means.mean(), 1e-5))
    analysis['daily_variation'] = float(daily_means.std() / max(daily_means.mean(), 1e-5))
    
    # Time range analysis
    time_diff = (df['ds'].max() - df['ds'].min()).total_seconds() / 3600
    analysis['duration_hours'] = float(time_diff)
    
    # Calculate data frequency in minutes
    freq = df['ds'].diff().mode().iloc[0].total_seconds() / 60
    analysis['data_frequency'] = float(freq)
    
    return analysis

def create_advanced_features(df):
    """Create advanced features for Prophet model"""
    df = df.copy()
    
    # Ensure datetime
    if 'ds' not in df.columns and 'timestamp' in df.columns:
        df['ds'] = pd.to_datetime(df['timestamp'])
    elif 'ds' not in df.columns:
        raise ValueError("No 'ds' or 'timestamp' column found in DataFrame")
    
    # Create rolling statistics
    if 'y' not in df.columns and 'pod_count' in df.columns:
        df['y'] = df['pod_count']
    elif 'y' not in df.columns:
        raise ValueError("No 'y' or 'pod_count' column found in DataFrame")
    
    windows = [6, 12, 24]
    for window in windows:
        df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()
    
    # Add trend indicators
    df['diff'] = df['y'].diff()
    df['diff_rolling_mean'] = df['diff'].rolling(window=12).mean()
    
    # Add cyclical features
    df['hour'] = df['ds'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Add day of week features
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Fill NaN values
    for col in df.columns:
        if col != 'ds':
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df

def get_optimal_params(analysis):
    """Get optimal Prophet parameters based on data analysis"""
    params = {}
    
    # Adaptive changepoint settings based on data characteristics
    if analysis['has_bursts'] == 'True':
        params['changepoint_prior_scale'] = 0.05  # More conservative for bursts
        params['n_changepoints'] = min(50, int(analysis['duration_hours'] / 24))  # Scale with data length
    else:
        params['changepoint_prior_scale'] = 0.01
        params['n_changepoints'] = min(25, int(analysis['duration_hours'] / 48))
    
    # Enhanced seasonality settings based on data frequency
    if analysis['data_frequency'] <= 15:  # High frequency data (15min or less)
        params['daily_seasonality'] = 20
        params['weekly_seasonality'] = 10
    else:  # Lower frequency data
        params['daily_seasonality'] = 10
        params['weekly_seasonality'] = 5
    
    # Adjust seasonality prior scale based on variations
    if analysis['hourly_variation'] > 0.3 or analysis['daily_variation'] > 0.3:
        params['seasonality_prior_scale'] = 20.0
    else:
        params['seasonality_prior_scale'] = 10.0
    
    # Seasonality mode based on coefficient of variation
    if analysis['cv'] > 0.3:  # Adjusted threshold
        params['seasonality_mode'] = 'multiplicative'
    else:
        params['seasonality_mode'] = 'additive'
    
    # Growth settings
    params['growth'] = 'linear'  # Using linear growth as default for pod scaling
    
    return params

def train_prophet_model(train_df, params):
    """Train Prophet model with optimized parameters"""
    # Create a copy of training data with only required columns initially
    prophet_df = train_df[['ds', 'y']].copy()
    
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        seasonality_mode=params['seasonality_mode'],
        growth=params['growth'],
        n_changepoints=params['n_changepoints']
    )
    
    # Add custom seasonalities
    model.add_seasonality(
        name='daily',
        period=24,
        fourier_order=params['daily_seasonality']
    )
    
    model.add_seasonality(
        name='weekly',
        period=168,  # 24*7
        fourier_order=params['weekly_seasonality']
    )
    
    # Add all numeric columns as regressors except special columns
    special_cols = ['ds', 'y', 'hour', 'day_of_week']  # Columns to exclude
    for col in train_df.columns:
        if col not in special_cols and np.issubdtype(train_df[col].dtype, np.number):
            prophet_df[col] = train_df[col]
            model.add_regressor(
                col,
                mode='additive',
                standardize=True,
                prior_scale=0.5
            )
    
    # Fit the model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # Suppress Prophet warnings
        model.fit(prophet_df)
    
    return model

def evaluate_forecast(y_true, y_pred):
    """Calculate forecast metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Handle zero values in MAPE calculation
    epsilon = 1e-8
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    return rmse, mae, mape, under_provision, over_provision

def validate_and_prepare_data(df):
    """Validate and prepare dataframe for Prophet"""
    # Create a copy to avoid modifying original
    df = df.copy()

    # Check for required columns with case-insensitive matching
    timestamp_cols = [col for col in df.columns if col.lower() in ['timestamp', 'time', 'date', 'ds']]
    pod_count_cols = [col for col in df.columns if col.lower() in ['pod_count', 'pods', 'count', 'y']]
    
    if not timestamp_cols:
        raise ValueError("No timestamp column found. Expected 'timestamp', 'time', 'date', or 'ds'.")
    if not pod_count_cols:
        raise ValueError("No pod count column found. Expected 'pod_count', 'pods', 'count', or 'y'.")
    
    # Use the first matching column for each
    timestamp_col = timestamp_cols[0]
    pod_count_col = pod_count_cols[0]
    
    # Rename columns to Prophet requirements ('ds', 'y')
    df = df.rename(columns={timestamp_col: 'ds', pod_count_col: 'y'})
    
    # Convert pod count to numeric
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Fill any NaN values created by coercion or already present
    df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
    
    # Convert timestamp to datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Ensure no NaN in ds
    if df['ds'].isnull().any():
        raise ValueError("NaN values found in 'ds' column after conversion.")

    # Ensure no NaN in y after fillna
    if df['y'].isnull().any():
        # This shouldn't happen after ffill/bfill unless the column was all NaNs
        raise ValueError("NaN values found in 'y' column after fillna. Check original data.")
    
    # Select only the required columns
    return df[['ds', 'y']]

def make_json_serializable(obj):
    """Convert values to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, bool):
        return str(obj)
    elif isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = os.path.join('train', 'models', 'prophet_model', 'runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        logging.info("Loading data...")
        train_df = pd.read_csv(args.train_file)
        test_df = pd.read_csv(args.test_file)
        
        # Print data columns to debug
        logging.info(f"Train DataFrame columns: {train_df.columns.tolist()}")
        logging.info(f"Test DataFrame columns: {test_df.columns.tolist()}")
        
        # Validate and prepare data for Prophet
        logging.info("Preparing data for Prophet...")
        train_df = validate_and_prepare_data(train_df)
        test_df = validate_and_prepare_data(test_df)
        
        logging.info("Creating advanced features...")
        train_df = create_advanced_features(train_df)
        test_df = create_advanced_features(test_df)
        
        logging.info("Analyzing time series characteristics...")
        analysis = analyze_timeseries(train_df)
        params = get_optimal_params(analysis)
        logging.info(f"Using optimized parameters: {json.dumps(params, indent=2)}")
        
        # Train model
        logging.info("Training Prophet model...")
        start_time = time.time()
        model = train_prophet_model(train_df, params)
        training_time = time.time() - start_time
        
        # Make predictions
        logging.info("Making predictions...")
        forecast = model.predict(test_df)
        y_pred = forecast['yhat'].values
        y_true = test_df['y'].values
        
        # Round predictions to nearest integer and ensure minimum of 1 pod
        y_pred = np.round(y_pred).clip(min=1)
        
        # Calculate metrics
        rmse, mae, mape, under_prov, over_prov = evaluate_forecast(y_true, y_pred)
        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"MAE: {mae:.2f}")
        logging.info(f"MAPE: {mape:.2f}%")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'timestamp': test_df['ds'].astype(str),
            'actual': y_true,
            'predicted': y_pred
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        # Save metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'under_provision': float(under_prov),
            'over_provision': float(over_prov),
            'training_time': float(training_time),
            'model_params': make_json_serializable(params)
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create and save plot
        plt.figure(figsize=(15, 7))
        plt.plot(test_df['ds'], y_true, label='Actual Pod Count', alpha=0.7)
        plt.plot(test_df['ds'], y_pred, label='Predicted Pod Count', alpha=0.7)
        plt.title('Prophet Model - Pod Count Prediction')
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
            'modelRunId': args.run_id
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
