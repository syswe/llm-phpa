#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
import logging
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate a Vector Autoregression (VAR) model')
    parser.add_argument('--train-file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--test-file', type=str, required=True, help='Path to test data file')
    parser.add_argument('--run-id', type=str, default=None, help='Unique identifier for this run')
    parser.add_argument('--maxlags', type=int, default=24, help='Maximum number of lags')
    return parser.parse_args()

def load_data(file_path):
    """Load the data from a CSV file."""
    df = pd.read_csv(file_path)
    
    # Ensure timestamp column is properly formatted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'ds' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ds'])
        if 'y' in df.columns and 'pod_count' not in df.columns:
            df['pod_count'] = df['y']
    
    # Make sure we have the required columns
    required_cols = ['timestamp', 'pod_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df

def prepare_var_data(data):
    """Prepare data for the VAR model."""
    df = data.copy()
    
    # Extract time features that will be used as additional variables in the VAR model
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # For VAR we need at least 2 series
    var_data = df[['pod_count', 'hour', 'day_of_week']].copy()
    
    # VAR requires stationary data - check if we need to difference
    # For simplicity, we'll just use the raw values
    
    return var_data

def train_var_model(train_data, maxlags=24):
    """Train a VAR model."""
    try:
        # Prepare data for VAR
        var_data = prepare_var_data(train_data)
        
        # Create and fit the VAR model
        model = VAR(var_data)
        model_fit = model.fit(maxlags=min(maxlags, len(var_data) // 5))  # Avoid using too many lags
        
        # Save information about the model
        info = {
            'maxlags': model_fit.k_ar,
            'params': model_fit.params.values.tolist() if hasattr(model_fit, 'params') else None,
            'variables': var_data.columns.tolist()
        }
        
        return {
            'model': model_fit,
            'info': info
        }
    except Exception as e:
        logging.error(f"Error training VAR model: {e}")
        # Return a fallback model that will use the baseline method
        return None

def predict_var(model_data, history_data, test_timestamps):
    """Generate predictions using the VAR model."""
    if model_data is None:
        # Fallback to baseline prediction
        last_value = int(history_data['pod_count'].iloc[-1])
        predictions = pd.DataFrame({'timestamp': test_timestamps, 'predicted': last_value})
        return predictions
    
    try:
        model = model_data['model']
        
        # Prepare var data from history
        var_data = prepare_var_data(history_data)
        
        # Get the lag order
        lag_order = model.k_ar
        
        # Get the most recent values for forecast
        forecast_input = var_data.values[-lag_order:]
        
        # Generate forecast
        forecast = model.forecast(y=forecast_input, steps=len(test_timestamps))
        
        # The first column is the pod_count prediction
        predictions = pd.DataFrame({
            'timestamp': test_timestamps,
            'predicted': np.maximum(0, np.round(forecast[:, 0])).astype(int)  # Ensure non-negative integers
        })
        
        return predictions
    except Exception as e:
        logging.error(f"Error generating VAR predictions: {e}")
        # Fallback to baseline prediction
        last_value = int(history_data['pod_count'].iloc[-1])
        predictions = pd.DataFrame({'timestamp': test_timestamps, 'predicted': last_value})
        return predictions

def evaluate_model(actual, predicted):
    """Calculate evaluation metrics."""
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Handle zeros in actual values to avoid division by zero in MAPE
    actual_safe = np.where(actual == 0, 0.1, actual)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual_safe, predicted) * 100
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }

def save_results(model_data, predictions, metrics, run_id, runtime):
    """Save model, predictions, and metrics."""
    # Create output directory
    output_dir = os.path.join('train/models/var_model/runs', run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model information
    if model_data is not None:
        model_file = os.path.join(output_dir, 'model_info.json')
        with open(model_file, 'w') as f:
            json.dump(model_data['info'], f)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    predictions.to_csv(predictions_file, index=False)
    
    # Add training time to metrics
    metrics['training_time'] = runtime
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    # Print metrics
    logging.info(f"Model metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
    
    # Return metrics as JSON string for train-models.py to capture
    return json.dumps(metrics)

def plot_predictions(train_data, test_data, predictions, run_id):
    """Create a plot of the predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(train_data['timestamp'], train_data['pod_count'], 'b-', label='Training Data')
    
    # Plot test data
    plt.plot(test_data['timestamp'], test_data['pod_count'], 'k-', label='Actual Values')
    
    # Plot predictions
    plt.plot(predictions['timestamp'], predictions['predicted'], 'r--', label='VAR Predictions')
    
    plt.title('Vector Autoregression (VAR) Model Predictions')
    plt.xlabel('Time')
    plt.ylabel('Pod Count')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.join('train/models/var_model/runs', run_id)
    # Create the full directory path if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'predictions_plot.png')
    plt.savefig(plot_file)
    plt.close()

def main():
    args = parse_args()
    
    # Create run_id if not provided
    if args.run_id is None:
        args.run_id = f"var_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure output directory exists
    os.makedirs('train/models/var_model/runs', exist_ok=True)
    
    start_time = datetime.now()
    
    # Load data
    logging.info(f"Loading data from {args.train_file} and {args.test_file}")
    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)
    
    # Train model
    logging.info("Training VAR model...")
    model_data = train_var_model(train_data, maxlags=args.maxlags)
    
    if model_data is None:
        logging.warning("VAR model training failed. Falling back to baseline predictions.")
    
    # Generate predictions
    logging.info("Generating predictions...")
    predictions = predict_var(model_data, train_data, test_data['timestamp'].values)
    
    # Combine predictions with actual values
    pred_with_actual = predictions.copy()
    pred_with_actual['actual'] = test_data['pod_count'].values[:len(predictions)]
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_model(
        pred_with_actual['actual'].values,
        pred_with_actual['predicted'].values
    )
    
    # Calculate runtime
    runtime = (datetime.now() - start_time).total_seconds()
    
    # Create visualization
    logging.info("Creating visualization...")
    plot_predictions(train_data, test_data, predictions, args.run_id)
    
    # Save results
    logging.info("Saving results...")
    metrics_json = save_results(model_data, pred_with_actual, metrics, args.run_id, runtime)
    
    # Print metrics to stdout for train-models.py to capture
    print(metrics_json)
    
    logging.info(f"VAR model training and evaluation completed in {runtime:.2f} seconds")

if __name__ == "__main__":
    main() 