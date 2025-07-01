#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate a Baseline model')
    parser.add_argument('--train-file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--test-file', type=str, required=True, help='Path to test data file')
    parser.add_argument('--run-id', type=str, default=None, help='Unique identifier for this run')
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

def train_baseline_model(train_data):
    """Train the Baseline model (simply stores the last observed value)."""
    # The Baseline model just uses the last observed value for prediction
    last_value = int(train_data['pod_count'].iloc[-1])
    return {'last_value': last_value}

def predict_baseline(model, history_data, test_timestamps):
    """Generate predictions using the Baseline model."""
    # Get the last value from the model
    last_value = model['last_value']
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({'timestamp': test_timestamps})
    predictions['predicted'] = last_value
    
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

def save_results(model, predictions, metrics, run_id, runtime):
    """Save model, predictions, and metrics."""
    # Create output directory
    output_dir = os.path.join('train/models/baseline_model/runs', run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_file = os.path.join(output_dir, 'model.json')
    with open(model_file, 'w') as f:
        json.dump(model, f)
    
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
    plt.plot(predictions['timestamp'], predictions['predicted'], 'r--', label='Baseline Predictions')
    
    plt.title('Baseline Model Predictions')
    plt.xlabel('Time')
    plt.ylabel('Pod Count')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_dir = os.path.join('train/models/baseline_model/runs', run_id)
    # Create the full directory path if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'predictions_plot.png')
    plt.savefig(plot_file)
    plt.close()

def main():
    args = parse_args()
    
    # Create run_id if not provided
    if args.run_id is None:
        args.run_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure output directory exists
    os.makedirs('train/models/baseline_model/runs', exist_ok=True)
    
    start_time = datetime.now()
    
    # Load data
    logging.info(f"Loading data from {args.train_file} and {args.test_file}")
    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)
    
    # Train model
    logging.info("Training Baseline model...")
    model = train_baseline_model(train_data)
    
    # Generate predictions
    logging.info("Generating predictions...")
    predictions = predict_baseline(model, train_data, test_data['timestamp'])
    
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
    metrics_json = save_results(model, pred_with_actual, metrics, args.run_id, runtime)
    
    # Print metrics to stdout for train-models.py to capture
    print(metrics_json)
    
    logging.info(f"Baseline model training and evaluation completed in {runtime:.2f} seconds")

if __name__ == "__main__":
    main() 