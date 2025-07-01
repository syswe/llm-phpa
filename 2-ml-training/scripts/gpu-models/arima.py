#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json
import sys
import logging
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.nn import LayerNorm

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Configure logging with a more concise format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Only show the message without the INFO:root: prefix
)

# Setup device with fallback and optimize CUDA settings
def get_device():
    """Get the appropriate device for training with optimized settings"""
    if torch.cuda.is_available():
        # Optimize CUDA performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda")
        # Only log device info if DEBUG level is enabled
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        logging.debug("Using MPS (Metal) device")
        return torch.device("mps")
    else:
        logging.debug("Using CPU device")
        return torch.device("cpu")

device = get_device()

class TorchARIMA(nn.Module):
    def __init__(self, p, d, q, seasonal_periods=None, batch_size=128):
        super(TorchARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.batch_size = batch_size
        self.seasonal_periods = seasonal_periods

        # Enhanced model architecture with better capacity
        hidden_size = 256
        self.ar_layer = nn.Sequential(
            nn.Linear(p, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.ma_layer = nn.Sequential(
            nn.Linear(q, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        if x.size(1) != self.p + self.q:
            raise ValueError(f"Expected input size {self.p + self.q}, got {x.size(1)}")

        ar_input = x[:, :self.p]
        ma_input = x[:, self.p:]

        ar_hidden = self.ar_layer(ar_input)
        ma_hidden = self.ma_layer(ma_input)
        
        combined = torch.cat([ar_hidden, ma_hidden], dim=1)
        output = self.combined_layer(combined)
        return output

def evaluate_forecast(y_true, y_pred):
    """Evaluate forecast with comprehensive metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    return rmse, mae, mape, under_provision, over_provision

def load_data(file_path):
    """Load and prepare data"""
    try:
        df = pd.read_csv(file_path)
        
        # Handle both formats (ds,y) and (timestamp,pod_count)
        if 'ds' in df.columns and 'y' in df.columns:
            df = df.rename(columns={'ds': 'timestamp', 'y': 'pod_count'})
        
        # Ensure we have required columns
        required_cols = ['timestamp', 'pod_count']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected {required_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Ensure pod_count is numeric
        df['pod_count'] = pd.to_numeric(df['pod_count'], errors='coerce')
        
        # Handle missing values
        df['pod_count'] = df['pod_count'].fillna(method='ffill').fillna(method='bfill')
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def train_model(model, train_data, epochs=100, lr=0.001):
    """Train the ARIMA model with optimized settings for A100"""
    device = get_device()
    model = model.to(device)
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = GradScaler()
    criterion = nn.HuberLoss()  # More robust than MSE
    
    # Prepare sequences for training
    sequences = []
    targets = []
    
    for i in range(len(train_data) - model.p - model.q):
        seq = train_data[i:i + model.p + model.q]
        target = train_data[i + model.p + model.q]
        sequences.append(seq)
        targets.append(target)
    
    # Convert to tensors and move to GPU if available
    sequences = torch.FloatTensor(sequences)
    targets = torch.FloatTensor(targets)
    
    # Optimize batch size for A100
    train_dataset = torch.utils.data.TensorDataset(sequences, targets.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=model.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use automatic mixed precision
            with autocast():
                output = model(batch_x)
                loss = criterion(output, batch_y)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = os.path.join('train', 'models', 'arima', 'runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        logging.info("Loading data...")
        train_df = load_data(args.train_file)
        test_df = load_data(args.test_file)
        
        # Model parameters
        p, d, q = 24, 1, 24  # Increased order for better modeling
        
        # Prepare training data
        train_data = train_df['pod_count'].values
        test_data = test_df['pod_count'].values
        
        # Create and train model with optimized batch size
        model = TorchARIMA(p=p, d=d, q=q, batch_size=args.batch_size)
        
        start_time = time.time()
        logging.info("Training model...")
        model = train_model(model, train_data, epochs=args.epochs, lr=0.001)
        training_time = time.time() - start_time
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            predictions = []
            for i in range(len(test_data)):
                if i < p + q:
                    predictions.append(float(test_data[i]))
                    continue
                
                # Prepare input sequence
                input_seq = test_data[i - (p + q):i]
                model_input = torch.FloatTensor(input_seq).unsqueeze(0).to(device)
                
                # Get prediction
                pred = model(model_input).item()
                predictions.append(pred)
        
        # Evaluate predictions
        rmse, mae, mape, under_prov, over_prov = evaluate_forecast(test_data, predictions)
        
        # Save metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'under_provision': float(under_prov),
            'over_provision': float(over_prov),
            'training_time': float(training_time),
            'model_params': {
                'p': p,
                'd': d,
                'q': q
            }
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'timestamp': test_df['timestamp'],
            'actual': test_data,
            'predicted': predictions
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        # Create plot
        plt.figure(figsize=(15, 7))
        plt.plot(predictions_df['timestamp'], predictions_df['actual'], 
                'b-', label='Actual', alpha=0.7)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted'], 
                'r--', label='Predicted', alpha=0.7)
        plt.title('ARIMA Model Predictions vs Actual Values')
        plt.xlabel('Time')
        plt.ylabel('Pod Count')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot.png'))
        plt.close()
        
        # Print metrics for API
        print(json.dumps(metrics))
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        print(json.dumps({
            'status': 'error',
            'message': str(e)
        }), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
