import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import signal
import traceback
import logging
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from functools import lru_cache

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.transformer')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging to be more concise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)

# Global flag for handling interrupts
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    logging.info("Signal received. Cleaning up...")
    interrupted = True

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

# Function to get the appropriate device (CPU, CUDA, or MPS)
def get_device():
    # Check for CUDA first
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    # Then check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            logging.info("Using MPS (Apple Silicon GPU)")
            return device
        except Exception as e:
            logging.warning(f"Error initializing MPS: {e}")
    
    # Fallback to CPU
    logging.info("No GPU available. Using CPU.")
    return torch.device("cpu")

# Get the appropriate device
device = get_device()

# Function to handle mixed precision training
def get_autocast_context():
    if device.type == "cuda":
        return autocast(device_type='cuda')
    elif device.type == "mps":
        # MPS doesn't fully support autocast yet
        return autocast(enabled=False)
    else:
        return autocast(enabled=False)

# Function to create GradScaler for mixed precision training
def get_grad_scaler():
    if device.type == "cuda":
        return GradScaler()
    else:
        # Return a dummy scaler for MPS and CPU
        class DummyScaler:
            def scale(self, loss):
                return loss
                
            def unscale_(self, optimizer):
                pass
                
            def step(self, optimizer):
                optimizer.step()
                
            def update(self):
                pass
        
        return DummyScaler()

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Pre-LayerNorm architecture for better stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LayerNorm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # GELU activation for better performance
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )

    def forward(self, src, src_mask=None):
        # Project input to d_model dimensions
        x = self.input_projection(src)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_mask)
        
        # Project to output
        output = self.output_projection(x)
        return output[:, -1, :]  # Only return the last prediction

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = torch.FloatTensor(features)  # Convert to tensor once
        self.targets = torch.FloatTensor(targets)    # Convert to tensor once
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length + 1

    @lru_cache(maxsize=1024)  # Cache frequently accessed items
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length - 1]
        return x, y.unsqueeze(-1)

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(self.current_step ** (-0.5),
                                        self.current_step * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def validate_data(df, name):
    """Validate input data for required columns and data quality"""
    required_columns = ['timestamp', 'pod_count']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {name} dataset: {missing_columns}")
    
    # Check for null values
    null_columns = df[required_columns].columns[df[required_columns].isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Found null values in columns {null_columns} in {name} dataset")
    
    # Check for negative values in pod_count
    if (df['pod_count'] < 0).any():
        raise ValueError(f"Found negative values in pod_count column in {name} dataset")

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
    
    # Handle zero values in MAPE calculation
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    # Additional metrics
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    return rmse, mae, mape, under_provision, over_provision, r2

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, clip_value=1.0):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Mixed precision training
        with get_autocast_context():
            output = model(data)
            loss = criterion(output, target)
        
        # Scale loss and compute gradients
        scaler.scale(loss).backward()
        
        # Clip gradients
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), clip_value)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad(), get_autocast_context():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())
            
    return total_loss / len(val_loader), np.array(predictions), np.array(actuals)

def save_plot(predictions_df, output_dir):
    """Save plot separately to handle interruptions gracefully"""
    try:
        # Suppress matplotlib logging
        plt.style.use('seaborn')
        plt.figure(figsize=(15, 7))
        plt.plot(pd.to_datetime(predictions_df['timestamp']), predictions_df['actual'], 'b-', label='Actual', alpha=0.7)
        plt.plot(pd.to_datetime(predictions_df['timestamp']), predictions_df['predicted'], 'r--', label='Predicted', alpha=0.7)
        plt.title('Transformer Model Predictions vs Actual Values')
        plt.xlabel('Time')
        plt.ylabel('Pod Count')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"Could not save plot: {str(e)}")

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-file', required=True)
        parser.add_argument('--test-file', required=True)
        parser.add_argument('--run-id', required=True)
        args = parser.parse_args()

        # Set device and enable benchmarking
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        logging.info(f"Device: {device}")

        # Create output directory
        output_dir = os.path.join('./train/models/transformer_model/runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Load and process data
        train_data = pd.read_csv(args.train_file)
        test_data = pd.read_csv(args.test_file)
        validate_data(train_data, "training")
        validate_data(test_data, "testing")

        train_data = create_features(train_data)
        test_data = create_features(test_data)

        # Prepare features
        exclude_cols = ['timestamp', 'pod_count']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[feature_cols])
        y_train = train_data['pod_count'].values
        X_test = scaler.transform(test_data[feature_cols])
        y_test = test_data['pod_count'].values

        # Model configuration
        seq_length = 32
        batch_size = 128
        input_dim = len(feature_cols)
        d_model = 256
        nhead = 8
        num_layers = 6
        dim_feedforward = 512
        dropout = 0.1

        # Initialize datasets and dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
        test_dataset = TimeSeriesDataset(X_test, y_test, seq_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2
        )

        # Initialize model and training components
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        warmup_steps = 4000
        scheduler = WarmupScheduler(optimizer, d_model, warmup_steps)
        scaler = get_grad_scaler()
        
        n_epochs = 50
        best_val_loss = float('inf')
        early_stopping_patience = 10
        early_stopping_counter = 0
        best_model_state = None

        # Training loop
        logging.info("Training started")
        start_time = time.time()
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            if interrupted:
                logging.info("Training interrupted")
                break

            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
            val_loss, _, _ = validate(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                early_stopping_counter = 0
                logging.info(f"Epoch {epoch+1}: New best model (val_loss: {val_loss:.4f})")
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
                
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f}s")

        # Generate predictions
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        model.eval()
        _, y_pred, y_true = validate(model, test_loader, criterion, device)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        y_pred = np.round(y_pred).clip(min=1)

        # Calculate and save metrics
        rmse, mae, mape, under_prov, over_prov, r2 = evaluate_forecast(y_true, y_pred)
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'under_provision': float(under_prov),
            'over_provision': float(over_prov),
            'r2_score': float(r2),
            'training_time': float(training_time),
            'model_params': {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'seq_length': seq_length,
                'batch_size': batch_size
            },
            'features_used': feature_cols
        }

        # Save results
        predictions_df = pd.DataFrame({
            'timestamp': test_data['timestamp'].astype(str)[-len(y_pred):],
            'actual': y_true.tolist(),
            'predicted': y_pred.tolist()
        })
        
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        save_plot(predictions_df, output_dir)
        print(json.dumps(metrics))
        
    except KeyboardInterrupt:
        logging.info("Process interrupted")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(json.dumps({'status': 'error', 'message': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 