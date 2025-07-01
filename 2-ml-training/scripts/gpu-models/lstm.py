#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import json
import sys
import logging
import warnings
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

def verify_gpu_operation():
    """Verify that GPU operations are actually working"""
    # First check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            # Test CUDA computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform compute-intensive operations
            z = torch.matmul(x, y)
            z = torch.nn.functional.relu(z)
            z = torch.matmul(z, y.t())
            
            # Force synchronization and get result
            z = z.cpu()
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            
            logging.info("CUDA GPU verification successful")
            return True
            
        except Exception as e:
            logging.warning(f"CUDA GPU verification failed: {str(e)}")
            # If CUDA fails, try MPS next
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test MPS computation
            x = torch.randn(1000, 1000).to("mps")
            y = torch.randn(1000, 1000).to("mps")
            
            # Perform compute-intensive operations
            z = torch.matmul(x, y)
            z = torch.nn.functional.relu(z)
            z = torch.matmul(z, y.t())
            
            # Force synchronization and get result
            z = z.cpu()
            
            # Clean up
            del x, y, z
            
            logging.info("MPS (Apple Silicon) GPU verification successful")
            return True
            
        except Exception as e:
            logging.warning(f"MPS GPU verification failed: {str(e)}")
    
    logging.warning("No GPU available or verification failed")
    return False

def get_device():
    """Get the appropriate device for training"""
    # First check for CUDA (NVIDIA)
    if torch.cuda.is_available() and verify_gpu_operation():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    # Then check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Perform a more robust test of MPS functionality
            device = torch.device("mps")
            # Test with real computation that would be done during training
            x = torch.randn(100, 100).to(device)
            y = torch.randn(100, 100).to(device)
            z = torch.matmul(x, y)
            # Force synchronization by moving back to CPU
            z = z.to("cpu")
            logging.info("Using MPS (Apple Silicon GPU)")
            return device
        except Exception as e:
            logging.warning(f"MPS setup error: {str(e)}")
    
    # Fallback to CPU
    logging.warning("GPU not available or verification failed. Using CPU instead.")
    return torch.device("cpu")

# Get device
device = get_device()

class PodScalingLoss(nn.Module):
    def __init__(self, alpha=3.0, beta=1.5):
        super().__init__()
        self.alpha = alpha  # Under-provisioning penalty
        self.beta = beta    # Over-provisioning penalty
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, pred, target):
        base_loss = self.smooth_l1(pred, target)
        diff = pred - target
        under_provision = torch.relu(-diff)
        over_provision = torch.relu(diff)
        
        weighted_loss = (self.alpha * under_provision**2 + 
                        self.beta * over_provision**2 +
                        base_loss)
        return torch.mean(weighted_loss)

class LightweightLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.using_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        # Optimized feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),  # Changed to SiLU for better performance
            nn.Dropout(dropout)
        )
        
        # Optimized attention for A100
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False  # Removed bias for faster computation
        )
        
        # Optimized LSTM - use different settings for MPS
        if self.using_mps:
            # MPS-compatible version without projection
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,  # Use smaller hidden size instead of projection
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            self.lstm_output_size = hidden_size
        else:
            # Original version with projection
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                proj_size=hidden_size // 2  # Added projection for better memory efficiency
            )
            self.lstm_output_size = hidden_size
        
        # Optimized regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Optimized attention computation
        attn_output, _ = self.attention(features, features, features)
        
        # Optimized LSTM computation with different initialization for MPS
        if self.using_mps:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size // 2, device=x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size // 2, device=x.device)
        else:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size // 2, device=x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
            
        lstm_out, _ = self.lstm(attn_output, (h0, c0))
        
        # Final prediction
        output = self.regressor(lstm_out[:, -1, :])
        return output

def analyze_dataset(df):
    """Analyze dataset characteristics to determine optimal hyperparameters"""
    analysis = {}
    
    # Analyze time series characteristics
    pod_counts = df['pod_count'].values
    analysis['mean_pods'] = np.mean(pod_counts)
    analysis['std_pods'] = np.std(pod_counts)
    analysis['max_pods'] = np.max(pod_counts)
    analysis['min_pods'] = np.min(pod_counts)
    
    # Detect bursts and patterns
    analysis['has_bursts'] = np.any(np.diff(pod_counts) > analysis['std_pods'] * 2)
    
    # Calculate autocorrelation
    autocorr = np.correlate(pod_counts, pod_counts, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    analysis['autocorr_strength'] = np.max(autocorr[1:12]) / autocorr[0]
    
    # Analyze seasonality
    timestamps = pd.to_datetime(df['timestamp'])
    analysis['time_range_hours'] = (timestamps.max() - timestamps.min()).total_seconds() / 3600
    
    # Detect if the data has regular patterns
    fft = np.fft.fft(pod_counts)
    frequencies = np.fft.fftfreq(len(pod_counts))
    peak_freq = frequencies[np.argmax(np.abs(fft[1:]) + 1)]
    analysis['has_regular_patterns'] = np.abs(peak_freq) > 0.1
    
    return analysis

def get_optimal_hyperparameters(analysis):
    """Determine optimal hyperparameters based on dataset characteristics"""
    params = {}
    
    # Optimize sequence length based on data characteristics
    if analysis['has_regular_patterns']:
        params['seq_length'] = 16  # Reduced from 24 for better GPU utilization
    else:
        params['seq_length'] = 8   # Reduced from 12 for better GPU utilization
    
    # Optimized model architecture for A100
    params['hidden_size'] = 128    # Increased for better capacity
    params['num_layers'] = 1       # Keep single layer for efficiency
    params['num_heads'] = 4        # Increased for better parallel processing
    params['dropout'] = 0.1        # Reduced for faster training
    params['bidirectional'] = True
    
    # Optimized batch size for A100
    params['batch_size'] = 128     # Increased for better GPU utilization
    params['learning_rate'] = 0.002 # Adjusted for larger batch size
    params['epochs'] = 100         # Reduced from 200
    params['patience'] = 15        # Reduced from 20
    
    # Loss parameters
    params['loss_alpha'] = 3.0
    params['loss_beta'] = 1.5
    
    return params

def create_advanced_features(df, analysis):
    """Create essential features only"""
    df = df.copy()
    
    # Essential temporal features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
    
    # Minimal window sizes
    windows = [4, 12]  # 1h, 3h for 15-min intervals
    
    for window in windows:
        # Essential statistics only
        df[f'pod_count_rolling_mean_{window}'] = df['pod_count'].rolling(window=window, min_periods=1).mean()
        df[f'pod_count_rolling_std_{window}'] = df['pod_count'].rolling(window=window, min_periods=1).std()
    
    # Basic trend features
    df['pod_count_diff'] = df['pod_count'].diff()
    
    # Basic cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def create_sequences(data, seq_length):
    """Create sequences with multiple features"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length, 0]  # pod_count is the first column
        sequences.append(seq)
        targets.append(target)
    
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

def train_model(model, train_sequences, train_targets, val_sequences=None, val_targets=None, 
                epochs=100, batch_size=128, lr=0.002, patience=15):
    """Train the model with optimizations for A100 GPU"""
    model = model.to(device)
    criterion = PodScalingLoss(alpha=3.0, beta=1.5)
    
    # Optimized optimizer configuration
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Optimized learning rate scheduler
    steps_per_epoch = len(train_sequences) // batch_size
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Reduced warm-up period
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Optimize CUDA settings for A100
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster computation
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cudnn.deterministic = False  # Disable deterministic mode for speed
        # Set memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Optimized data loading
    train_dataset = TensorDataset(train_sequences.float(), train_targets.float().unsqueeze(1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Increased for faster data loading
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    if val_sequences is not None:
        val_dataset = TensorDataset(val_sequences.float(), val_targets.float().unsqueeze(1))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch size for validation
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop with optimizations
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Use tqdm for progress tracking
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Optimized forward and backward pass
            with autocast():
                output = model(batch_x)
                loss = criterion(output, batch_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Efficient memory cleanup
            del batch_x, batch_y, output, loss
        
        avg_train_loss = total_loss / batch_count
        
        if val_sequences is not None:
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad(), autocast():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    del batch_x, batch_y, output, loss
            
            avg_val_loss = val_loss / val_batch_count
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 5 == 0:  # Reduced logging frequency
            log_msg = f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}'
            if val_sequences is not None:
                log_msg += f', Val Loss: {avg_val_loss:.4f}'
            logging.info(log_msg)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions with comprehensive metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Handle zero values in MAPE calculation
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Resource provisioning metrics
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    # Additional metrics
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    return rmse, mae, mape, under_provision, over_provision, r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    try:
        # Verify GPU availability
        device = get_device()
        
        # Create output directory
        output_dir = os.path.join('./train/models/lstm_model/runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        logging.info("Loading data...")
        train_df = pd.read_csv(args.train_file)
        test_df = pd.read_csv(args.test_file)
        
        # Analyze dataset
        logging.info("Analyzing dataset characteristics...")
        analysis = analyze_dataset(train_df)
        
        # Get optimal hyperparameters
        params = get_optimal_hyperparameters(analysis)
        
        # Create features
        logging.info("Creating advanced features...")
        train_df = create_advanced_features(train_df, analysis)
        test_df = create_advanced_features(test_df, analysis)
        
        # Select features
        feature_columns = [col for col in train_df.columns 
                         if col not in ['timestamp']]
        
        # Scale features
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_df[feature_columns])
        test_scaled = scaler.transform(test_df[feature_columns])
        
        # Create sequences
        train_sequences, train_targets = create_sequences(train_scaled, params['seq_length'])
        test_sequences, test_targets = create_sequences(test_scaled, params['seq_length'])
        
        # Split training data for validation
        val_size = int(len(train_sequences) * 0.2)
        train_sequences, val_sequences = train_sequences[:-val_size], train_sequences[-val_size:]
        train_targets, val_targets = train_targets[:-val_size], train_targets[-val_size:]
        
        # Initialize model with lightweight architecture
        model = LightweightLSTMPredictor(
            input_size=len(feature_columns),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            dropout=params['dropout']
        )
        
        # Train model
        logging.info("Training model with optimized parameters...")
        start_time = time.time()
        model = train_model(
            model, train_sequences, train_targets,
            val_sequences, val_targets,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            lr=params['learning_rate'],
            patience=params['patience']
        )
        training_time = time.time() - start_time
        
        # Generate predictions with memory optimization
        model.eval()
        predictions = []
        batch_size = params['batch_size']
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_sequences = test_sequences[i:i + batch_size].to(device)
                batch_predictions = model(batch_sequences).cpu().numpy()
                predictions.append(batch_predictions)
                del batch_sequences
                torch.cuda.empty_cache()
        
        predictions = np.vstack(predictions)
        
        # Inverse transform predictions
        pod_count_scaler = RobustScaler().fit(train_df[['pod_count']])
        predictions_original = pod_count_scaler.inverse_transform(predictions)
        test_targets_original = pod_count_scaler.inverse_transform(test_targets.reshape(-1, 1))
        
        # Round predictions and ensure minimum value of 1
        predictions_original = np.round(predictions_original).clip(min=1)
        
        # Evaluate predictions
        rmse, mae, mape, under_prov, over_prov, r2 = evaluate_predictions(
            test_targets_original, predictions_original
        )
        
        # Save metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'under_provision': float(under_prov),
            'over_provision': float(over_prov),
            'r2_score': float(r2),
            'training_time': float(training_time),
            'model_params': {
                'seq_length': params['seq_length'],
                'hidden_size': params['hidden_size'],
                'num_layers': params['num_layers'],
                'num_heads': params['num_heads'],
                'features': feature_columns
            }
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create predictions DataFrame
        valid_timestamps = test_df['timestamp'].iloc[params['seq_length']:len(test_targets_original) + params['seq_length']]
        predictions_df = pd.DataFrame({
            'timestamp': valid_timestamps,
            'actual': test_targets_original.squeeze(),
            'predicted': predictions_original.squeeze()
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(15, 7))
        plt.plot(predictions_df['timestamp'], predictions_df['actual'], 
                'b-', label='Actual', alpha=0.7)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted'], 
                'r--', label='Predicted', alpha=0.7)
        plt.title('LSTM Model Predictions vs Actual Values')
        plt.xlabel('Time')
        plt.ylabel('Pod Count')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plot.png'), dpi=300, bbox_inches='tight')
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