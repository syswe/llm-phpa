import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json
import sys
from datetime import datetime
import torch.backends.cudnn as cudnn
import logging

# Configure logging to be more concise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Enable CUDA optimizations only when using CUDA
def configure_backends():
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking and deterministic algorithms for CUDA
        cudnn.benchmark = True
        cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere
        logging.info("CUDA optimizations enabled")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS (Apple Silicon) doesn't need the same optimizations as CUDA
        logging.info("MPS backend detected - using default settings")
    else:
        logging.info("Using CPU backend")

# Configure backends based on availability
configure_backends()

def get_device():
    """Get the appropriate device for training"""
    # First check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            # Test CUDA computation
            x = torch.ones(1).cuda()
            device = torch.device("cuda")
            logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        except Exception as e:
            logging.warning(f"CUDA error: {str(e)}. Trying MPS...")
    
    # Then check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test MPS with a simple tensor operation
            x = torch.ones(1).to("mps")
            if x.device.type == "mps":
                device = torch.device("mps")
                logging.info("Using MPS (Apple Silicon GPU)")
                return device
        except Exception as e:
            logging.warning(f"MPS setup error: {str(e)}")
    
    # Fallback to CPU
    logging.info("GPU not available. Using CPU instead.")
    return torch.device("cpu")

# Get device to be used throughout
device = get_device()

# Function to determine if mixed precision is available
def use_mixed_precision():
    if device.type == "cuda":
        return True
    elif device.type == "mps":
        # Currently MPS does not fully support mixed precision
        return False
    else:
        return False

# Function to handle autocast context for both CUDA and MPS
def get_autocast_context():
    if device.type == "cuda":
        return autocast(device_type='cuda', dtype=torch.float16)
    elif device.type == "mps":
        # MPS doesn't fully support autocast yet, but this is for future compatibility
        return autocast(device_type='mps', dtype=torch.float16, enabled=False)
    else:
        # Dummy context for CPU
        return autocast(enabled=False)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_idx=0):
        # Pre-compute all sequences to speed up data loading
        data_tensor = torch.FloatTensor(data).contiguous()
        
        # More efficient sequence generation using tensor operations
        indices = torch.arange(len(data) - seq_length)
        sequences = torch.stack([data_tensor[i:i + seq_length] for i in indices])
        targets = data_tensor[seq_length:, target_idx]
        
        self.sequences = sequences.contiguous()
        self.targets = targets.contiguous()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Use nn.Sequential for better CUDA optimization
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for layer in self.conv_block1:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        for layer in self.conv_block2:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Pre-compute dilations and paddings
        dilations = [2 ** i for i in range(num_levels)]
        paddings = [(kernel_size-1) * d for d in dilations]
        
        # Create layers with optimized parameters
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilations[i],
                    padding=paddings[i],
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Optimize memory access patterns
        x = x.transpose(1, 2).contiguous()
        y1 = self.network(x)
        o = self.linear(y1[:, :, -1])
        return o

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
    
    # Check for negative values in numeric columns
    if (df['pod_count'] < 0).any():
        raise ValueError(f"Found negative values in pod_count column in {name} dataset")

def create_features(data):
    """Create time series features from datetime index"""
    result = data.copy()
    
    # Convert timestamp to datetime if it's not already
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    # Extract time features
    result['hour'] = result['timestamp'].dt.hour
    result['dayofweek'] = result['timestamp'].dt.dayofweek
    result['month'] = result['timestamp'].dt.month
    result['day'] = result['timestamp'].dt.day
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['day_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)
    
    # Create rolling statistics
    result['pod_count_rolling_mean_3'] = result['pod_count'].rolling(window=3).mean()
    result['pod_count_rolling_std_3'] = result['pod_count'].rolling(window=3).std()
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result

def evaluate_forecast(y_true, y_pred):
    """Evaluate forecast with multiple metrics"""
    if len(y_true) != len(y_pred):
        raise ValueError("Length of actual and predicted values must match")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Safer MAPE calculation
    y_true_safe = np.where(y_true < 0.5, 0.5, y_true)  # Avoid division by very small numbers
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    return rmse, mae, mape, under_provision, over_provision

# Custom loss function for pod count prediction
def combined_loss(y_pred, y_true):
    # MSE component
    mse = nn.MSELoss()(y_pred, y_true)
    
    # Add Poisson-like loss component for count data
    # Add small epsilon to avoid log(0)
    y_pred_safe = y_pred.clone() + 1e-8
    poisson = torch.mean(y_pred_safe - y_true * torch.log(y_pred_safe))
    
    # Weighted combination
    return mse + 0.1 * poisson

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=10, accumulation_steps=2):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    train_losses = []
    val_losses = []
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Pre-allocate tensors for loss calculation
    train_loss = torch.zeros(1, device=device)
    val_loss = torch.zeros(1, device=device)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        # Training phase
        model.train()
        train_loss.zero_()
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        for i, (X_batch, y_batch) in enumerate(train_loader):
            # Transfer to GPU asynchronously
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            # Mixed precision training
            with get_autocast_context():
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss = loss / accumulation_steps
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.detach() * accumulation_steps
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss.item())
        
        # Validation phase
        model.eval()
        val_loss.zero_()
        
        with torch.no_grad(), get_autocast_context():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred.squeeze(), y_batch)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss.item())
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # Clear GPU cache periodically
        if epoch % 10 == 0:  # Reduced frequency of cache clearing
            torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-file', required=True)
        parser.add_argument('--test-file', required=True)
        parser.add_argument('--run-id', required=True)
        args = parser.parse_args()

        # Create output directory
        output_dir = os.path.join('./train/models/tcn_model/runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Set device and optimize CUDA settings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            # Set CUDA optimization flags
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        print(f"Using device: {device}")

        # Load and validate data
        print(f"Loading training data from {args.train_file}")
        train_data = pd.read_csv(args.train_file)
        print(f"Loading test data from {args.test_file}")
        test_data = pd.read_csv(args.test_file)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        validate_data(train_data, "training")
        validate_data(test_data, "testing")

        # Create features
        print("Creating features...")
        train_data = create_features(train_data)
        test_data = create_features(test_data)

        # Prepare feature columns
        feature_columns = ['pod_count', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                         'pod_count_rolling_mean_3', 'pod_count_rolling_std_3']

        # Optimize data preprocessing
        with get_autocast_context():
            # Create a separate scaler for pod_count
            pod_count_scaler = MinMaxScaler()
            pod_count_scaled = pod_count_scaler.fit_transform(train_data[['pod_count']])
            train_data['pod_count_scaled'] = pod_count_scaled
            test_data['pod_count_scaled'] = pod_count_scaler.transform(test_data[['pod_count']])
            
            # Scale other features
            feature_scaler = MinMaxScaler()
            scaled_feature_columns = ['pod_count_scaled' if col == 'pod_count' else col for col in feature_columns]
            
            other_features = [col for col in feature_columns if col != 'pod_count']
            if other_features:
                other_features_scaled = feature_scaler.fit_transform(train_data[other_features])
                for i, col in enumerate(other_features):
                    train_data[f"{col}_scaled"] = other_features_scaled[:, i]
                    test_data[f"{col}_scaled"] = feature_scaler.transform(test_data[other_features])[:, i]
                
                scaled_feature_columns = [f"{col}_scaled" if col != 'pod_count_scaled' else col for col in scaled_feature_columns]
        
        print(f"Using features: {scaled_feature_columns}")
        
        # Split training data
        train_size = int(len(train_data) * 0.8)
        train_df = train_data.iloc[:train_size]
        val_df = train_data.iloc[train_size:]
        
        print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")

        # Optimize dataset creation
        seq_length = 24  # 6 hours with 15-minute intervals
        train_dataset = TimeSeriesDataset(train_df[scaled_feature_columns].values, seq_length)
        val_dataset = TimeSeriesDataset(val_df[scaled_feature_columns].values, seq_length)
        test_dataset = TimeSeriesDataset(test_data[scaled_feature_columns].values, seq_length)

        # Optimize data loading
        num_workers = min(6, os.cpu_count())  # Adjusted workers
        batch_size = 256  # Increased batch size
        pin_memory = True if torch.cuda.is_available() else False
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=3
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=3
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=3
        )

        # Optimize model architecture
        input_dim = len(scaled_feature_columns)
        num_channels = [32, 64, 96, 128]  # Modified architecture
        kernel_size = 3
        dropout = 0.2

        # Initialize model with optimizations
        model = TCNModel(input_dim, num_channels, kernel_size, dropout).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Use channels last memory format for better performance
        model = model.to(memory_format=torch.channels_last)
        print(f"Model architecture:\n{model}")
        
        # Optimize training parameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=True  # Enable faster optimizer implementation
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        start_time = time.time()

        # Train model with optimizations
        print("Starting model training...")
        model, train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=100,
            patience=10,
            accumulation_steps=2
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Optimized prediction phase
        model.eval()
        predictions = []
        actuals = []
        
        print("Generating predictions on test data...")
        with torch.no_grad(), get_autocast_context():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_pred = model(X_batch)
                predictions.extend(y_pred.cpu().numpy().squeeze())
                actuals.extend(y_batch.numpy())

        # Process predictions
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        predictions = pod_count_scaler.inverse_transform(predictions)
        actuals = pod_count_scaler.inverse_transform(actuals)
        
        y_pred = np.round(predictions).clip(min=1)
        y_test = actuals

        # Calculate metrics
        rmse, mae, mape, under_provision, over_provision = evaluate_forecast(y_test.flatten(), y_pred.flatten())

        # Save results
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'under_provision': float(under_provision),
            'over_provision': float(over_provision),
            'training_time': training_time
        }

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
        torch.save({
            'pod_count_scaler': pod_count_scaler,
            'feature_scaler': feature_scaler
        }, os.path.join(output_dir, 'scalers.pt'))

        # Generate plots
        plt.figure(figsize=(15, 6))
        plt.plot(y_test, label='Actual', alpha=0.5)
        plt.plot(y_pred, label='Predicted', alpha=0.5)
        plt.title('TCN Model: Actual vs Predicted Pod Count')
        plt.xlabel('Time Steps')
        plt.ylabel('Pod Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'predictions.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('TCN Model: Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
        plt.close()

        print(f"Training completed. Results saved in {output_dir}")
        print(f"Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}")
        print(f"Under-provision={under_provision:.4f}, Over-provision={over_provision:.4f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 