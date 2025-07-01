#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import json
import sys
import logging
import warnings
from typing import Optional, Tuple, List, Dict
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
warnings.filterwarnings('ignore')

def get_device():
    """Get the appropriate device for training"""
    # Try CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        try:
            # Test GPU computation with a small tensor
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x.t())
            del x, y
            torch.cuda.empty_cache()
            
            device = torch.device("cuda")
            logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        except Exception as e:
            logging.warning(f"CUDA error: {str(e)}. Trying MPS...")
    
    # Try Apple Silicon GPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            # Test MPS with a simple tensor operation
            x = torch.randn(100, 100).to(device)
            y = torch.matmul(x, x.t())
            del x, y
            
            logging.info("Using MPS (Apple Silicon GPU)")
            return device
        except Exception as e:
            logging.warning(f"MPS error: {str(e)}. Falling back to CPU.")
    
    # Fallback to CPU
    logging.info("Using CPU for computations")
    return torch.device("cpu")

# Get device to be used throughout the script
device = get_device()

# Helper function to get appropriate autocast context based on device
def get_autocast_context():
    """Get the appropriate autocast context based on device type"""
    if device.type == "cuda":
        return autocast(device_type='cuda', dtype=torch.float16)
    elif device.type == "mps":
        # MPS doesn't fully support autocast yet, but for future compatibility
        return autocast(device_type='mps', dtype=torch.float32, enabled=False)
    else:
        # Dummy context for CPU
        return autocast(enabled=False)

# Helper function to get appropriate scaler based on device
def get_grad_scaler():
    """Get the appropriate gradient scaler based on device type"""
    if device.type == "cuda":
        return GradScaler()
    else:
        # For MPS and CPU - implement a dummy scaler
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

class Expert(nn.Module):
    """Expert network specialized for different patterns in time series data"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Use GRU instead of LSTM for faster training
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional for better feature extraction
        )
        
        # Efficient attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.GELU(),  # GELU activation for better gradient flow
            nn.Linear(hidden_size, 1)
        )
        
        # Projection layers with skip connection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),  # Layer normalization for stability
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Efficient attention computation
        attn_weights = self.attention(gru_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum using einsum for efficiency
        context = torch.einsum('bsh,bsi->bh', gru_out, attn_weights)
        
        # Final prediction with residual connection
        return self.projection(context)  # [batch_size, 1]

class Router(nn.Module):
    """Router network that determines which expert to use for each input"""
    def __init__(self, input_size: int, num_experts: int, hidden_size: int = 64):
        super().__init__()
        self.num_experts = num_experts
        
        # More efficient feature extractor using separable convolutions
        self.feature_extractor = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1, groups=input_size),
            nn.Conv1d(input_size, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            
            # Second depthwise separable convolution
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Global context module
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU()
        )
        
        # Gating network with temperature scaling
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        # Transpose for convolution: [batch_size, input_size, seq_len]
        x = x.transpose(1, 2)
        
        # Extract features
        local_features = self.feature_extractor(x)  # [batch_size, hidden_size, seq_len]
        
        # Global context
        global_features = self.global_context(local_features)  # [batch_size, hidden_size, 1]
        global_features = global_features.squeeze(-1)  # [batch_size, hidden_size]
        
        # Local features pooling
        local_features = F.adaptive_avg_pool1d(local_features, 1).squeeze(-1)  # [batch_size, hidden_size]
        
        # Combine local and global features
        combined_features = torch.cat([local_features, global_features], dim=1)  # [batch_size, hidden_size*2]
        
        # Get routing logits
        routing_logits = self.gate(combined_features)  # [batch_size, num_experts]
        
        # Apply temperature scaling
        routing_logits = routing_logits / (self.temperature.abs() + 1e-7)
        
        return F.softmax(routing_logits, dim=-1)  # [batch_size, num_experts]

class AdvancedMoEModel(nn.Module):
    """Advanced Mixture of Experts model for time series forecasting"""
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 256, 
        num_experts: int = 4, 
        top_k: int = 2,
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create expert networks with shared parameters for efficiency
        expert_core = Expert(input_size, hidden_size, num_layers, dropout)
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, num_layers, dropout)
            for _ in range(num_experts)
        ])
        
        # Initialize experts with similar but slightly different parameters
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                # Copy base parameters
                expert.load_state_dict(expert_core.state_dict())
                # Add small random perturbation for specialization
                for param in expert.parameters():
                    param.data += torch.randn_like(param) * 0.01
        
        # Create router network
        self.router = Router(input_size, num_experts, hidden_size=hidden_size//2)
        
        # Dynamic noise scaling
        self.register_buffer('noise_scale', torch.tensor(1.0))
        self.noise_decay = 0.99
        
        # Expert specializations
        self.expert_specializations = [
            "cyclical patterns",
            "trend patterns",
            "anomaly patterns",
            "baseline patterns"
        ][:num_experts]
    
    def _add_routing_noise(self, routing_weights: torch.Tensor, training: bool) -> torch.Tensor:
        """Add noise to routing weights during training for exploration"""
        if training and self.noise_scale > 0:
            # Generate noise and scale it
            noise = torch.randn_like(routing_weights) * self.noise_scale
            routing_weights = routing_weights + noise
            # Decay noise scale
            self.noise_scale *= self.noise_decay
            return F.softmax(routing_weights, dim=-1)
        return routing_weights
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        # Ensure input is float32
        x = x.to(dtype=torch.float32)
        batch_size = x.size(0)
        
        # Get routing weights
        routing_weights = self.router(x)  # [batch_size, num_experts]
        routing_weights = self._add_routing_noise(routing_weights, self.training)
        
        # Initialize output tensor
        combined_output = torch.zeros((batch_size, 1), device=x.device, dtype=torch.float32)
        
        # Get expert outputs for all inputs first
        expert_outputs = torch.zeros((batch_size, self.num_experts, 1), device=x.device, dtype=torch.float32)
        for i in range(self.num_experts):
            expert_outputs[:, i, :] = self.experts[i](x)
        
        if self.top_k < self.num_experts:
            # Get top-k experts and weights
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
            # Normalize weights
            top_k_weights = F.normalize(top_k_weights, p=1, dim=1)
            
            # Compute weighted sum for each sample
            for b in range(batch_size):
                expert_idx = top_k_indices[b]  # [top_k]
                weights = top_k_weights[b].unsqueeze(1)  # [top_k, 1]
                combined_output[b] = torch.sum(
                    expert_outputs[b, expert_idx] * weights,
                    dim=0
                )
        else:
            # If using all experts, simple matrix multiplication
            routing_weights = routing_weights.unsqueeze(2)  # [batch_size, num_experts, 1]
            combined_output = torch.sum(expert_outputs * routing_weights, dim=1)
        
        if return_attention:
            return combined_output, routing_weights.squeeze(-1)
        return combined_output

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features for the model"""
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                             (~df['is_weekend'])).astype(int)
    
    # Cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Rolling statistics
    window_sizes = [4, 12, 24]  # 1 hour, 3 hours, 6 hours (assuming 15-min intervals)
    
    for window in window_sizes:
        # Add rolling statistics with minimum periods to handle the start
        df[f'pod_count_rolling_mean_{window}'] = df['pod_count'].rolling(window=window, min_periods=1).mean()
        df[f'pod_count_rolling_std_{window}'] = df['pod_count'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'pod_count_rolling_max_{window}'] = df['pod_count'].rolling(window=window, min_periods=1).max()
        df[f'pod_count_rolling_min_{window}'] = df['pod_count'].rolling(window=window, min_periods=1).min()
    
    # Lag features
    for lag in [1, 2, 4, 8]:
        df[f'pod_count_lag_{lag}'] = df['pod_count'].shift(lag)
    
    # Trend indicators
    df['pod_count_diff'] = df['pod_count'].diff()
    df['pod_count_diff2'] = df['pod_count_diff'].diff()  # Second-order difference
    df['pod_count_diff_sign'] = np.sign(df['pod_count_diff']).fillna(0)
    
    # Fill NAs that result from lag/diff operations
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create input sequences and target values for time series forecasting"""
    sequences, targets = [], []
    
    for i in range(len(data) - seq_length):
        # Input sequence
        seq = data[i:i + seq_length]
        # Target is the next pod count after the sequence
        target = data[i + seq_length, 0]  # pod_count is the first column
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

class PodScalingLoss(nn.Module):
    """Custom loss function for pod scaling prediction"""
    def __init__(self, alpha: float = 3.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Weight for underprediction penalty
        self.beta = beta    # Weight for overprediction penalty
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Base MSE loss
        mse_loss = self.mse(pred, target)
        
        # Calculate prediction error
        error = pred - target
        
        # Penalties
        underprediction = F.relu(-error)  # Error when pred < target
        overprediction = F.relu(error)    # Error when pred > target
        
        # Combined loss with asymmetric penalties
        weighted_loss = mse_loss + (self.alpha * underprediction**2) + (self.beta * overprediction**2)
        
        return torch.mean(weighted_loss)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 15,
    grad_accum_steps: int = 4
) -> nn.Module:
    """Train the model with early stopping and optimized memory usage"""
    model = model.to(device)
    criterion = PodScalingLoss(alpha=3.0, beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Enable debugging for CUDA errors
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader) // grad_accum_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                try:
                    # Move data to GPU and ensure correct dtype
                    inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
                    targets = targets.to(device, non_blocking=True, dtype=torch.float32)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    loss = loss / grad_accum_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # Update learning rate
                        scheduler.step()
                    
                    train_loss += loss.item() * grad_accum_steps
                    
                    # Clear GPU cache periodically
                    if device.type == 'cuda' and (batch_idx + 1) % 100 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        logging.warning(f"Out of memory in batch {batch_idx}. Skipping batch.")
                        if optimizer.state_dict()['state']:
                            optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        raise e
            
            train_loss /= len(train_loader)
            
            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
                        targets = targets.to(device, non_blocking=True, dtype=torch.float32)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets.unsqueeze(1))
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping with model checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log progress with memory stats
            if (epoch + 1) % 10 == 0:
                log_message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}"
                if val_loader is not None:
                    log_message += f", Val Loss: {val_loss:.4f}"
                if device.type == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    memory_reserved = torch.cuda.memory_reserved() / 1024**2
                    log_message += f", GPU Memory: {memory_allocated:.1f}MB/{memory_reserved:.1f}MB"
                logging.info(log_message)
                
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        raise e
    
    # Load best model if we have one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    return model

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate predictions with comprehensive metrics"""
    # Ensure arrays have the right shape
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    # Basic metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Handling zero values in MAPE calculation
    epsilon = 1e-10
    y_true_safe = np.where(y_true < epsilon, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    # Resource provisioning metrics
    under_provision = np.mean(np.maximum(0, y_true - y_pred))
    over_provision = np.mean(np.maximum(0, y_pred - y_true))
    
    # Additional metrics
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'under_provision': float(under_provision),
        'over_provision': float(over_provision),
        'r2_score': float(r2)
    }

def analyze_expert_specialization(
    model: AdvancedMoEModel, 
    dataloader: DataLoader,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Analyze which patterns each expert specializes in"""
    model.eval()
    
    all_routing_weights = []
    all_inputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            
            # Get routing weights
            _, routing_weights = model(inputs, return_attention=True)
            
            all_routing_weights.append(routing_weights.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate results
    routing_weights = np.concatenate(all_routing_weights, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate dominant expert for each sample
    dominant_experts = np.argmax(routing_weights, axis=1)
    
    # Create a dataframe for analysis
    results = {
        'target': targets.ravel(),
        'dominant_expert': dominant_experts
    }
    
    results_df = pd.DataFrame(results)
    
    # Calculate expert distribution
    expert_dist = results_df['dominant_expert'].value_counts(normalize=True).sort_index() * 100
    
    # Format for display
    if class_names is None:
        class_names = [f"Expert {i}" for i in range(len(expert_dist))]
    
    expert_analysis = pd.DataFrame({
        'Expert': class_names[:len(expert_dist)],
        'Specialization': model.expert_specializations[:len(expert_dist)],
        'Usage (%)': expert_dist.values.round(2)
    })
    
    return expert_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, help='Path to training data CSV')
    parser.add_argument('--test-file', required=True, help='Path to test data CSV')
    parser.add_argument('--run-id', required=True, help='Unique ID for this run')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--grad-accum-steps', type=int, default=4, help='Number of gradient accumulation steps')
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = os.path.join('./train/models/advanced_moe/runs', args.run_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        logging.info("Loading data...")
        train_df = pd.read_csv(args.train_file)
        test_df = pd.read_csv(args.test_file)
        
        # Create features
        logging.info("Creating features...")
        train_df = create_time_features(train_df)
        test_df = create_time_features(test_df)
        
        # Select feature columns (excluding timestamp)
        exclude_cols = ['timestamp']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        # Scale features
        logging.info("Scaling features...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        test_scaled = scaler.transform(test_df[feature_cols])
        
        # Create sequences
        seq_length = 24  # 6 hours with 15-minute intervals
        logging.info(f"Creating sequences with length {seq_length}...")
        train_sequences, train_targets = create_sequences(train_scaled, seq_length)
        test_sequences, test_targets = create_sequences(test_scaled, seq_length)
        
        # Split training data for validation
        val_size = int(len(train_sequences) * 0.2)
        train_sequences, val_sequences = train_sequences[:-val_size], train_sequences[-val_size:]
        train_targets, val_targets = train_targets[:-val_size], train_targets[-val_size:]
        
        # Create dataloaders with optimized settings
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_sequences), torch.FloatTensor(train_targets)),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_sequences), torch.FloatTensor(val_targets)),
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_sequences), torch.FloatTensor(test_targets)),
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2
        )
        
        # Initialize model with optimized settings
        input_size = train_sequences.shape[2]
        hidden_size = 256  # Increased for better capacity
        num_experts = 4
        top_k = 2
        
        logging.info(f"Initializing MoE model with {num_experts} experts, using top-{top_k} routing...")
        model = AdvancedMoEModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            num_layers=2,  # Increased depth
            dropout=0.2
        )
        
        # Train model with optimized settings
        logging.info("Training model...")
        start_time = time.time()
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            lr=0.001,
            patience=15,
            grad_accum_steps=args.grad_accum_steps
        )
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
        # Generate predictions
        logging.info("Generating predictions...")
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        predictions = np.array(all_predictions).ravel()
        test_targets = np.array(all_targets)
        
        # Scale pod count back to original range
        pod_count_idx = feature_cols.index('pod_count')
        pod_count_mean = scaler.mean_[pod_count_idx]
        pod_count_scale = scaler.scale_[pod_count_idx]
        
        predictions_original = (predictions * pod_count_scale) + pod_count_mean
        targets_original = (test_targets * pod_count_scale) + pod_count_mean
        
        # Round predictions to nearest integer and ensure minimum of 1 pod
        predictions_final = np.round(predictions_original).clip(min=1)
        
        # Evaluate predictions
        logging.info("Evaluating model performance...")
        metrics = evaluate_predictions(targets_original, predictions_final)
        
        # Add training time and model parameters to metrics
        metrics.update({
            'training_time': float(training_time),
            'model_params': {
                'num_experts': num_experts,
                'top_k': top_k,
                'hidden_size': hidden_size,
                'seq_length': seq_length,
                'features': feature_cols
            }
        })
        
        # Analyze expert specialization
        logging.info("Analyzing expert specialization...")
        expert_analysis = analyze_expert_specialization(model, test_loader, class_names=None)
        logging.info("\nExpert Specialization Analysis:")
        for _, row in expert_analysis.iterrows():
            logging.info(f"{row['Expert']} ({row['Specialization']}): {row['Usage (%)']}% of samples")
        
        # Create predictions DataFrame
        valid_timestamps = test_df['timestamp'].iloc[seq_length:len(test_targets) + seq_length]
        predictions_df = pd.DataFrame({
            'timestamp': valid_timestamps,
            'actual': targets_original,
            'predicted': predictions_final
        })
        
        # Save results
        logging.info("Saving results...")
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(15, 7))
        plt.plot(predictions_df['timestamp'], predictions_df['actual'], 
                'b-', label='Actual', alpha=0.7)
        plt.plot(predictions_df['timestamp'], predictions_df['predicted'], 
                'r--', label='Predicted', alpha=0.7)
        plt.title('Advanced MoE Model Predictions vs Actual Values')
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