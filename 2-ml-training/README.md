# Machine Learning Model Training and Evaluation Framework

## Overview

This directory contains a comprehensive machine learning framework for evaluating and comparing forecasting models designed for predictive Kubernetes autoscaling. The research addresses the critical challenge of balancing prediction accuracy with computational efficiency in resource-constrained container orchestration environments.

## Research Context

### Problem Statement

Traditional Kubernetes autoscaling relies on reactive heuristic methods that fail to capture complex temporal dependencies in modern microservices workloads. This framework evaluates advanced forecasting models to enable proactive pod scaling decisions while maintaining operational feasibility across diverse deployment scenarios.

### Key Challenges

- **Computational Constraints**: Models must operate within typical Kubernetes pod resource allocations
- **Real-time Requirements**: Training and inference must complete within acceptable latency thresholds
- **Pattern Complexity**: Workloads exhibit diverse temporal patterns from regular seasonal cycles to chaotic burst scenarios
- **Resource Efficiency**: Solutions must balance accuracy with memory usage and computational overhead

## Model Portfolio

### CPU-Compatible Models (Production-Ready)

These models prioritize operational feasibility while maintaining competitive accuracy:

#### 1. **Gradient Boosting Decision Trees (GBDT)**
- **Implementation**: Scikit-learn GradientBoostingRegressor
- **Architecture**: 200 estimators, max depth 6, learning rate 0.1
- **Strengths**: Robust handling of non-linear patterns, efficient training
- **Configuration**: Stochastic sampling (0.8), early stopping, regularization

#### 2. **XGBoost**
- **Implementation**: Optimized gradient boosting with advanced regularization
- **Architecture**: Histogram-based tree construction, L1/L2 regularization
- **Strengths**: Second-order optimization, efficient memory usage
- **Configuration**: 200 rounds, depth 6, early stopping patience 20

#### 3. **LightGBM**
- **Implementation**: Gradient-based One-Side Sampling (GOSS)
- **Architecture**: Leaf-wise growth, exclusive feature bundling
- **Strengths**: Superior computational efficiency, large-scale optimization
- **Configuration**: 31 leaves, feature/bagging fraction 0.8

#### 4. **CatBoost**
- **Implementation**: Ordered boosting methodology
- **Architecture**: Bayesian target statistics, categorical feature processing
- **Strengths**: Eliminates prediction shift, robust categorical handling
- **Configuration**: 1000 iterations, Bernoulli bootstrap, L2 regularization

#### 5. **Prophet**
- **Implementation**: Facebook's time series forecasting framework
- **Architecture**: Additive/multiplicative seasonality, trend decomposition
- **Strengths**: Automated parameter tuning, holiday effects, missing data handling
- **Configuration**: Adaptive parameters based on data characteristics

#### 6. **Vector Autoregression (VAR)**
- **Implementation**: Multivariate time series modeling
- **Architecture**: Information-theoretic lag selection, temporal dependencies
- **Strengths**: Captures inter-variable relationships, statistical foundations
- **Configuration**: BIC-based lag selection, maximum 24 lags

#### 7. **Baseline Model**
- **Implementation**: Last-value persistence
- **Architecture**: Simple heuristic baseline
- **Strengths**: Minimal computational overhead, benchmark reference
- **Configuration**: Single parameter (last observed value)

### GPU-Accelerated Models (Advanced Research)

Advanced deep learning architectures with superior theoretical capacity but significant computational requirements:

#### 1. **Long Short-Term Memory (LSTM)**
- **Architecture**: Bidirectional LSTM with multi-head attention
- **Features**: 128-256 hidden units, temporal encoding, dropout regularization
- **Requirements**: 4-8GB GPU memory, PyTorch runtime

#### 2. **Transformer**
- **Architecture**: Multi-head self-attention, positional encoding
- **Features**: 6-8 encoder layers, 8 attention heads, GELU activation
- **Requirements**: 6-12GB GPU memory, extensive hyperparameter tuning

#### 3. **Temporal Convolutional Network (TCN)**
- **Architecture**: Dilated causal convolutions, residual connections
- **Features**: Exponential dilation, parallel computation, causal constraints
- **Requirements**: 2-6GB GPU memory, optimized for sequential processing

#### 4. **Advanced Mixture of Experts (MoE)**
- **Architecture**: 4 specialized expert networks with intelligent routing
- **Features**: Pattern-specific experts, top-k selection, separable convolutions
- **Requirements**: 8-16GB GPU memory, complex training dynamics

#### 5. **Neural ARIMA**
- **Architecture**: PyTorch-based ARIMA with neural enhancements
- **Features**: Deep fully-connected layers, LayerNorm, mixed precision
- **Requirements**: 2-4GB GPU memory, autoregressive modeling

## Feature Engineering Framework

### Temporal Features

```python
# Cyclical encoding preserves temporal continuity
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
day_sin = sin(2π × day_of_week / 7)
day_cos = cos(2π × day_of_week / 7)
```

### Autoregressive Features

```python
# Lag features capture temporal dependencies
lag_features = [pod_count(t-1), pod_count(t-2), pod_count(t-3), pod_count(t-4)]
```

### Rolling Statistics

```python
# Multi-scale window analysis
windows = [3, 6, 12]  # Short, medium, long-term patterns
rolling_mean_w = (1/w) × Σ(pod_count(t-k)) for k in [0, w-1]
rolling_std_w = sqrt((1/(w-1)) × Σ(pod_count(t-k) - rolling_mean_w)²)
```

### Advanced Features

- **Business hour indicators**: Working hours vs. off-hours classification
- **Weekend/weekday patterns**: Weekly seasonality capture
- **Trend indicators**: First and second-order differences
- **Burst detection**: Gradient-based anomaly identification

## Evaluation Methodology

### Metrics Framework

**Primary Metrics:**
- **RMSE**: Root Mean Square Error for overall accuracy
- **MAE**: Mean Absolute Error for robust central tendency
- **MAPE**: Mean Absolute Percentage Error for relative accuracy

**Kubernetes-Specific Metrics:**
- **Under-provisioning**: Mean(max(0, actual - predicted))
- **Over-provisioning**: Mean(max(0, predicted - actual))
- **Resource efficiency**: Cost implications of prediction errors

### Cross-Validation Strategy

```python
# Temporal cross-validation preserving time series structure
for k in range(K_folds):
    train_period = [t_0, t_k]
    test_period = [t_k+1, t_k+h]
    cv_score += evaluate_model(train_period, test_period)
```

### Performance Analysis

- **Pattern-specific evaluation**: Model performance across different workload types
- **Computational profiling**: Training time, memory usage, inference latency
- **Scalability analysis**: Performance as dataset size increases
- **Robustness testing**: Model stability across diverse scenarios

## Installation and Setup

### Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Gradient boosting frameworks
pip install xgboost lightgbm catboost

# Time series specialized
pip install prophet statsmodels

# GPU models (optional)
pip install torch torchvision torchaudio
pip install transformers
```

### Environment Configuration

```bash
# Create conda environment
conda create -n ml-training python=3.9
conda activate ml-training

# Install dependencies
pip install -r requirements.txt

# Verify GPU support (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage Examples

### Basic Model Training

```bash
# Train all models on all scenarios
python scripts/train-models.py

# Train specific models
python scripts/train-models.py --models "xgboost,lightgbm,prophet"

# Train on specific scenarios
python scripts/train-models.py --scenarios "seasonal_001,burst_003,chaotic_005"

# Limit number of scenarios for testing
python scripts/train-models.py --limit 5
```

### Advanced Configuration

```bash
# Load existing results and create visualizations only
python scripts/train-models.py --visualize-only

# Parallel execution with custom worker count
python scripts/train-models.py --parallel --workers 8

# Generate comprehensive HTML report
python scripts/train-models.py --html-report

# Specify custom scenarios list
python scripts/train-models.py --scenarios-list scenarios_list.txt
```

### Individual Model Training

```bash
# Train specific models independently
python scripts/cpu-models/train_xgboost.py \
  --train-file ../1-dataset-generation/scripts/test_output_all/seasonal_001_train.csv \
  --test-file ../1-dataset-generation/scripts/test_output_all/seasonal_001_test.csv \
  --run-id seasonal_001_xgb_001

# GPU models (requires CUDA/MPS)
python scripts/gpu-models/lstm.py \
  --train-file train_data.csv \
  --test-file test_data.csv \
  --run-id lstm_experiment_001
```

## Output Structure

### Results Organization

```
train/
├── models/                    # Model-specific results
│   ├── xgboost_model/
│   │   └── runs/
│   │       └── {run_id}/
│   │           ├── metrics.json
│   │           ├── predictions.csv
│   │           └── plot.png
│   ├── lightgbm_model/
│   ├── prophet_model/
│   └── ...
├── results/                   # Comparative analysis
│   ├── plots/                 # Visualization outputs
│   ├── summary/               # Aggregated metrics
│   └── {scenario}/            # Scenario-specific results
└── logs/                      # Training logs
```

### Metrics Format

```json
{
  "rmse": 2.45,
  "mae": 1.87,
  "mape": 12.3,
  "under_provision": 0.85,
  "over_provision": 1.23,
  "training_time": 3.47,
  "model_params": {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1
  }
}
```

## Visualization and Analysis

### Automated Visualizations

The framework generates comprehensive analysis outputs:

- **Performance comparison matrices**: Model vs. scenario heatmaps
- **Radar charts**: Multi-dimensional model comparison
- **Box plots**: Distribution analysis across scenarios
- **Time series plots**: Prediction vs. actual trajectories
- **Pattern-specific analysis**: Performance by workload type

### Interactive HTML Reports

```bash
# Generate comprehensive HTML report
python scripts/train-models.py --html-report
```

Features:
- Interactive performance dashboards
- Pattern-specific model recommendations
- Computational efficiency analysis
- Deployment feasibility assessment

## Model Selection Guidelines

### Production Deployment Recommendations

**High-Performance Requirements:**
- **Primary**: XGBoost or LightGBM
- **Alternative**: CatBoost for categorical features
- **Fallback**: GBDT for stability

**Resource-Constrained Environments:**
- **Primary**: Baseline model
- **Alternative**: Prophet with minimal features
- **Consideration**: VAR for multivariate dependencies

**Pattern-Specific Optimization:**
- **Seasonal patterns**: Prophet with adaptive seasonality
- **Burst patterns**: XGBoost with burst-aware features
- **Chaotic patterns**: Ensemble of gradient boosting models
- **Growing patterns**: Models with trend capabilities

### Computational Trade-offs

| Model Class | Training Time | Memory Usage | Accuracy | Deployment Complexity |
|-------------|---------------|--------------|----------|----------------------|
| CPU Models  | 1-10 seconds  | 50-500 MB    | High     | Low                  |
| GPU Models  | 3-15 minutes  | 4-16 GB      | Higher   | Very High            |

## Advanced Configuration

### Hyperparameter Optimization

```python
# Pattern-adaptive parameter selection
if pattern_type == "burst":
    xgb_params["max_depth"] = 8
    xgb_params["learning_rate"] = 0.05
elif pattern_type == "seasonal":
    xgb_params["max_depth"] = 6
    xgb_params["learning_rate"] = 0.1
```

### Custom Feature Engineering

```python
# Extend feature set for domain-specific patterns
def create_custom_features(df):
    # Application-specific indicators
    df['business_critical_hour'] = df['hour'].isin([9, 10, 14, 15])
    
    # Multi-scale rolling features
    for window in [6, 24, 168]:  # Hourly, daily, weekly
        df[f'rolling_quantile_90_{window}'] = df['pod_count'].rolling(window).quantile(0.9)
    
    return df
```

### Production Integration

```python
# Model serving configuration
class ProductionPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.scaler = load_scaler(f"{model_path}/scaler.pkl")
    
    def predict(self, features):
        scaled_features = self.scaler.transform(features)
        prediction = self.model.predict(scaled_features)
        return max(1, round(prediction))  # Minimum 1 pod
```

## Research Applications

### Pattern Taxonomy Validation

The framework validates pattern taxonomy effectiveness by evaluating model performance across different workload types generated by the pattern generation system.

### Computational Efficiency Analysis

Systematic evaluation of computational overhead provides guidance for production deployment decisions in resource-constrained environments.

### Hyperparameter Sensitivity

Automated hyperparameter optimization reveals optimal configurations for different pattern types and computational constraints.

### Model Interpretability

Feature importance analysis and SHAP values provide insights into model decision-making processes for Kubernetes autoscaling applications.

## Contributing

### Adding New Models

1. Create model script in appropriate directory (`cpu-models/` or `gpu-models/`)
2. Implement standardized interface with required methods
3. Add model to `get_available_models()` function
4. Update documentation and examples

### Custom Evaluation Metrics

```python
def custom_sla_violation_metric(y_true, y_pred):
    """Calculate SLA violation rate for custom business logic."""
    violations = np.sum(y_pred < y_true * 0.9)  # 90% SLA threshold
    return violations / len(y_true)
```

## Citation and References

This framework supports research in predictive Kubernetes autoscaling and contributes to the broader field of time series forecasting in cloud computing environments. The comprehensive evaluation methodology and pattern-specific analysis provide valuable insights for both academic research and practical deployment considerations.