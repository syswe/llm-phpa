# Kubernetes Workload Pattern Generation for Predictive Autoscaling Research

This repository contains the pattern generation system for **"Predictive Horizontal Pod Autoscaling"** research, implementing a comprehensive taxonomy of Kubernetes workload behaviors for systematic evaluation of autoscaling algorithms.

## 🎯 Research Context

This dataset generation framework supports predictive autoscaling research by creating realistic workload patterns that challenge traditional reactive autoscaling approaches. The system generates six fundamental pattern types that collectively represent the spectrum of workload characteristics encountered in production Kubernetes deployments.

### Key Research Contributions
- **Comprehensive Pattern Taxonomy**: Six mathematically-formulated pattern types covering the full spectrum of production workload behaviors
- **Statistical Rigor**: Over 2 million data points across 600 distinct scenarios for robust algorithm evaluation
- **Real-world Calibration**: Patterns validated against NASA web servers, FIFA World Cup datasets, and cloud application logs
- **Temporal Consistency**: 15-minute granularity over 35-day periods with realistic Kubernetes constraints

## 📊 Pattern Taxonomy

Our taxonomy encompasses six fundamental pattern types, each addressing specific temporal behaviors that challenge autoscaling systems:

### 1. **Seasonal Patterns** - Predictable Cyclic Behaviors
**Mathematical Formulation:**
```
P_t^{seasonal} = B + Σ A_k sin(2πt/T_k + φ_k) + N_t
```

**Characteristics:**
- Regular, predictable cycles with consistent periodicity
- Multiple harmonic components (daily, weekly, monthly)
- Applications: E-commerce platforms, business applications, CRM systems

**Real-world Examples:**
- E-commerce traffic with business hour peaks and weekend surges
- Business intelligence systems with weekday usage patterns
- Customer service platforms with monthly reporting cycles

### 2. **Growing Patterns** - Progressive Resource Demand Evolution
**Mathematical Formulation:**
```
P_t^{growing} = B + G·f(t) + S·sin(2πh_t/24) + N_t
```

**Characteristics:**
- Sustained upward trends over time
- Linear or exponential growth functions
- Combined with seasonal modulation

**Real-world Examples:**
- Startup applications during user acquisition phases
- Video streaming services during content releases
- IoT platforms with expanding device deployment

### 3. **Burst Patterns** - Transient High-Intensity Events
**Mathematical Formulation:**
```
P_t^{burst} = B + Σ B_i·g(t-t_i,d_i)·1_{t_i≤t<t_i+d_i} + S·sin(2πh_t/24) + N_t
```

**Characteristics:**
- Stable baseline with sudden, short-lived spikes
- Configurable decay functions (linear, exponential, step)
- Unpredictable timing and magnitude

**Real-world Examples:**
- Flash sale events in e-commerce
- Viral content propagation in social media
- Financial trading platforms during market events

### 4. **On/Off Patterns** - Binary State Transitions
**Mathematical Formulation:**
```
P_t^{onoff} = {P_high + ΔN_t^high if S_t = 1
              {P_low + ΔN_t^low  if S_t = 0
```

**Characteristics:**
- Alternating between distinct high/low activity states
- Sharp or smooth transitions
- Scheduled operational cycles

**Real-world Examples:**
- Batch processing applications with scheduled windows
- Backup systems during off-peak hours
- Business applications with strict operating hours

### 5. **Chaotic Patterns** - Irregular Unpredictable Behaviors
**Mathematical Formulation:**
```
P_t^{chaotic} = B + S·sin(2πh_t/24) + Σ T_i·1_{t≥c_i}·(t-c_i) + Σ S_j·exp(-(t-t_j)²/2σ_j²) + N_t^{high}
```

**Characteristics:**
- Highly irregular with unpredictable changes
- Multiple trend changes and random spikes
- High-variance noise components

**Real-world Examples:**
- Development/testing environments with irregular activity
- Social media during trending events
- Multi-tenant platforms with diverse usage patterns

### 6. **Stepped Patterns** - Discrete Level Transitions
**Mathematical Formulation:**
```
P_t^{stepped} = B_base + L_t·S_step + S·sin(2πh_t/24) + N_t^{transition} + N_t
```

**Characteristics:**
- Distinct plateaus with sudden level transitions
- Planned capacity changes
- Discrete operational phases

**Real-world Examples:**
- Phased application rollouts
- Infrastructure capacity planning events
- Load testing with incremental intensity increases

## 🏗️ System Architecture

```
1-dataset-generation/
├── scripts/
│   ├── generate_patterns.py          # Main orchestration script
│   ├── patterns/                     # Pattern implementation modules
│   │   ├── base_pattern.py          # Abstract base class with common functionality
│   │   ├── seasonal.py              # Seasonal pattern implementation
│   │   ├── growing.py               # Growing pattern implementation  
│   │   ├── burst.py                 # Burst pattern implementation
│   │   ├── onoff.py                 # On/Off pattern implementation
│   │   ├── stepped.py               # Stepped pattern implementation
│   │   └── chaotic.py               # Chaotic pattern implementation
│   ├── utils/                        # Utility modules
│   │   ├── data_utils.py           # Data processing and validation
│   │   └── plot_utils.py           # Visualization utilities
│   └── config/                       # Configuration management
│       ├── pattern_configs.py      # Parameter variations for each pattern
│       └── constants.py            # System constants and defaults
```

### 🔧 Key Features

- **Modular Architecture**: Each pattern type in separate, extensible modules
- **Mathematical Rigor**: Implementations based on formal mathematical models
- **Comprehensive Parameterization**: 10 scenarios × 10 variations per pattern type
- **Statistical Validation**: Autocorrelation analysis, spectral density, stationarity tests
- **Temporal Consistency**: 15-minute intervals, 35-day duration, realistic constraints
- **Production Calibration**: Validated against real-world workload traces

## 🚀 Usage

### Quick Start - Generate All Patterns
```bash
# Generate complete dataset (600 patterns across all types)
cd scripts/
python generate_patterns.py --output-dir ./complete_dataset --days 35 --train-days 28

# Output: 600 pattern datasets with comprehensive metadata and visualizations
```

### Pattern-Specific Generation
```bash
# Generate only seasonal patterns
python generate_patterns.py --pattern-type seasonal --output-dir ./seasonal_only

# Generate with custom parameters
python generate_patterns.py --days 42 --train-days 35 --log-level DEBUG
```

### Programmatic Usage
```python
from patterns import SeasonalPattern, BurstPattern
from utils import DataProcessor, PatternPlotter

# Create a custom seasonal pattern
pattern = SeasonalPattern(
    days=28,
    base_level=30,
    seasonality_strength=1.5,
    complexity=2,  # Multiple seasonalities
    noise_level=0.3
)

# Generate and analyze
df = pattern.generate()
features = DataProcessor.calculate_pattern_features(df)
formula = pattern.get_mathematical_formula()

print(f"Mathematical Formula: {formula}")
print(f"Pattern Statistics: {features}")
```

## 📈 Generated Datasets

### Dataset Specifications
- **Temporal Resolution**: 15-minute intervals (matching Kubernetes HPA)
- **Duration**: 35 days total (28 training + 7 testing)
- **Data Points**: 3,360 per pattern (2,688 training + 672 testing)
- **Pod Range**: 1-100 pods (realistic Kubernetes constraints)
- **Format**: CSV files with timestamp and pod_count columns

### Output Structure
```
output_directory/
├── [pattern]_[number]_full.csv      # Complete dataset
├── [pattern]_[number]_train.csv     # Training split (80%)
├── [pattern]_[number]_test.csv      # Testing split (20%)
├── [pattern]_[number]_plot.png      # Visualization with statistics
├── patterns_metadata.json           # Comprehensive metadata
├── pattern_summary.png             # Cross-pattern comparison
└── pattern_generation.log          # Detailed generation logs
```

### Statistical Features Extracted
- **Basic Statistics**: mean, std, min, max, variance, coefficient of variation
- **Temporal Analysis**: trend slopes, autocorrelation, seasonality detection
- **Volatility Measures**: change variance, maximum change, mean absolute change
- **Distribution Properties**: percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)

## 🔬 Research Applications

### Algorithm Evaluation Framework
The generated datasets enable systematic evaluation of autoscaling algorithms across:

1. **Predictability Spectrum**: From highly predictable seasonal patterns to chaotic behaviors
2. **Scaling Challenges**: Various magnitudes of change and temporal dynamics
3. **Temporal Patterns**: Different time scales and operational characteristics
4. **Real-world Scenarios**: Production-validated behavioral patterns

### Performance Metrics
- **Prediction Accuracy**: RMSE, MAE, MAPE across different prediction horizons
- **Scaling Efficiency**: Resource utilization, over/under-provisioning metrics
- **Temporal Robustness**: Performance across different pattern characteristics
- **Adaptation Speed**: Response to pattern changes and anomalies

## 🛠️ Technical Requirements

- **Python**: 3.8+
- **Dependencies**: pandas, numpy, matplotlib, seaborn
- **Memory**: ~2GB for complete dataset generation
- **Storage**: ~500MB for all 600 generated patterns
- **Compute**: ~30 minutes for complete generation on standard hardware

## 📊 Validation and Quality Assurance

### Statistical Validation
- **Temporal Consistency**: Regular interval validation, missing data detection
- **Range Constraints**: Pod count boundaries, scaling velocity limits
- **Pattern Fidelity**: Mathematical formula compliance, parameter validation
- **Noise Characteristics**: Gaussian distribution validation, autocorrelation analysis

### Real-world Calibration
- **Production Traces**: NASA web server logs, FIFA World Cup datasets
- **Cloud Applications**: Contemporary SaaS platform metrics
- **Statistical Alignment**: Distribution matching, temporal pattern correlation

## 🤝 Contributing

The pattern generation framework supports extensible development:

1. **New Pattern Types**: Implement `BasePattern` interface for additional patterns
2. **Parameter Exploration**: Extend `PatternConfigurations` for new scenarios
3. **Validation Methods**: Add statistical tests in `DataProcessor`
4. **Visualization**: Enhance plotting capabilities in `PatternPlotter`

## 📚 Related Works

This framework builds upon and validates against:
- **Kubernetes HPA**: Official horizontal pod autoscaling mechanisms
- **Production Traces**: NASA HTTP logs, FIFA World Cup datasets
- **Cloud Computing Research**: ARIMA, LSTM, and Prophet forecasting approaches
- **Autoscaling Literature**: Reactive and predictive scaling methodologies

---

**Framework Version**: 2.0  
**Research Phase**: Pattern Generation and Validation  
**Last Updated**: 2025  
**Maintainer**: [Canberk Duman and Suleyman Eken]
