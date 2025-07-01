# Enhanced LLM Pattern Recognition Benchmark for PHPA

A comprehensive, modular benchmark system for evaluating Large Language Model capabilities in Kubernetes workload pattern recognition and forecasting model recommendation using existing datasets from the 1-dataset-generation module.

## Overview

This enhanced benchmark framework provides systematic evaluation of multiple LLM architectures for Predictive Horizontal Pod Autoscaling (PHPA) pattern recognition tasks. The system supports multiple LLM providers, dual analysis methods (text-based and visual-based), and generates comprehensive performance reports using pre-generated datasets from the comprehensive dataset generation module.

### Key Features

- **Multi-LLM Support**: Gemini 2.5 Pro, Qwen3, and Grok-3 APIs
- **Dual Analysis Methods**: Text-based CSV analysis and visual chart analysis
- **Mathematical Pattern Taxonomy**: Six scientifically-grounded Kubernetes workload patterns
- **Sophisticated Prompting**: Advanced system prompts incorporating mathematical formulations
- **Comprehensive Evaluation**: Detailed accuracy metrics and performance analysis
- **Modular Architecture**: Clean separation of concerns with extensible design
- **Rich Reporting**: HTML reports with interactive charts and heatmaps
- **Dataset Integration**: Uses high-quality synthetic datasets from the 1-dataset-generation module

## Architecture

The benchmark system follows a modular architecture with clear separation of responsibilities:

```
enhanced_benchmark.py          # Main orchestrator
├── llm_providers.py          # Multi-LLM API integration
├── prompt_builder.py         # Sophisticated prompt generation
├── pattern_generator.py      # Synthetic dataset creation
├── evaluator.py             # Performance evaluation & metrics
├── visualizer.py            # Report & chart generation
└── config.py               # Configuration management
```

### System Components

#### 1. LLM Providers (`llm_providers.py`)
- **Unified Interface**: Abstract base class for consistent LLM integration
- **Multiple APIs**: Support for Gemini, Qwen, and Grok with provider-specific optimizations
- **Fault Tolerance**: Retry logic, timeout handling, and graceful error management
- **Configuration Management**: Flexible API key and parameter configuration

#### 2. Pattern Loader (`pattern_generator.py`)
- **Dataset Integration**: Loads existing datasets from the 1-dataset-generation module
- **Six Pattern Types**: Seasonal, Growing, Burst, On/Off, Chaotic, Stepped
- **Format Standardization**: Ensures consistent data formats for LLM analysis
- **Visualization Support**: Automatic chart generation for visual analysis

#### 3. Prompt Builder (`prompt_builder.py`)
- **Sophisticated Prompts**: Advanced system prompts incorporating mathematical taxonomy
- **Dual-Modal Support**: Specialized prompts for text and visual analysis
- **Pattern Definitions**: Comprehensive mathematical formulations and characteristics
- **Research-Grade Quality**: Prompts designed for rigorous academic evaluation

#### 4. Evaluation System (`evaluator.py`)
- **Response Parsing**: Robust extraction of pattern predictions and model recommendations
- **Accuracy Metrics**: Pattern recognition and model recommendation accuracy
- **Performance Analysis**: Comparative analysis across LLMs, patterns, and methods
- **Statistical Validation**: Comprehensive metrics for research validation

#### 5. Visualization (`visualizer.py`)
- **Rich Reports**: HTML reports with embedded charts and interactive elements
- **Performance Heatmaps**: Pattern-specific performance visualization
- **Comparison Charts**: LLM and method comparison visualizations
- **Export Capabilities**: Chart export for academic publication

## Pattern Taxonomy

The benchmark evaluates recognition of six fundamental Kubernetes workload patterns:

### 1. Seasonal Patterns
- **Mathematical Model**: `P_t = B + ∑A_k sin(2πt/T_k + φ_k) + N_t`
- **Characteristics**: Regular cyclic behaviors with predictable periodicity
- **Examples**: E-commerce daily cycles, business application weekly patterns
- **Recommended Model**: GBDT

### 2. Growing Patterns  
- **Mathematical Model**: `P_t = B + G·f(t) + S·sin(2πh_t/24) + N_t`
- **Characteristics**: Sustained upward trends with optional seasonal components
- **Examples**: Startup user acquisition, SaaS platform growth
- **Recommended Model**: VAR

### 3. Burst Patterns
- **Mathematical Model**: `P_t = B + ∑B_i·g(t-t_i,d_i)·𝟙 + S·sin(2πh_t/24) + N_t`
- **Characteristics**: Sudden high-intensity spikes with rapid decay
- **Examples**: Flash sales, viral content, breaking news events
- **Recommended Model**: GBDT

### 4. On/Off Patterns
- **Mathematical Model**: `P_t = {P_high + N_t^high if S_t=1; P_low + N_t^low if S_t=0}`
- **Characteristics**: Binary state transitions between distinct activity levels
- **Examples**: Batch processing, backup systems, business hour applications
- **Recommended Model**: CatBoost

### 5. Chaotic Patterns
- **Mathematical Model**: `P_t = B + S·sin(2πh_t/24) + ∑T_i·𝟙_{t≥c_i}·(t-c_i) + ∑S_j·exp(-(t-t_j)²/2σ_j²) + N_t`
- **Characteristics**: Highly irregular, unpredictable behaviors
- **Examples**: Development environments, experimental features
- **Recommended Model**: GBDT

### 6. Stepped Patterns
- **Mathematical Model**: `P_t = B_base + L_t·S_step + S·sin(2πh_t/24) + N_t`
- **Characteristics**: Discrete level transitions with plateau phases
- **Examples**: Phased rollouts, infrastructure capacity planning
- **Recommended Model**: GBDT

## Installation

### Prerequisites
- Python 3.8+
- Required packages: `pandas`, `numpy`, `matplotlib`, `requests`, `pyyaml`, `scipy`
- **Pre-generated datasets from 1-dataset-generation module**

### Setup
```bash
# 1. First, generate datasets using the 1-dataset-generation module
cd 1-dataset-generation/scripts
python generate_patterns.py --output-dir test_output_all

# 2. Setup LLM pattern recognition module
cd ../../3-llm-pattern-recognition

# Install dependencies
pip install -r requirements.txt

# Create configuration file
python scripts/config.py

# Configure API keys
export GEMINI_API_KEY="your_gemini_key"
export QWEN_API_KEY="your_qwen_key"
export GROK_API_KEY="your_grok_key"
```

### Configuration

Edit `config.yaml` to customize benchmark settings:

```yaml
# LLM Selection
enabled_llms: ['gemini', 'qwen', 'grok']
analysis_methods: ['text', 'visual']

# Dataset Configuration
samples_per_pattern: 3
max_rows_per_example: 500
pattern_types: ['seasonal', 'growing', 'burst', 'onoff', 'chaotic', 'stepped']

# API Configuration
gemini_config:
  api_key: ${GEMINI_API_KEY}
  model_name: 'gemini-2.0-flash'
  max_tokens: 1000
  temperature: 0.1

qwen_config:
  api_key: ${QWEN_API_KEY}
  model_name: 'qwen-turbo'
  max_tokens: 1000
  temperature: 0.1

grok_config:
  api_key: ${GROK_API_KEY}
  model_name: 'grok-beta'
  max_tokens: 1000
  temperature: 0.1
```

## Usage

### Basic Benchmark Execution

**Note**: Ensure you have generated datasets using the 1-dataset-generation module first.

```bash
# Full benchmark with all LLMs and methods
python scripts/enhanced_benchmark.py --llm all --method all

# Specific LLM evaluation
python scripts/enhanced_benchmark.py --llm gemini --method text

# Load datasets only (for verification)
python scripts/enhanced_benchmark.py --generate-only

# Test single file
python scripts/enhanced_benchmark.py --test-file data.csv --llm qwen --method visual
```

### Advanced Usage

```bash
# Custom configuration
python scripts/enhanced_benchmark.py --config custom_config.yaml

# Increased sample size
python scripts/enhanced_benchmark.py --samples-per-pattern 5

# Custom output directory
python scripts/enhanced_benchmark.py --output-dir my_results

# Verbose logging
python scripts/enhanced_benchmark.py --verbose
```

## Output and Results

### Generated Artifacts

The benchmark generates comprehensive outputs in the specified results directory:

```
benchmark_results/
├── benchmark_results_20241201_143022.json    # Raw results data
├── benchmark_report_20241201_143022.html     # Comprehensive HTML report
├── accuracy_chart_20241201_143022.png        # Performance overview chart
├── performance_heatmap_20241201_143022.png   # Pattern-specific heatmap
└── pattern-examples/                         # Loaded datasets (from 1-dataset-generation)
    ├── seasonal/
    ├── growing/
    ├── burst/
    ├── onoff/
    ├── chaotic/
    └── stepped/
```

### Performance Metrics

The system provides comprehensive performance analysis:

- **Overall Accuracy**: Pattern recognition and model recommendation accuracy by LLM
- **Pattern-Specific Performance**: Accuracy breakdown by individual pattern types
- **Method Comparison**: Text-based vs. visual-based analysis effectiveness
- **LLM Comparison**: Relative strengths and weaknesses across providers
- **Difficulty Analysis**: Pattern complexity assessment based on recognition rates

### Sample Results

Based on preliminary evaluation (representative results):

| LLM | Text Accuracy | Visual Accuracy | Overall Performance |
|-----|---------------|-----------------|-------------------|
| Gemini 2.5 Pro | 95.0% | 98.3% | 96.7% |
| Qwen3 | 58.3% | 62.5% | 60.4% |
| Grok-3 | 45.8% | 75.0% | 60.4% |

**Pattern Difficulty Ranking** (based on recognition accuracy):
1. Seasonal (easiest) - Regular mathematical patterns
2. Growing - Clear trend identification
3. Stepped - Discrete level detection
4. On/Off - Binary state recognition
5. Burst - Event detection complexity
6. Chaotic (hardest) - Irregular behavior analysis

## Research Applications

This benchmark framework supports academic research in several areas:

### 1. LLM Capability Assessment
- Quantitative reasoning evaluation in infrastructure domains
- Multimodal analysis capability comparison
- Pattern recognition performance benchmarking

### 2. Prompt Engineering Research
- Mathematical formulation incorporation effectiveness
- Domain-specific prompt optimization
- Cross-modal prompt design analysis

### 3. Infrastructure Intelligence
- Automated pattern recognition feasibility
- Human expert vs. AI capability comparison
- Production deployment readiness assessment

### 4. Comparative Studies
- Cross-provider performance analysis
- Method effectiveness evaluation
- Pattern complexity characterization

## Extending the Framework

### Adding New LLM Providers

1. Create provider class inheriting from `LLMProvider`
2. Implement required abstract methods
3. Add provider to `LLMProviderFactory`
4. Update configuration schema

```python
class NewLLMProvider(LLMProvider):
    def _validate_config(self):
        # Provider-specific validation
        pass
    
    def _prepare_request(self, prompt: str):
        # API request formatting
        pass
    
    def _extract_response(self, response_data: dict):
        # Response parsing
        pass
```

### Adding New Patterns

1. Define pattern in `PatternDefinition` format
2. Implement generation function in `PatternGenerator`
3. Add mathematical formulation and characteristics
4. Update configuration and documentation

### Custom Evaluation Metrics

1. Extend `BenchmarkEvaluator` class
2. Implement additional metrics calculation
3. Update visualization components
4. Modify report generation