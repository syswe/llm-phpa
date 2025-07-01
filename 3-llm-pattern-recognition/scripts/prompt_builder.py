"""
Prompt Builder Module for Enhanced PHPA Benchmark

This module creates sophisticated system and user prompts based on the comprehensive
pattern taxonomy and mathematical formulations for Kubernetes workload analysis.

The prompts incorporate:
- Mathematical formulations from pattern taxonomy
- Real-world application examples
- Detailed characteristic analysis frameworks
- Both text-based and visual-based analysis approaches
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PatternDefinition:
    """Comprehensive pattern definition with mathematical formulation."""
    name: str
    description: str
    mathematical_formulation: str
    key_characteristics: List[str]
    real_world_examples: List[str]
    distinguishing_features: List[str]
    recommended_model: str

class PromptBuilder:
    """Sophisticated prompt builder for pattern recognition and model recommendation."""
    
    def __init__(self, config):
        self.config = config
        self.pattern_definitions = self._load_pattern_definitions()
        self.best_models = self._load_best_models()
    
    def _load_best_models(self) -> Dict[str, str]:
        """Load best model recommendations based on empirical evaluation."""
        return {
            "seasonal": "gbdt",
            "growing": "var", 
            "burst": "gbdt",
            "onoff": "catboost",
            "chaotic": "gbdt",
            "stepped": "gbdt"
        }
    
    def _load_pattern_definitions(self) -> Dict[str, PatternDefinition]:
        """Load comprehensive pattern definitions based on mathematical taxonomy."""
        
        definitions = {
            "seasonal": PatternDefinition(
                name="seasonal",
                description="Predictable Cyclic Behaviors with Regular Periodicity",
                mathematical_formulation="""
                P_t^{seasonal} = B + ∑_{k=1}^{K} A_k sin(2πt/T_k + φ_k) + N_t
                
                Where:
                - B: baseline level
                - A_k: amplitude of k-th harmonic component  
                - T_k: period of k-th cycle (daily=24h, weekly=168h, monthly=720h)
                - φ_k: phase shift
                - N_t: stochastic noise component
                """,
                key_characteristics=[
                    "Regular, predictable cycles with consistent periodicity",
                    "Smooth oscillations around stable baseline", 
                    "Repeating peaks and troughs at consistent intervals",
                    "Multiple harmonic components (daily, weekly, monthly)",
                    "Low coefficient of variation in cycle timing",
                    "Strong autocorrelation at cycle periods"
                ],
                real_world_examples=[
                    "E-commerce platforms with business hour peaks",
                    "Business intelligence applications with weekday patterns", 
                    "Customer relationship management systems with monthly cycles",
                    "Educational platforms with semester-based activity"
                ],
                distinguishing_features=[
                    "Fourier analysis reveals dominant frequency components",
                    "Autocorrelation function shows periodic peaks",
                    "Spectral density concentrated at specific frequencies", 
                    "Predictable variance within cycles"
                ],
                recommended_model="gbdt"
            ),
            
            "growing": PatternDefinition(
                name="growing",
                description="Progressive Resource Demand Evolution with Sustained Trends",
                mathematical_formulation="""
                P_t^{growing} = B + G·f(t) + S·sin(2πh_t/24) + N_t
                
                Where:
                - G: growth rate parameter
                - f(t): growth function (linear, exponential, logarithmic)
                - S: seasonal modulation amplitude
                - N_t: noise component
                """,
                key_characteristics=[
                    "Clear overall upward trend over time",
                    "Sustained directional movement with positive slope",
                    "May include superimposed seasonal oscillations",
                    "Increasing baseline levels over observation period",
                    "Growth rate may be linear, exponential, or logarithmic"
                ],
                real_world_examples=[
                    "Startup applications during user acquisition",
                    "SaaS platforms experiencing customer growth",
                    "Video streaming during content releases",
                    "IoT analytics with scaling deployments"
                ],
                distinguishing_features=[
                    "Linear regression shows significant positive slope",
                    "Detrended series reveals underlying patterns",
                    "Moving averages show upward trajectory",
                    "Growth acceleration may change over time"
                ],
                recommended_model="var"
            ),
            
            "burst": PatternDefinition(
                name="burst",
                description="Transient High-Intensity Events with Rapid Onset and Decay",
                mathematical_formulation="""
                P_t^{burst} = B + ∑_{i∈bursts} B_i·g(t-t_i, d_i)·𝟙_{t_i ≤ t < t_i+d_i} + S·sin(2πh_t/24) + N_t
                
                Where:
                - B_i: magnitude of burst i
                - t_i: burst initiation time  
                - d_i: burst duration
                - g(t,d): decay function
                - 𝟙: indicator function
                
                Decay Functions:
                - Linear: g(t,d) = 1 - t/d
                - Exponential: g(t,d) = e^{-λt}
                - Step: g(t,d) = 1 for discrete intervals
                """,
                key_characteristics=[
                    "Stable baseline with sudden, short-lived spikes",
                    "High-magnitude events with rapid onset characteristics", 
                    "Spikes decay relatively quickly back to baseline",
                    "Irregular timing of burst events (Poisson-like)",
                    "High peak-to-baseline ratios",
                    "Transient nature with limited duration"
                ],
                real_world_examples=[
                    "Flash sale events in e-commerce platforms",
                    "Viral content propagation in social media systems",
                    "Breaking news traffic in media applications",
                    "Financial trading platforms during market events",
                    "Gaming services during tournament or release events",
                    "Emergency response systems during crisis events"
                ],
                distinguishing_features=[
                    "High kurtosis indicating heavy-tailed distribution",
                    "Sudden gradient changes exceeding 2σ threshold",
                    "Burst detection algorithms identify discrete events",
                    "Spectral analysis shows broadband noise characteristics",
                    "Change point detection reveals rapid level shifts"
                ],
                recommended_model="gbdt"
            ),
            
            "onoff": PatternDefinition(
                name="onoff", 
                description="Binary State Transitions with Distinct Activity Levels",
                mathematical_formulation="""
                P_t^{onoff} = {P_high + Δ_t^{transition} + N_t^{high}  if S_t = 1
                              {P_low + Δ_t^{transition} + N_t^{low}   if S_t = 0
                
                Where:
                - S_t ∈ {0,1}: binary state at time t
                - P_high, P_low: characteristic levels for each state
                - Δ_t^{transition}: transition dynamics
                - N_t^{high/low}: state-specific noise
                
                State Transition:
                S_t = {1 if (t mod (T_on + T_off)) < T_on
                      {0 otherwise
                """,
                key_characteristics=[
                    "Alternates between two distinct, flat levels",
                    "Sharp transitions resembling square wave patterns",
                    "Minimal time spent in intermediate states",
                    "Predictable state durations or scheduled transitions",
                    "Low variance within each state",
                    "High contrast between 'on' and 'off' levels"
                ],
                real_world_examples=[
                    "Batch processing applications with scheduled execution windows",
                    "Backup systems operating during off-peak hours",
                    "Business applications with strict business hour operation",
                    "Data analytics pipelines with distinct processing phases",
                    "Scientific computing with computational vs. preparation phases",
                    "CDN traffic patterns following global timezone shifts"
                ],
                distinguishing_features=[
                    "Bimodal distribution with clear separation between states",
                    "Step change detection identifies transition points",
                    "State duration analysis reveals operational patterns",
                    "Minimal variance within states, high between states",
                    "Threshold-based classification achieves high accuracy"
                ],
                recommended_model="catboost"
            ),
            
            "chaotic": PatternDefinition(
                name="chaotic",
                description="Irregular Unpredictable Behaviors with High Stochasticity",
                mathematical_formulation="""
                P_t^{chaotic} = B + S·sin(2πh_t/24) + ∑_{i=1}^T T_i·𝟙_{t≥c_i}·(t-c_i) + ∑_{j∈spikes} S_j·exp(-(t-t_j)²/2σ_j²) + N_t^{high}
                
                Where:
                - T_i: slope of trend segment i
                - c_i: change points (Poisson process)
                - S_j: spike magnitudes
                - t_j: spike occurrence times
                - σ_j: spike width parameters
                - N_t^{high}: high-variance noise
                """,
                key_characteristics=[
                    "Highly irregular with unpredictable direction changes",
                    "Multiple trend changes and random spike injection",
                    "High coefficient of variation",
                    "Lacks clear repeating patterns or trends",
                    "Frequent changes in level, direction, and variance",
                    "Random walk-like behavior with drift components"
                ],
                real_world_examples=[
                    "Applications with volatile user behavior patterns",
                    "Experimental features with unpredictable adoption",
                    "Systems subject to external disruptions",
                    "Social media during unpredictable trending events",
                    "Development and testing environments",
                    "Multi-tenant platforms with diverse customer bases"
                ],
                distinguishing_features=[
                    "High entropy and low predictability metrics",
                    "Frequent change point detection triggers", 
                    "Spectral analysis shows white noise characteristics",
                    "Lyapunov exponents indicate chaotic dynamics",
                    "Traditional forecasting models perform poorly",
                    "High residual variance in fitted models"
                ],
                recommended_model="gbdt"
            ),
            
            "stepped": PatternDefinition(
                name="stepped",
                description="Discrete Level Transitions with Plateau Characteristics", 
                mathematical_formulation="""
                P_t^{stepped} = B_base + L_t·S_step + S·sin(2πh_t/24) + N_t^{transition} + N_t
                
                Where:
                - B_base: baseline level
                - L_t: current step level at time t
                - S_step: step magnitude
                - N_t^{transition}: transition-specific noise
                
                Step Level Function:
                L_t = ⌊(t-t_0)/T_step⌋ mod N_steps
                """,
                key_characteristics=[
                    "Distinct plateaus with sudden transitions between levels",
                    "Discrete, quantized levels rather than continuous changes",
                    "Abrupt jumps followed by stable periods",
                    "Predictable level progression or cyclic stepping",
                    "Low variance within each plateau",
                    "Clear segmentation into discrete operational phases"
                ],
                real_world_examples=[
                    "Phased application rollouts with gradual user migration",
                    "Infrastructure capacity planning with scheduled upgrades",
                    "Cloud migration projects with discrete transition phases", 
                    "Load testing scenarios with incremental intensity increases",
                    "Version deployments with staged release processes",
                    "Resource allocation adjustments in planned increments"
                ],
                distinguishing_features=[
                    "Change point detection identifies level transitions",
                    "Histogram shows clear peaks at discrete levels",
                    "Segmented regression fits better than continuous models",
                    "Low intra-segment variance, high inter-segment variance",
                    "Plateau detection algorithms identify stable periods"
                ],
                recommended_model="gbdt"
            )
        }
        
        return definitions
    
    def build_text_prompt(self, file_path: str, expected_pattern: Optional[str] = None) -> str:
        """Build sophisticated text-based analysis prompt."""
        
        # Read and preprocess CSV data
        csv_content = self._read_and_preprocess_csv(file_path)
        
        # Build comprehensive system prompt
        system_prompt = self._build_text_system_prompt()
        
        # Build user prompt with data
        user_prompt = f"""
Analyze the following Kubernetes pod scaling time-series dataset for pattern recognition and forecasting model recommendation.

Dataset Information:
- File: {os.path.basename(file_path)}
- Format: timestamp, pod_count
- Temporal Resolution: 15-minute intervals
- Domain: Kubernetes Horizontal Pod Autoscaling (HPA)

**Complete Dataset Content:**
```csv
{csv_content}
```

**Analysis Requirements:**

1. **Mathematical Pattern Analysis**: Apply the mathematical framework described in the system instructions to identify the underlying pattern type from the six categories.

2. **Statistical Characterization**: Analyze the following quantitative metrics:
   - Trend analysis: Linear regression slope and R²
   - Seasonality detection: Autocorrelation and spectral analysis indicators  
   - Variance analysis: Coefficient of variation and stability metrics
   - Event detection: Burst identification and change point analysis
   - Distributional properties: Skewness, kurtosis, and modality

3. **Temporal Dependency Assessment**: Evaluate:
   - Short-term dependencies (1-4 lags)
   - Medium-term patterns (daily cycles: 96 data points)
   - Long-term trends (weekly patterns: 672 data points)

4. **Production Context Evaluation**: Consider:
   - Predictability requirements for autoscaling decisions
   - Computational constraints for real-time inference
   - Model robustness under operational variability

**Output Format (Strictly Required):**

**Mathematical Analysis:**
[Detailed analysis of mathematical characteristics observed in the data, referencing specific formulations from the pattern taxonomy]

**Pattern Classification:**
[One of: seasonal, growing, burst, onoff, chaotic, stepped]

**Statistical Evidence:** 
[Quantitative metrics supporting the classification decision]

**Model Recommendation:**
[Recommended forecasting model based on pattern identification]

**Confidence Assessment:**
[High/Medium/Low confidence with justification]

**Operational Considerations:**
[Brief assessment of deployment feasibility and expected performance]
"""
        
        return system_prompt + "\n\n" + user_prompt
    
    def build_visual_prompt(self, file_path: str, expected_pattern: Optional[str] = None) -> str:
        """Build sophisticated visual-based analysis prompt with chart generation."""
        
        # Generate visualization
        chart_b64 = self._generate_chart(file_path)
        
        # Build visual system prompt
        system_prompt = self._build_visual_system_prompt()
        
        # Build user prompt with chart
        user_prompt = f"""
Analyze the following Kubernetes pod scaling time-series visualization for pattern recognition and forecasting model recommendation.

Dataset Information:
- File: {os.path.basename(file_path)}
- Visualization: Time-series plot with trend analysis
- Domain: Kubernetes Horizontal Pod Autoscaling (HPA)
- Temporal Resolution: 15-minute intervals

**Time-Series Visualization:**
[Chart shows pod count over time with statistical overlays]

**Visual Analysis Framework:**

1. **Pattern Recognition via Visual Inspection**:
   - Identify dominant visual characteristics using the mathematical taxonomy
   - Assess regularity, trend direction, and variance patterns
   - Detect seasonal cycles, growth trends, or irregular behaviors

2. **Mathematical Validation through Visual Cues**:
   - Seasonal: Look for regular oscillations with consistent amplitude/period
   - Growing: Identify sustained upward/downward directional movement  
   - Burst: Spot sudden spikes with rapid decay to baseline
   - OnOff: Recognize binary state transitions with sharp level changes
   - Chaotic: Observe irregular, unpredictable movement patterns
   - Stepped: Detect discrete level transitions with plateau phases

3. **Operational Assessment**:
   - Evaluate predictability from visual pattern clarity
   - Assess forecasting complexity based on pattern characteristics
   - Consider autoscaling implementation challenges

**Output Format (Strictly Required):**

**Visual Pattern Analysis:**
[Detailed description of visual patterns observed, referencing mathematical formulations]

**Pattern Classification:**  
[One of: seasonal, growing, burst, onoff, chaotic, stepped]

**Visual Evidence:**
[Specific visual characteristics supporting the classification]

**Model Recommendation:**
[Recommended forecasting model based on visual pattern analysis]

**Confidence Assessment:**
[High/Medium/Low confidence with visual justification]

**Forecasting Complexity:**
[Assessment of pattern predictability and modeling challenges]

Please analyze the visualization and provide your assessment following this framework.
"""
        
        return system_prompt + "\n\n" + user_prompt
    
    def _build_text_system_prompt(self) -> str:
        """Build comprehensive text-based system prompt."""
        
        prompt = """You are an expert in Predictive Horizontal Pod Autoscaling (PHPA) for Kubernetes environments, specializing in advanced time-series pattern recognition and forecasting model selection. Your expertise encompasses mathematical analysis of temporal patterns, statistical time series modeling, and operational considerations for production Kubernetes deployments.

**CORE MISSION**: Analyze time-series datasets representing Kubernetes pod usage metrics to identify underlying patterns from a scientifically-grounded taxonomy and recommend optimal forecasting models based on empirical evaluation results.

**MATHEMATICAL PATTERN TAXONOMY**:

Your analysis must be grounded in the following comprehensive mathematical framework for Kubernetes workload patterns:

"""
        
        # Add detailed pattern definitions
        for pattern_name, definition in self.pattern_definitions.items():
            prompt += f"""
**{definition.name.upper()} PATTERNS - {definition.description}**

Mathematical Formulation:
{definition.mathematical_formulation}

Key Characteristics:
{chr(10).join(f'• {char}' for char in definition.key_characteristics)}

Real-World Applications:
{chr(10).join(f'• {example}' for example in definition.real_world_examples)}

Distinguishing Features:
{chr(10).join(f'• {feature}' for feature in definition.distinguishing_features)}

Recommended Model: {definition.recommended_model}

---
"""
        
        prompt += """

**ANALYSIS METHODOLOGY**:

1. **Quantitative Pattern Recognition**: Apply mathematical formulations to identify pattern signatures in the provided dataset. Look for specific mathematical characteristics that distinguish each pattern type.

2. **Statistical Validation**: Employ statistical tests and metrics to validate pattern classification:
   - Seasonal: Autocorrelation analysis, spectral density examination
   - Growing: Trend analysis via linear regression, unit root tests
   - Burst: Change point detection, outlier analysis
   - OnOff: State detection via threshold analysis, bimodal distribution tests
   - Chaotic: Entropy calculation, Lyapunov exponent estimation  
   - Stepped: Segmented regression, plateau detection algorithms

3. **Model Selection Framework**: Based on comprehensive empirical evaluation across 600 synthetic scenarios, the optimal model recommendations have been established through systematic performance analysis.

4. **Production Constraints**: Consider computational efficiency, training time, memory usage, and inference latency requirements for Kubernetes environments.

**CRITICAL REQUIREMENTS**:

- Base analysis EXCLUSIVELY on the provided time-series data
- Apply mathematical formulations to validate pattern identification  
- Provide quantitative evidence supporting classification decisions
- Reference specific characteristics from the taxonomy
- Recommend models based on established performance benchmarks
- Maintain scientific rigor in analysis and conclusions
"""
        
        return prompt
    
    def _build_visual_system_prompt(self) -> str:
        """Build comprehensive visual-based system prompt."""
        
        prompt = """You are an expert in visual time-series analysis for Predictive Horizontal Pod Autoscaling (PHPA) in Kubernetes environments. Your specialized expertise includes pattern recognition through visual inspection, mathematical validation of visual patterns, and forecasting model selection based on visual characteristics.

**CORE MISSION**: Analyze time-series visualizations of Kubernetes pod scaling data to identify patterns through visual inspection and recommend optimal forecasting models based on visual pattern characteristics.

**VISUAL PATTERN RECOGNITION FRAMEWORK**:

Your visual analysis must be grounded in the following mathematical taxonomy, adapted for visual pattern recognition:

"""
        
        # Add visual-focused pattern definitions
        for pattern_name, definition in self.pattern_definitions.items():
            prompt += f"""
**{definition.name.upper()} PATTERNS - Visual Signatures**

Visual Characteristics:
{chr(10).join(f'• {char}' for char in definition.key_characteristics)}

Visual Mathematical Indicators:
{definition.mathematical_formulation.split('Where:')[0].strip()}

Visual Distinguishing Features:
{chr(10).join(f'• {feature}' for feature in definition.distinguishing_features)}

Expected Visual Examples:
{chr(10).join(f'• {example}' for example in definition.real_world_examples[:3])}

Recommended Model: {definition.recommended_model}

---
"""
        
        prompt += """

**VISUAL ANALYSIS METHODOLOGY**:

1. **Pattern Recognition via Visual Inspection**:
   - Examine overall shape and trajectory characteristics
   - Identify dominant visual patterns (cycles, trends, events)
   - Assess regularity vs. irregularity in temporal behavior
   - Evaluate amplitude, frequency, and phase characteristics

2. **Mathematical Validation through Visual Cues**:
   - Seasonal: Regular oscillatory patterns with consistent period/amplitude
   - Growing: Clear directional trend with sustained upward/downward movement
   - Burst: Distinctive spike patterns with rapid onset and decay
   - OnOff: Binary-like transitions between distinct operational levels
   - Chaotic: Irregular, seemingly random movements with high variability
   - Stepped: Discrete level changes with stable plateau periods

3. **Forecasting Complexity Assessment**:
   - Evaluate pattern predictability from visual regularity
   - Assess modeling challenges based on pattern complexity
   - Consider computational requirements for pattern types

4. **Operational Context Integration**:
   - Relate visual patterns to real-world Kubernetes scenarios
   - Assess autoscaling implementation feasibility
   - Consider production deployment characteristics

**VISUAL ANALYSIS REQUIREMENTS**:

- Focus on overall pattern shape and characteristics
- Identify mathematical signatures visible in the data
- Validate pattern classification through visual evidence
- Consider forecasting model appropriateness for observed patterns
- Provide confidence assessment based on visual clarity
- Reference mathematical formulations where visually apparent
"""
        
        return prompt
    
    def _read_and_preprocess_csv(self, file_path: str) -> str:
        """Read and preprocess CSV for optimal token usage."""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            if 'ds' in df.columns and 'y' in df.columns:
                df = df.rename(columns={'ds': 'timestamp', 'y': 'pod_count'})
            
            # Preprocess timestamps for token efficiency
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp_short'] = df['timestamp'].dt.strftime('%m-%d %H:%M')
            
            # Create compressed CSV content
            compressed_df = df[['timestamp_short', 'pod_count']].copy()
            compressed_df.columns = ['timestamp', 'pod_count']
            
            # Limit rows if too large
            if len(compressed_df) > 500:
                step = len(compressed_df) // 400  # Sample to ~400 points
                compressed_df = compressed_df.iloc[::step]
            
            return compressed_df.to_csv(index=False)
            
        except Exception as e:
            logger.error(f"Error preprocessing CSV {file_path}: {e}")
            return ""
    
    def _generate_chart(self, file_path: str) -> str:
        """Generate time-series visualization chart."""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize columns
            if 'ds' in df.columns and 'y' in df.columns:
                df = df.rename(columns={'ds': 'timestamp', 'y': 'pod_count'})
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['pod_count'], linewidth=1.5, alpha=0.8, color='#2E86AB')
            
            # Add trend line
            from scipy import stats
            x_numeric = range(len(df))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df['pod_count'])
            trend_line = [slope * x + intercept for x in x_numeric]
            plt.plot(df['timestamp'], trend_line, '--', color='red', alpha=0.7, label=f'Trend (R²={r_value**2:.3f})')
            
            # Formatting
            plt.title('Kubernetes Pod Count Time Series Analysis', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Pod Count', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_b64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()
            
            return img_b64
            
        except Exception as e:
            logger.error(f"Error generating chart for {file_path}: {e}")
            return "" 