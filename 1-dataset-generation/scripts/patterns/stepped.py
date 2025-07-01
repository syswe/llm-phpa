"""
Stepped Pattern Generator

Implements stepped patterns with discrete level transitions
according to the mathematical formulation in the research paper.
"""

import numpy as np
import pandas as pd
from .base_pattern import BasePattern


class SteppedPattern(BasePattern):
    """
    Stepped Pattern Generator
    
    Mathematical formulation:
    P_t^{stepped} = B_{base} + L_t · S_{step} + S·sin(2πh_t/24) + N_t^{transition} + N_t
    
    Where:
    L_t = floor((t - t_0)/T_step) mod N_steps
    
    - B_base: baseline level
    - L_t: current step level at time t
    - S_step: step magnitude
    - T_step: step duration
    - N_steps: total number of discrete levels
    - S: seasonal strength
    - N_t: noise components
    """
    
    def __init__(self,
                 days: int = 28,
                 freq: str = '15min',
                 seed: int = 42,
                 min_pods: float = 10,
                 max_pods: float = 40,
                 step_count: int = 4,
                 step_duration: float = 48,
                 step_type: str = 'equal',
                 seasonal_strength: float = 1,
                 transition_noise: float = 0.5,
                 noise_level: float = 0.3,
                 **kwargs):
        """
        Initialize stepped pattern generator.
        
        Args:
            min_pods: Minimum pod count
            max_pods: Maximum pod count
            step_count: Number of distinct steps (N_steps in formula)
            step_duration: Duration of each step in hours (T_step in formula)
            step_type: 'equal' for uniform steps, 'random' for variable heights
            seasonal_strength: Seasonal amplitude (S in formula)
            transition_noise: Noise during transitions (N_t^{transition})
            noise_level: General noise level (N_t)
        """
        super().__init__(days, freq, seed, **kwargs)
        self.min_pods = min_pods
        self.max_pods = max_pods
        self.step_count = step_count
        self.step_duration = step_duration
        self.step_type = step_type
        self.seasonal_strength = seasonal_strength
        self.transition_noise = transition_noise
        self.noise_level = noise_level
        
    def _calculate_step_levels(self) -> np.ndarray:
        """Calculate step level values."""
        np.random.seed(self.seed)
        
        if self.step_type == 'equal':
            # Equal step sizes
            return np.linspace(self.min_pods, self.max_pods, self.step_count + 1)
        else:  # random
            # Random step sizes between min and max
            step_sizes = np.sort(np.random.uniform(
                self.min_pods, self.max_pods, self.step_count + 1
            ))
            step_sizes[0] = self.min_pods
            step_sizes[-1] = self.max_pods
            return step_sizes
    
    def _calculate_step_function(self, time_index: np.ndarray) -> np.ndarray:
        """
        Calculate step function L_t according to formula.
        
        Args:
            time_index: Array of time indices
            
        Returns:
            Step level indices
        """
        intervals_per_hour = 4  # 15-minute intervals
        T_step = self.step_duration * intervals_per_hour
        t_0 = 0  # Pattern initiation time
        
        # L_t = floor((t - t_0)/T_step) mod N_steps
        return np.floor((time_index - t_0) / T_step).astype(int) % (self.step_count + 1)
    
    def generate(self) -> pd.DataFrame:
        """Generate stepped pattern data."""
        np.random.seed(self.seed)
        
        # Create time index
        date_range = self._create_time_index()
        n = len(date_range)
        time_index = np.arange(n)
        
        # Calculate step sizes
        step_sizes = self._calculate_step_levels()
        
        # Calculate step function L_t
        step_indices = self._calculate_step_function(time_index)
        
        # Apply step levels: B_base + L_t · S_step
        pod_counts = step_sizes[step_indices]
        
        # Add seasonal component: S·sin(2πh_t/24)
        seasonal = self._add_seasonality(
            date_range,
            strength=self.seasonal_strength,
            period_hours=24
        )
        pod_counts += seasonal
        
        # Add transition noise (N_t^{transition})
        for i in range(1, n):
            if abs(pod_counts[i] - pod_counts[i-1]) > 1:  # Transition point
                noise_value = np.random.normal(
                    0, 
                    self.transition_noise * (self.max_pods - self.min_pods) / self.step_count
                )
                pod_counts[i] += noise_value
        
        # Add general noise (N_t)
        pod_counts = self._add_noise(pod_counts, self.noise_level)
        
        # Apply constraints
        pod_counts = self._apply_constraints(pod_counts)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'pod_count': pod_counts
        })
    
    def get_mathematical_formula(self) -> str:
        """Return the mathematical formula for stepped patterns."""
        return (r"P_t^{stepped} = B_{base} + L_t \cdot S_{step} + "
               r"S \sin\left(\frac{2\pi h_t}{24}\right) + N_t^{transition} + N_t") 