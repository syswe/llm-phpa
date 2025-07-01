"""
Burst Pattern Generator

Implements burst patterns with configurable decay functions
according to the mathematical formulation in the research paper.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from .base_pattern import BasePattern


class BurstPattern(BasePattern):
    """
    Burst Pattern Generator
    
    Mathematical formulation:
    P_t^{burst} = B + Σ B_i · g(t - t_i, d_i) · 1_{t_i ≤ t < t_i+d_i} + S·sin(2πh_t/24) + N_t
    
    Where:
    - B: baseline level
    - B_i: magnitude of burst i
    - g(t - t_i, d_i): decay function
    - t_i: start time of burst i
    - d_i: duration of burst i
    - S: seasonal strength
    - N_t: noise component
    
    Decay functions:
    - Linear: g(t, d) = 1 - t/d
    - Exponential: g(t, d) = e^{-λt}
    - Step: g(t, d) = 1 (constant)
    """
    
    def __init__(self,
                 days: int = 28,
                 freq: str = '15min',
                 seed: int = 42,
                 base_pods: float = 15,
                 burst_frequency: float = 0.01,
                 burst_magnitude: float = 20,
                 burst_duration: float = 24,
                 burst_shape: str = 'exponential_decay',
                 seasonal_strength: float = 2,
                 noise_level: float = 0.5,
                 **kwargs):
        """
        Initialize burst pattern generator.
        
        Args:
            base_pods: Baseline pod count (B in formula)
            burst_frequency: Probability of burst starting at any point
            burst_magnitude: Average burst height (B_i in formula)
            burst_duration: Maximum burst duration in hours (d_i in formula)
            burst_shape: 'exponential_decay', 'linear_decay', or 'step_decay'
            seasonal_strength: Seasonal amplitude (S in formula)
            noise_level: Standard deviation of noise component
        """
        super().__init__(days, freq, seed, **kwargs)
        self.base_pods = base_pods
        self.burst_frequency = burst_frequency
        self.burst_magnitude = burst_magnitude
        self.burst_duration = burst_duration
        self.burst_shape = burst_shape
        self.seasonal_strength = seasonal_strength
        self.noise_level = noise_level
        
    def _decay_function(self, t_rel: float, duration: float) -> float:
        """
        Calculate decay function g(t - t_i, d_i).
        
        Args:
            t_rel: Relative time since burst start
            duration: Burst duration
            
        Returns:
            Decay factor
        """
        if self.burst_shape == 'exponential_decay':
            # Exponential decay: g(t, d) = e^{-λt} where λ = 3/d
            return np.exp(-3.0 * t_rel / duration)
        elif self.burst_shape == 'linear_decay':
            # Linear decay: g(t, d) = 1 - t/d
            return max(0, 1 - t_rel / duration)
        elif self.burst_shape == 'step_decay':
            # Step decay: g(t, d) = 1 (constant for duration)
            return 1.0
        else:
            raise ValueError(f"Unknown burst shape: {self.burst_shape}")
    
    def _generate_bursts(self, n: int) -> List[Dict]:
        """Generate burst events."""
        np.random.seed(self.seed)
        bursts = []
        
        for i in range(n):
            if np.random.random() < self.burst_frequency:
                # Burst magnitude with variation (70-130% of base magnitude)
                height = self.burst_magnitude * (0.7 + 0.6 * np.random.random())
                
                # Burst duration in intervals (15-minute intervals)
                duration = np.random.randint(4, int(self.burst_duration * 4))
                
                bursts.append({
                    'start_time': i,
                    'height': height,
                    'duration': duration
                })
        
        return bursts
    
    def generate(self) -> pd.DataFrame:
        """Generate burst pattern data."""
        np.random.seed(self.seed)
        
        # Create time index
        date_range = self._create_time_index()
        n = len(date_range)
        
        # Base level (B in formula)
        pod_counts = np.full(n, self.base_pods, dtype=float)
        
        # Generate and apply bursts: Σ B_i · g(t - t_i, d_i) · 1_{t_i ≤ t < t_i+d_i}
        bursts = self._generate_bursts(n)
        
        for burst in bursts:
            t_i = burst['start_time']
            B_i = burst['height']
            d_i = burst['duration']
            
            for t in range(n):
                if t_i <= t < t_i + d_i:  # Indicator function
                    t_rel = t - t_i
                    g_value = self._decay_function(t_rel, d_i)
                    pod_counts[t] += B_i * g_value
        
        # Add seasonal component: S·sin(2πh_t/24)
        seasonal = self._add_seasonality(
            date_range, 
            strength=self.seasonal_strength,
            period_hours=24
        )
        pod_counts += seasonal
        
        # Add noise (N_t in formula)
        pod_counts = self._add_noise(pod_counts, self.noise_level)
        
        # Apply constraints
        pod_counts = self._apply_constraints(pod_counts)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'pod_count': pod_counts
        })
    
    def get_mathematical_formula(self) -> str:
        """Return the mathematical formula for burst patterns."""
        return (r"P_t^{burst} = B + \sum_{i \in \text{bursts}} B_i \cdot g(t - t_i, d_i) \cdot "
               r"\mathbf{1}_{t_i \leq t < t_i+d_i} + S \sin\left(\frac{2\pi h_t}{24}\right) + N_t") 