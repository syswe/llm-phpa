"""
Seasonal Pattern Generator

Implements seasonal patterns with configurable cyclical behaviors
according to the mathematical formulation in the research paper.
"""

import numpy as np
import pandas as pd
from .base_pattern import BasePattern


class SeasonalPattern(BasePattern):
    """
    Seasonal Pattern Generator
    
    Mathematical formulation:
    P_t^{seasonal} = B + Σ A_k sin(2πt/T_k + φ_k) + N_t
    
    Where:
    - B: baseline level
    - A_k: amplitude of k-th harmonic component  
    - T_k: period of k-th cycle
    - φ_k: phase shift
    - N_t: noise component
    """
    
    def __init__(self, 
                 days: int = 28,
                 freq: str = '15min',
                 seed: int = 42,
                 base_level: float = 30,
                 seasonality_strength: float = 1.0,
                 seasonality_period: float = 24,
                 complexity: int = 1,
                 noise_level: float = 0.5,
                 **kwargs):
        """
        Initialize seasonal pattern generator.
        
        Args:
            base_level: Baseline pod count (B in formula)
            seasonality_strength: Amplitude multiplier for seasonal components
            seasonality_period: Primary seasonality period in hours
            complexity: 1=single seasonality, 2=multiple seasonalities  
            noise_level: Standard deviation of noise component
        """
        super().__init__(days, freq, seed, **kwargs)
        self.base_level = base_level
        self.seasonality_strength = seasonality_strength
        self.seasonality_period = seasonality_period
        self.complexity = complexity
        self.noise_level = noise_level
        
    def generate(self) -> pd.DataFrame:
        """Generate seasonal pattern data."""
        np.random.seed(self.seed)
        
        # Create time index
        date_range = self._create_time_index()
        n = len(date_range)
        
        # Base level (B in formula)
        pod_counts = np.full(n, self.base_level, dtype=float)
        
        # Primary seasonal component
        hours = date_range.hour + date_range.minute / 60
        primary_seasonal = (self.seasonality_strength * 10 * 
                          np.sin(hours * (2 * np.pi / self.seasonality_period)))
        pod_counts += primary_seasonal
        
        # Additional harmonics for complexity >= 2
        if self.complexity >= 2:
            # Weekly component (T_k = 168 hours = 1 week)
            weekly = 5 * np.sin(date_range.dayofweek * (2 * np.pi / 7))
            
            # Monthly component (T_k = 720 hours ≈ 30 days)  
            monthly = 7 * np.sin(date_range.day * (2 * np.pi / 30))
            
            pod_counts += weekly + monthly
            
        # Add noise (N_t in formula)
        pod_counts = self._add_noise(pod_counts, self.noise_level)
        
        # Apply constraints
        pod_counts = self._apply_constraints(pod_counts)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'pod_count': pod_counts
        })
    
    def get_mathematical_formula(self) -> str:
        """Return the mathematical formula for seasonal patterns."""
        if self.complexity == 1:
            return r"P_t^{seasonal} = B + A \sin\left(\frac{2\pi h_t}{T}\right) + N_t"
        else:
            return (r"P_t^{seasonal} = B + A_{daily} \sin\left(\frac{2\pi h_t}{24}\right) + "
                   r"A_{weekly} \sin\left(\frac{2\pi d_t}{7}\right) + "
                   r"A_{monthly} \sin\left(\frac{2\pi d_t}{30}\right) + N_t") 