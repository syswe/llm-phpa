"""
Growing Pattern Generator

Implements growing patterns with linear or exponential growth trends
according to the mathematical formulation in the research paper.
"""

import numpy as np
import pandas as pd
from .base_pattern import BasePattern


class GrowingPattern(BasePattern):
    """
    Growing Pattern Generator
    
    Mathematical formulation:
    P_t^{growing} = B + G·f(t) + S·sin(2πh_t/24) + N_t
    
    Where:
    - B: baseline level (starting point)
    - G: growth magnitude (target_level - start_level)
    - f(t): growth function (linear or exponential)
    - S: seasonal strength
    - N_t: noise component
    
    Growth functions:
    - Linear: f(t) = t/T_scale
    - Exponential: f(t) = (e^{αt} - 1)/(e^α - 1)
    """
    
    def __init__(self,
                 days: int = 28,
                 freq: str = '15min', 
                 seed: int = 42,
                 start_level: float = 15,
                 target_level: float = 75,
                 growth_type: str = 'linear',
                 seasonality_strength: float = 1.0,
                 cyclic_growth: bool = False,
                 noise_level: float = 0.8,
                 **kwargs):
        """
        Initialize growing pattern generator.
        
        Args:
            start_level: Starting pod count level (B in formula)
            target_level: Target pod count level to reach by the end
            growth_type: 'linear' or 'exponential' growth function
            seasonality_strength: Seasonal amplitude (S in formula)
            cyclic_growth: If True, seasonal amplitude increases with trend
            noise_level: Standard deviation of noise component
        """
        super().__init__(days, freq, seed, **kwargs)
        self.start_level = start_level
        self.target_level = target_level
        self.growth_magnitude = target_level - start_level
        self.growth_type = growth_type
        self.seasonality_strength = seasonality_strength
        self.cyclic_growth = cyclic_growth
        self.noise_level = noise_level
        
        # Validate parameters
        if self.target_level <= self.start_level:
            raise ValueError("Target level must be greater than start level for growing pattern")
        
    def _calculate_growth_function(self, time_index: np.ndarray) -> np.ndarray:
        """
        Calculate normalized growth function f(t) that goes from 0 to 1.
        
        Args:
            time_index: Array of time indices
            
        Returns:
            Growth function values (0 to 1)
        """
        T_scale = len(time_index) - 1  # Normalization factor
        
        if self.growth_type == 'linear':
            # Linear growth: f(t) = t/T_scale
            return time_index / T_scale
        elif self.growth_type == 'exponential':
            # Exponential growth: f(t) = (e^{αt/T_scale} - 1)/(e^α - 1)
            # α controls the exponential curve steepness
            alpha = 2.0  # Moderate exponential growth
            normalized_t = alpha * time_index / T_scale
            return (np.exp(normalized_t) - 1) / (np.exp(alpha) - 1)
        else:
            raise ValueError(f"Unknown growth type: {self.growth_type}")
    
    def generate(self) -> pd.DataFrame:
        """Generate growing pattern data."""
        np.random.seed(self.seed)
        
        # Create time index
        date_range = self._create_time_index()
        n = len(date_range)
        time_index = np.arange(n)
        
        # Calculate growth component: start_level + growth_magnitude * f(t)
        f_t = self._calculate_growth_function(time_index)
        base_trend = self.start_level + (self.growth_magnitude * f_t)
        
        # Seasonal component: S·sin(2πh_t/24)
        hours = date_range.hour + date_range.minute / 60
        
        if self.cyclic_growth:
            # Seasonal amplitude increases with growth (starts small, grows larger)
            growth_factor = 0.3 + f_t * 0.7  # From 30% to 100% amplitude
            seasonal = (self.seasonality_strength * 4 * 
                       np.sin(hours * (2 * np.pi / 24)) * growth_factor)
        else:
            # Standard seasonal component
            seasonal = (self.seasonality_strength * 3 * 
                       np.sin(hours * (2 * np.pi / 24)))
        
        # Combine components: P_t^{growing} = B + G·f(t) + S·sin(2πh_t/24) + N_t
        pod_counts = base_trend + seasonal
        
        # Add noise (N_t in formula) - slightly more noise for growing patterns
        pod_counts = self._add_noise(pod_counts, self.noise_level)
        
        # Apply constraints (ensure we stay within 0-100 range)
        pod_counts = self._apply_constraints(pod_counts)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'pod_count': pod_counts
        })
    
    def get_mathematical_formula(self) -> str:
        """Return the mathematical formula for growing patterns."""
        if self.growth_type == 'linear':
            return (r"P_t^{growing} = B_{start} + (B_{target} - B_{start}) \cdot \frac{t}{T_{scale}} + "
                   r"S \sin\left(\frac{2\pi h_t}{24}\right) + N_t")
        else:
            return (r"P_t^{growing} = B_{start} + (B_{target} - B_{start}) \cdot \frac{e^{\alpha t/T} - 1}{e^{\alpha} - 1} + "
                   r"S \sin\left(\frac{2\pi h_t}{24}\right) + N_t") 