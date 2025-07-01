"""
Chaotic Pattern Generator

Implements chaotic patterns with irregular unpredictable behaviors
according to the mathematical formulation in the research paper.
"""

import numpy as np
import pandas as pd
from .base_pattern import BasePattern


class ChaoticPattern(BasePattern):
    """
    Chaotic Pattern Generator
    
    Mathematical formulation:
    P_t^{chaotic} = B + S·sin(2πh_t/24) + Σ T_i·1_{t≥c_i}·(t-c_i) + Σ S_j·exp(-(t-t_j)²/2σ_j²) + N_t^{high}
    
    Where:
    - B: baseline level
    - S: seasonal strength  
    - T_i: slope of trend segment i
    - c_i: change points
    - S_j: spike magnitudes
    - t_j: spike times
    - σ_j: spike width parameters
    - N_t^{high}: high-variance noise
    """
    
    def __init__(self,
                 days: int = 28,
                 freq: str = '15min',
                 seed: int = 42,
                 base_pods: float = 40,
                 chaos_level: float = 1.0,
                 seasonality_strength: float = 0.5,
                 trend_changes: int = 3,
                 spike_density: float = 0.005,
                 noise_level: float = 0.8,
                 **kwargs):
        """
        Initialize chaotic pattern generator.
        
        Args:
            base_pods: Baseline pod count (B in formula)
            chaos_level: Intensity of chaotic behavior
            seasonality_strength: Seasonal amplitude (S in formula)
            trend_changes: Number of major trend shifts
            spike_density: Density of Gaussian spikes
            noise_level: High-variance noise level (N_t^{high})
        """
        super().__init__(days, freq, seed, **kwargs)
        self.base_pods = base_pods
        self.chaos_level = chaos_level
        self.seasonality_strength = seasonality_strength
        self.trend_changes = trend_changes
        self.spike_density = spike_density
        self.noise_level = noise_level
        
    def _generate_trend_changes(self, n: int) -> tuple:
        """Generate trend change points and slopes."""
        master_rng = np.random.RandomState(self.seed)
        
        # Generate change points (c_i in formula)
        num_change_points = self.trend_changes * 3  # More change points for chaos
        week_segments = n // 7
        change_points = []
        
        for week in range(self.days // 7 + 1):
            week_start = week * week_segments
            week_end = min((week + 1) * week_segments, n)
            
            if week_end > week_start:
                num_points = master_rng.randint(3, 7)
                week_points = master_rng.choice(
                    range(week_start, week_end),
                    size=min(num_points, week_end - week_start),
                    replace=False
                )
                change_points.extend(week_points)
        
        change_points = sorted(change_points)
        
        # Generate trend slopes (T_i in formula)
        trends = master_rng.uniform(-0.25, 0.25, len(change_points) + 1) * self.chaos_level
        
        return change_points, trends
    
    def _generate_gaussian_spikes(self, n: int) -> np.ndarray:
        """
        Generate Gaussian spikes: Σ S_j·exp(-(t-t_j)²/2σ_j²)
        
        Args:
            n: Number of time points
            
        Returns:
            Spike component array
        """
        spike_rng = np.random.RandomState(self.seed + 2)
        
        # Generate spike parameters
        num_spikes = int(n * self.spike_density * self.chaos_level)
        spike_times = spike_rng.choice(range(n), size=num_spikes, replace=False)
        
        spike_component = np.zeros(n)
        
        for t_j in spike_times:
            # Spike magnitude (S_j in formula)
            spike_magnitude = spike_rng.uniform(15, 40) * self.chaos_level
            if spike_rng.random() < 0.5:
                spike_magnitude *= -1  # Negative spike
            
            # Spike width parameter (σ_j in formula)
            sigma_j = spike_rng.uniform(2, 8)
            
            # Apply Gaussian spike: S_j·exp(-(t-t_j)²/2σ_j²)
            for t in range(n):
                gaussian_value = np.exp(-((t - t_j) ** 2) / (2 * sigma_j ** 2))
                spike_component[t] += spike_magnitude * gaussian_value
        
        return spike_component
    
    def generate(self) -> pd.DataFrame:
        """Generate chaotic pattern data."""
        master_rng = np.random.RandomState(self.seed)
        
        # Create time index
        date_range = self._create_time_index()
        n = len(date_range)
        
        # Base level with random fluctuations (B in formula)
        base_fluctuation = master_rng.uniform(-10, 10, n) * (self.chaos_level * 0.3)
        pod_counts = np.full(n, self.base_pods, dtype=float) + base_fluctuation
        
        # Add distorted seasonality: S·sin(2πh_t/24)
        hours = date_range.hour + date_range.minute / 60
        seasonal_distortion = master_rng.uniform(0.7, 1.3, n)
        seasonal = (self.seasonality_strength * 5 * 
                   np.sin(hours * (2 * np.pi / 24)) * seasonal_distortion)
        pod_counts += seasonal
        
        # Add trend changes: Σ T_i·1_{t≥c_i}·(t-c_i)
        change_points, trends = self._generate_trend_changes(n)
        
        current_trend_idx = 0
        last_change_point = 0
        
        for i in range(n):
            if current_trend_idx < len(change_points) and i >= change_points[current_trend_idx]:
                last_change_point = change_points[current_trend_idx]
                current_trend_idx += 1
            
            # Apply trend: T_i·(t-c_i)
            pod_counts[i] += trends[current_trend_idx] * (i - last_change_point) * 1.5
        
        # Add Gaussian spikes: Σ S_j·exp(-(t-t_j)²/2σ_j²)
        spike_component = self._generate_gaussian_spikes(n)
        pod_counts += spike_component
        
        # Add sudden jumps for additional chaos
        jump_rng = np.random.RandomState(self.seed + 1)
        num_jumps = jump_rng.randint(3, 6)
        jump_points = jump_rng.choice(range(n), size=num_jumps, replace=False)
        
        for jump_point in jump_points:
            jump_size = jump_rng.choice([-1, 1]) * jump_rng.randint(15, 35) * self.chaos_level
            decay_length = jump_rng.randint(50, 200)
            
            for j in range(decay_length):
                if jump_point + j < n:
                    decay = max(0, 1 - (j / decay_length) ** 0.5)
                    pod_counts[jump_point + j] += jump_size * decay
        
        # Add high-variance noise (N_t^{high})
        noise_rng = np.random.RandomState(self.seed + 3)
        noise = noise_rng.normal(0, self.noise_level * self.chaos_level * 1.5, n)
        
        # Add extreme noise bursts
        for i in range(n):
            if noise_rng.random() < 0.03:
                noise[i] *= 3
        
        pod_counts += noise
        
        # Apply constraints
        pod_counts = self._apply_constraints(pod_counts)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'pod_count': pod_counts
        })
    
    def get_mathematical_formula(self) -> str:
        """Return the mathematical formula for chaotic patterns."""
        return (r"P_t^{chaotic} = B + S \sin\left(\frac{2\pi h_t}{24}\right) + "
               r"\sum_{i=1}^{T} T_i \cdot \mathbf{1}_{t \geq c_i} \cdot (t - c_i) + "
               r"\sum_{j \in \text{spikes}} S_j \cdot \exp\left(-\frac{(t - t_j)^2}{2\sigma_j^2}\right) + N_t^{high}") 