"""
On/Off Pattern Generator

Implements on/off patterns with binary state transitions
according to the mathematical formulation in the research paper.
"""

import numpy as np
import pandas as pd
from .base_pattern import BasePattern


class OnOffPattern(BasePattern):
    """
    On/Off Pattern Generator
    
    Mathematical formulation:
    P_t^{onoff} = {P_high + ΔN_t^high if S_t = 1
                  {P_low + ΔN_t^low  if S_t = 0
    
    Where:
    S_t = 1 if (t mod (T_on + T_off)) < T_on, 0 otherwise
    
    - P_high: pod count during 'on' state
    - P_low: pod count during 'off' state
    - T_on: duration of 'on' state
    - T_off: duration of 'off' state
    - Δ: transition dynamics
    - N_t: state-specific noise
    """
    
    def __init__(self,
                 days: int = 28,
                 freq: str = '15min',
                 seed: int = 42,
                 min_pods: float = 5,
                 max_pods: float = 30,
                 on_duration: float = 8,
                 off_duration: float = 16,
                 transition_type: str = 'sharp',
                 transition_noise: float = 0.5,
                 **kwargs):
        """
        Initialize on/off pattern generator.
        
        Args:
            min_pods: Pod count during 'off' state (P_low in formula)
            max_pods: Pod count during 'on' state (P_high in formula)
            on_duration: Duration of 'on' state in hours (T_on in formula)
            off_duration: Duration of 'off' state in hours (T_off in formula)
            transition_type: 'sharp' for binary switching, 'smooth' for gradual
            transition_noise: Noise level during transitions
        """
        super().__init__(days, freq, seed, **kwargs)
        self.min_pods = min_pods
        self.max_pods = max_pods
        self.on_duration = on_duration
        self.off_duration = off_duration
        self.transition_type = transition_type
        self.transition_noise = transition_noise
        
    def _calculate_state(self, time_index: np.ndarray) -> np.ndarray:
        """
        Calculate binary state S_t according to formula.
        
        Args:
            time_index: Array of time indices
            
        Returns:
            Binary state array (1 for 'on', 0 for 'off')
        """
        intervals_per_hour = 4  # 15-minute intervals
        T_on = self.on_duration * intervals_per_hour
        T_off = self.off_duration * intervals_per_hour
        cycle_length = T_on + T_off
        
        # S_t = 1 if (t mod (T_on + T_off)) < T_on
        return ((time_index % cycle_length) < T_on).astype(int)
    
    def generate(self) -> pd.DataFrame:
        """Generate on/off pattern data."""
        np.random.seed(self.seed)
        
        # Create time index
        date_range = self._create_time_index()
        n = len(date_range)
        time_index = np.arange(n)
        
        # Calculate binary state
        state = self._calculate_state(time_index)
        
        # Initialize with minimum pod count
        pod_counts = np.full(n, self.min_pods, dtype=float)
        
        if self.transition_type == 'sharp':
            # Sharp transitions: binary switching
            pod_counts = np.where(state == 1, self.max_pods, self.min_pods)
        else:
            # Smooth transitions: gradual changes
            intervals_per_hour = 4
            transition_duration = 2 * intervals_per_hour  # 2 hours transition
            
            for i in range(1, n):
                if state[i] != state[i-1]:  # State change detected
                    # Find transition start
                    transition_start = i
                    is_on = state[i] == 1
                    
                    for j in range(transition_duration):
                        if transition_start + j < n:
                            progress = j / transition_duration
                            if is_on:
                                # Transition to 'on' state
                                pod_counts[transition_start + j] = (
                                    self.min_pods + 
                                    (self.max_pods - self.min_pods) * progress
                                )
                            else:
                                # Transition to 'off' state
                                pod_counts[transition_start + j] = (
                                    self.max_pods - 
                                    (self.max_pods - self.min_pods) * progress
                                )
                else:
                    # No state change, use current state value
                    if i >= transition_duration:  # After transition period
                        pod_counts[i] = self.max_pods if state[i] == 1 else self.min_pods
        
        # Add transition noise
        for i in range(1, n):
            if abs(pod_counts[i] - pod_counts[i-1]) > 1:  # Transition point
                noise_value = np.random.normal(
                    0, 
                    self.transition_noise * (self.max_pods - self.min_pods)
                )
                pod_counts[i] += noise_value
        
        # Apply constraints
        pod_counts = self._apply_constraints(pod_counts)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'pod_count': pod_counts
        })
    
    def get_mathematical_formula(self) -> str:
        """Return the mathematical formula for on/off patterns."""
        return r"P_t^{onoff} = P_{high} \cdot S_t + P_{low} \cdot (1-S_t) + \Delta_t + N_t" 