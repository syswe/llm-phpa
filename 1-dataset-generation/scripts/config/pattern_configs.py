"""
Pattern Configuration Management

This module defines parameter variations for different pattern types
to ensure comprehensive coverage of the behavioral space.
"""

from typing import Dict, List, Any


class PatternConfigurations:
    """Configuration manager for pattern generation parameters."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.seasonal_configs = self._get_seasonal_variations()
        self.growing_configs = self._get_growing_variations()
        self.burst_configs = self._get_burst_variations()
        self.onoff_configs = self._get_onoff_variations()
        self.stepped_configs = self._get_stepped_variations()
        self.chaotic_configs = self._get_chaotic_variations()
    
    def _get_seasonal_variations(self) -> List[Dict[str, Any]]:
        """Get seasonal pattern parameter variations."""
        return [
            # Standard daily seasonality variations
            {'base_level': 30, 'seasonality_strength': 1.0, 'complexity': 1, 'noise_level': 0.3},
            {'base_level': 25, 'seasonality_strength': 1.5, 'complexity': 1, 'noise_level': 0.4},
            {'base_level': 35, 'seasonality_strength': 0.8, 'complexity': 1, 'noise_level': 0.2},
            
            # Multi-seasonal patterns
            {'base_level': 40, 'seasonality_strength': 1.2, 'complexity': 2, 'noise_level': 0.3},
            {'base_level': 20, 'seasonality_strength': 1.0, 'complexity': 2, 'noise_level': 0.5},
            
            # Different periods
            {'base_level': 30, 'seasonality_strength': 1.0, 'seasonality_period': 12, 'noise_level': 0.3},
            {'base_level': 30, 'seasonality_strength': 1.0, 'seasonality_period': 48, 'noise_level': 0.3},
            
            # High/low noise variations
            {'base_level': 30, 'seasonality_strength': 1.0, 'complexity': 1, 'noise_level': 0.1},
            {'base_level': 30, 'seasonality_strength': 1.0, 'complexity': 1, 'noise_level': 0.8},
            {'base_level': 30, 'seasonality_strength': 1.0, 'complexity': 2, 'noise_level': 0.6}
        ]
    
    def _get_growing_variations(self) -> List[Dict[str, Any]]:
        """Get growing pattern parameter variations."""
        return [
            # Linear growth variations - clear growth patterns
            {'start_level': 15, 'target_level': 75, 'growth_type': 'linear', 'seasonality_strength': 1.0, 'cyclic_growth': False, 'noise_level': 0.6},
            {'start_level': 10, 'target_level': 60, 'growth_type': 'linear', 'seasonality_strength': 0.8, 'cyclic_growth': False, 'noise_level': 0.5},
            {'start_level': 20, 'target_level': 80, 'growth_type': 'linear', 'seasonality_strength': 1.2, 'cyclic_growth': False, 'noise_level': 0.7},
            {'start_level': 25, 'target_level': 50, 'growth_type': 'linear', 'seasonality_strength': 0.6, 'cyclic_growth': False, 'noise_level': 0.4},
            
            # Exponential growth variations
            {'start_level': 12, 'target_level': 65, 'growth_type': 'exponential', 'seasonality_strength': 1.0, 'cyclic_growth': False, 'noise_level': 0.6},
            {'start_level': 18, 'target_level': 70, 'growth_type': 'exponential', 'seasonality_strength': 0.7, 'cyclic_growth': False, 'noise_level': 0.8},
            {'start_level': 8, 'target_level': 55, 'growth_type': 'exponential', 'seasonality_strength': 0.9, 'cyclic_growth': False, 'noise_level': 0.5},
            
            # Cyclic growth variations
            {'start_level': 20, 'target_level': 75, 'growth_type': 'linear', 'seasonality_strength': 1.0, 'cyclic_growth': True, 'noise_level': 0.6},
            {'start_level': 15, 'target_level': 60, 'growth_type': 'linear', 'seasonality_strength': 1.5, 'cyclic_growth': True, 'noise_level': 0.7},
            
            # Moderate growth with different noise levels
            {'start_level': 30, 'target_level': 70, 'growth_type': 'linear', 'seasonality_strength': 0.8, 'cyclic_growth': False, 'noise_level': 0.3},
            {'start_level': 25, 'target_level': 65, 'growth_type': 'exponential', 'seasonality_strength': 1.1, 'cyclic_growth': False, 'noise_level': 0.9}
        ]
    
    def _get_burst_variations(self) -> List[Dict[str, Any]]:
        """Get burst pattern parameter variations."""
        return [
            # Standard burst patterns
            {'base_pods': 15, 'burst_frequency': 0.01, 'burst_magnitude': 20, 'burst_shape': 'exponential_decay', 'burst_duration': 24},
            {'base_pods': 20, 'burst_frequency': 0.008, 'burst_magnitude': 25, 'burst_shape': 'exponential_decay', 'burst_duration': 18},
            {'base_pods': 25, 'burst_frequency': 0.012, 'burst_magnitude': 15, 'burst_shape': 'exponential_decay', 'burst_duration': 30},
            
            # Linear decay bursts
            {'base_pods': 18, 'burst_frequency': 0.01, 'burst_magnitude': 22, 'burst_shape': 'linear_decay', 'burst_duration': 20},
            {'base_pods': 22, 'burst_frequency': 0.009, 'burst_magnitude': 18, 'burst_shape': 'linear_decay', 'burst_duration': 25},
            
            # Step decay bursts
            {'base_pods': 20, 'burst_frequency': 0.008, 'burst_magnitude': 20, 'burst_shape': 'step_decay', 'burst_duration': 12},
            {'base_pods': 15, 'burst_frequency': 0.015, 'burst_magnitude': 30, 'burst_shape': 'step_decay', 'burst_duration': 8},
            
            # High frequency variations
            {'base_pods': 25, 'burst_frequency': 0.02, 'burst_magnitude': 12, 'burst_shape': 'exponential_decay', 'burst_duration': 15},
            {'base_pods': 30, 'burst_frequency': 0.025, 'burst_magnitude': 10, 'burst_shape': 'linear_decay', 'burst_duration': 10},
            {'base_pods': 20, 'burst_frequency': 0.005, 'burst_magnitude': 35, 'burst_shape': 'exponential_decay', 'burst_duration': 40}
        ]
    
    def _get_onoff_variations(self) -> List[Dict[str, Any]]:
        """Get on/off pattern parameter variations."""
        return [
            # Standard business hours patterns
            {'min_pods': 5, 'max_pods': 30, 'on_duration': 8, 'off_duration': 16, 'transition_type': 'sharp'},
            {'min_pods': 10, 'max_pods': 40, 'on_duration': 12, 'off_duration': 12, 'transition_type': 'sharp'},
            {'min_pods': 8, 'max_pods': 35, 'on_duration': 6, 'off_duration': 18, 'transition_type': 'sharp'},
            
            # Smooth transitions
            {'min_pods': 15, 'max_pods': 45, 'on_duration': 10, 'off_duration': 14, 'transition_type': 'smooth'},
            {'min_pods': 12, 'max_pods': 38, 'on_duration': 8, 'off_duration': 16, 'transition_type': 'smooth'},
            
            # Asymmetric patterns
            {'min_pods': 5, 'max_pods': 50, 'on_duration': 4, 'off_duration': 20, 'transition_type': 'sharp'},
            {'min_pods': 20, 'max_pods': 60, 'on_duration': 16, 'off_duration': 8, 'transition_type': 'sharp'},
            
            # Different transition noise levels
            {'min_pods': 10, 'max_pods': 30, 'on_duration': 8, 'off_duration': 16, 'transition_type': 'smooth', 'transition_noise': 0.2},
            {'min_pods': 10, 'max_pods': 30, 'on_duration': 8, 'off_duration': 16, 'transition_type': 'smooth', 'transition_noise': 0.8},
            {'min_pods': 15, 'max_pods': 25, 'on_duration': 12, 'off_duration': 12, 'transition_type': 'sharp', 'transition_noise': 0.1}
        ]
    
    def _get_stepped_variations(self) -> List[Dict[str, Any]]:
        """Get stepped pattern parameter variations."""
        return [
            # Equal step patterns
            {'min_pods': 10, 'max_pods': 40, 'step_count': 4, 'step_duration': 48, 'step_type': 'equal'},
            {'min_pods': 15, 'max_pods': 50, 'step_count': 3, 'step_duration': 60, 'step_type': 'equal'},
            {'min_pods': 20, 'max_pods': 60, 'step_count': 5, 'step_duration': 36, 'step_type': 'equal'},
            
            # Random step patterns
            {'min_pods': 10, 'max_pods': 45, 'step_count': 4, 'step_duration': 40, 'step_type': 'random'},
            {'min_pods': 12, 'max_pods': 38, 'step_count': 6, 'step_duration': 30, 'step_type': 'random'},
            
            # Short duration steps
            {'min_pods': 15, 'max_pods': 35, 'step_count': 8, 'step_duration': 24, 'step_type': 'equal'},
            {'min_pods': 20, 'max_pods': 40, 'step_count': 10, 'step_duration': 18, 'step_type': 'random'},
            
            # Different noise levels
            {'min_pods': 10, 'max_pods': 40, 'step_count': 4, 'step_duration': 48, 'step_type': 'equal', 'transition_noise': 0.2},
            {'min_pods': 10, 'max_pods': 40, 'step_count': 4, 'step_duration': 48, 'step_type': 'equal', 'transition_noise': 0.8},
            {'min_pods': 25, 'max_pods': 45, 'step_count': 3, 'step_duration': 72, 'step_type': 'equal', 'seasonal_strength': 2.0}
        ]
    
    def _get_chaotic_variations(self) -> List[Dict[str, Any]]:
        """Get chaotic pattern parameter variations."""
        return [
            # Moderate chaos
            {'base_pods': 40, 'chaos_level': 0.8, 'seasonality_strength': 0.5, 'trend_changes': 3, 'spike_density': 0.003},
            {'base_pods': 35, 'chaos_level': 1.0, 'seasonality_strength': 0.4, 'trend_changes': 4, 'spike_density': 0.005},
            {'base_pods': 45, 'chaos_level': 0.6, 'seasonality_strength': 0.6, 'trend_changes': 2, 'spike_density': 0.004},
            
            # High chaos
            {'base_pods': 50, 'chaos_level': 1.2, 'seasonality_strength': 0.3, 'trend_changes': 6, 'spike_density': 0.008},
            {'base_pods': 30, 'chaos_level': 1.5, 'seasonality_strength': 0.2, 'trend_changes': 8, 'spike_density': 0.010},
            
            # Low chaos with strong seasonality
            {'base_pods': 40, 'chaos_level': 0.4, 'seasonality_strength': 0.8, 'trend_changes': 1, 'spike_density': 0.002},
            {'base_pods': 35, 'chaos_level': 0.5, 'seasonality_strength': 0.7, 'trend_changes': 2, 'spike_density': 0.003},
            
            # Extreme variations
            {'base_pods': 25, 'chaos_level': 2.0, 'seasonality_strength': 0.1, 'trend_changes': 10, 'spike_density': 0.015},
            {'base_pods': 60, 'chaos_level': 0.3, 'seasonality_strength': 1.0, 'trend_changes': 1, 'spike_density': 0.001},
            {'base_pods': 40, 'chaos_level': 1.0, 'seasonality_strength': 0.5, 'trend_changes': 5, 'spike_density': 0.007}
        ]
    
    def get_variations(self, pattern_type: str) -> List[Dict[str, Any]]:
        """
        Get parameter variations for a specific pattern type.
        
        Args:
            pattern_type: Type of pattern
            
        Returns:
            List of parameter variations
        """
        return getattr(self, f"{pattern_type}_configs", [{}])


# Pre-defined pattern variations for easy access
PATTERN_VARIATIONS = {
    'seasonal': PatternConfigurations()._get_seasonal_variations(),
    'growing': PatternConfigurations()._get_growing_variations(),
    'burst': PatternConfigurations()._get_burst_variations(),
    'onoff': PatternConfigurations()._get_onoff_variations(),
    'stepped': PatternConfigurations()._get_stepped_variations(),
    'chaotic': PatternConfigurations()._get_chaotic_variations()
} 