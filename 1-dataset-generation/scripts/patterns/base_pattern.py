"""
Base Pattern Generator Class

This module provides the base class for all pattern generators, ensuring
consistent interface and common functionality across different pattern types.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BasePattern(ABC):
    """
    Abstract base class for all pattern generators.
    
    Provides common functionality and enforces consistent interface
    across all pattern types according to the mathematical formulations
    in the research paper.
    """
    
    def __init__(self, 
                 days: int = 28, 
                 freq: str = '15min', 
                 seed: int = 42,
                 **kwargs):
        """
        Initialize base pattern generator.
        
        Args:
            days: Number of days to generate
            freq: Frequency of data points (e.g., '15min')
            seed: Random seed for reproducibility
            **kwargs: Additional pattern-specific parameters
        """
        self.days = days
        self.freq = freq
        self.seed = seed
        self.params = kwargs
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate common parameters."""
        if self.days <= 0:
            raise ValueError("Days must be positive")
        if self.seed < 0:
            raise ValueError("Seed must be non-negative")
            
    def _create_time_index(self) -> pd.DatetimeIndex:
        """Create standardized time index."""
        periods = self.days * 24 * 4  # 15-minute intervals
        return pd.date_range(
            start='2024-01-01', 
            periods=periods, 
            freq=self.freq
        )
    
    def _apply_constraints(self, pod_counts: np.ndarray) -> np.ndarray:
        """Apply standard constraints (non-negative, max 100, integer)."""
        pod_counts = np.maximum(pod_counts, 0)
        pod_counts = np.minimum(pod_counts, 100)
        return np.round(pod_counts).astype(int)
    
    def _add_noise(self, 
                   pod_counts: np.ndarray, 
                   noise_level: float) -> np.ndarray:
        """Add Gaussian noise to pod counts."""
        np.random.seed(self.seed)
        noise = np.random.normal(0, noise_level, len(pod_counts))
        return pod_counts + noise
    
    def _add_seasonality(self, 
                        date_range: pd.DatetimeIndex,
                        strength: float = 1.0,
                        period_hours: float = 24.0) -> np.ndarray:
        """
        Add seasonal component: S·sin(2πh_t/T)
        
        Args:
            date_range: Time index
            strength: Seasonal strength (S in formula)
            period_hours: Period in hours (T in formula)
            
        Returns:
            Seasonal component array
        """
        hours = date_range.hour + date_range.minute / 60
        return strength * np.sin(hours * (2 * np.pi / period_hours))
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """
        Generate the pattern data.
        
        Returns:
            DataFrame with 'timestamp' and 'pod_count' columns
        """
        pass
    
    @abstractmethod
    def get_mathematical_formula(self) -> str:
        """
        Return the mathematical formula for this pattern type.
        
        Returns:
            LaTeX-formatted mathematical formula string
        """
        pass
    
    def get_pattern_info(self) -> Dict[str, Any]:
        """
        Get pattern metadata and parameters.
        
        Returns:
            Dictionary containing pattern information
        """
        return {
            'pattern_type': self.__class__.__name__,
            'days': self.days,
            'freq': self.freq,
            'seed': self.seed,
            'parameters': self.params,
            'formula': self.get_mathematical_formula()
        }
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate pattern statistics.
        
        Args:
            df: Generated pattern dataframe
            
        Returns:
            Dictionary of statistical measures
        """
        pod_counts = df['pod_count']
        return {
            'min': float(pod_counts.min()),
            'max': float(pod_counts.max()),
            'mean': float(pod_counts.mean()),
            'std': float(pod_counts.std()),
            'cv': float(pod_counts.std() / pod_counts.mean()) if pod_counts.mean() > 0 else 0.0,
            'variance': float(pod_counts.var())
        }
    
    def split_train_test(self, 
                        df: pd.DataFrame, 
                        train_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: Full dataset
            train_days: Number of days for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_point = train_days * 24 * 4  # 15-minute intervals
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        return train_df, test_df 