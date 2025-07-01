"""
Data Processing Utilities

This module provides utilities for data processing, validation,
and feature calculation for generated patterns.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy data types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DataProcessor:
    """Utility class for processing pattern data."""
    
    @staticmethod
    def save_with_train_test_split(df: pd.DataFrame, 
                                  output_dir: str, 
                                  pattern_name: str, 
                                  train_ratio: float = 0.8) -> Dict[str, str]:
        """
        Save dataset with train/test split.
        
        Args:
            df: Pattern dataframe
            output_dir: Output directory
            pattern_name: Pattern name for file naming
            train_ratio: Ratio of data for training (default 0.8 for 80%)
            
        Returns:
            Dictionary with file paths
        """
        # Calculate split point based on percentage
        total_rows = len(df)
        split_point = int(total_rows * train_ratio)
        
        # Ensure we have at least some data for both train and test
        if split_point >= total_rows - 1:
            split_point = total_rows - max(1, int(total_rows * 0.1))  # At least 10% for test
        if split_point < 1:
            split_point = 1  # At least 1 row for train
        
        # Split data
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        
        # Create file paths
        train_file = os.path.join(output_dir, f"{pattern_name}_train.csv")
        test_file = os.path.join(output_dir, f"{pattern_name}_test.csv")
        full_file = os.path.join(output_dir, f"{pattern_name}_full.csv")
        
        # Save files
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        df.to_csv(full_file, index=False)
        
        train_pct = (len(train_df) / total_rows) * 100
        test_pct = (len(test_df) / total_rows) * 100
        
        logger.info(f"Saved {pattern_name} datasets: train={len(train_df)} ({train_pct:.1f}%), test={len(test_df)} ({test_pct:.1f}%)")
        
        return {
            'train_file': train_file,
            'test_file': test_file,
            'full_file': full_file
        }
    
    @staticmethod
    def calculate_pattern_features(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive pattern features.
        
        Args:
            df: Pattern dataframe
            
        Returns:
            Dictionary of calculated features
        """
        pod_counts = df['pod_count'].values
        
        # Basic statistics
        features = {
            'mean': float(np.mean(pod_counts)),
            'std': float(np.std(pod_counts)),
            'min': float(np.min(pod_counts)),
            'max': float(np.max(pod_counts)),
            'range': float(np.max(pod_counts) - np.min(pod_counts)),
            'variance': float(np.var(pod_counts)),
            'coefficient_of_variation': float(np.std(pod_counts) / np.mean(pod_counts)) if np.mean(pod_counts) > 0 else 0.0
        }
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            features[f'percentile_{p}'] = float(np.percentile(pod_counts, p))
        
        # Trend analysis
        time_index = np.arange(len(pod_counts))
        if len(time_index) > 1:
            trend_coeff = np.polyfit(time_index, pod_counts, 1)[0]
            features['trend_slope'] = float(trend_coeff)
        else:
            features['trend_slope'] = 0.0
        
        # Volatility measures
        if len(pod_counts) > 1:
            diff = np.diff(pod_counts)
            features['mean_absolute_change'] = float(np.mean(np.abs(diff)))
            features['max_change'] = float(np.max(np.abs(diff)))
            features['change_variance'] = float(np.var(diff))
        else:
            features['mean_absolute_change'] = 0.0
            features['max_change'] = 0.0
            features['change_variance'] = 0.0
        
        # Autocorrelation at lag 1
        if len(pod_counts) > 1:
            autocorr_lag1 = np.corrcoef(pod_counts[:-1], pod_counts[1:])[0, 1]
            features['autocorr_lag1'] = float(autocorr_lag1) if not np.isnan(autocorr_lag1) else 0.0
        else:
            features['autocorr_lag1'] = 0.0
        
        # Seasonality detection (simplified)
        if len(pod_counts) >= 96:  # At least 1 day of data
            daily_pattern = pod_counts[:96]  # First day
            if len(pod_counts) >= 192:  # At least 2 days
                second_day = pod_counts[96:192]
                if len(daily_pattern) == len(second_day):
                    daily_correlation = np.corrcoef(daily_pattern, second_day)[0, 1]
                    features['daily_seasonality'] = float(daily_correlation) if not np.isnan(daily_correlation) else 0.0
                else:
                    features['daily_seasonality'] = 0.0
            else:
                features['daily_seasonality'] = 0.0
        else:
            features['daily_seasonality'] = 0.0
        
        return features
    
    @staticmethod
    def validate_pattern_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate pattern data quality.
        
        Args:
            df: Pattern dataframe
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check required columns
        required_columns = ['timestamp', 'pod_count']
        for col in required_columns:
            if col not in df.columns:
                validation['is_valid'] = False
                validation['issues'].append(f"Missing required column: {col}")
        
        if not validation['is_valid']:
            return validation
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            validation['issues'].append("Timestamp column is not datetime type")
        
        if not pd.api.types.is_numeric_dtype(df['pod_count']):
            validation['issues'].append("Pod count column is not numeric type")
        
        # Check for missing values
        if df['timestamp'].isnull().any():
            validation['issues'].append("Missing values in timestamp column")
        
        if df['pod_count'].isnull().any():
            validation['issues'].append("Missing values in pod_count column")
        
        # Check value ranges
        if (df['pod_count'] < 0).any():
            validation['warnings'].append("Negative pod counts detected")
        
        if (df['pod_count'] > 100).any():
            validation['warnings'].append("Pod counts exceed 100")
        
        # Check temporal consistency
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            expected_diff = pd.Timedelta(minutes=15)
            if not (time_diffs == expected_diff).all():
                validation['warnings'].append("Irregular time intervals detected")
        
        # Update validity based on issues
        if validation['issues']:
            validation['is_valid'] = False
        
        return validation
    
    @staticmethod
    def load_pattern_data(filepath: str) -> pd.DataFrame:
        """
        Load pattern data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded dataframe
        """
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error loading pattern data from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_metadata(metadata: Dict[str, Any], filepath: str):
        """
        Save pattern metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filepath: Output file path
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            logger.debug(f"Saved metadata to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metadata to {filepath}: {e}")
            raise 