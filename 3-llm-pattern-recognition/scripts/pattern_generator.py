"""
Pattern Generator Module for Enhanced PHPA Benchmark

This module loads existing Kubernetes workload datasets generated by the
1-dataset-generation module instead of creating new synthetic data.
"""

import os
import pandas as pd
import logging
import glob
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PatternGenerator:
    """Load existing patterns from 1-dataset-generation module."""
    
    def __init__(self, config):
        self.config = config
        # Path to existing datasets from 1-dataset-generation module
        self.existing_dataset_path = "../../1-dataset-generation/scripts/test_output_all"
        
    def generate_pattern_examples(self, pattern_type: str, num_samples: int) -> List[str]:
        """Load existing examples for a specific pattern type."""
        
        if pattern_type not in self.config.pattern_types:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Check if existing dataset directory exists
        if not os.path.exists(self.existing_dataset_path):
            logger.error(f"Existing dataset directory not found: {self.existing_dataset_path}")
            logger.error("Please run the 1-dataset-generation module first to generate datasets.")
            return []
        
        # Find existing files for this pattern type
        pattern_files = glob.glob(os.path.join(self.existing_dataset_path, f"{pattern_type}_*_full.csv"))
        
        if not pattern_files:
            logger.warning(f"No existing {pattern_type} datasets found in {self.existing_dataset_path}")
            return []
        
        # Select up to num_samples files
        selected_files = pattern_files[:num_samples]
        
        # Copy selected files to the expected output directory if needed
        output_dir = os.path.join(self.config.dataset_dir, pattern_type)
        os.makedirs(output_dir, exist_ok=True)
        
        copied_files = []
        for i, source_file in enumerate(selected_files):
            # Create standardized filename
            filename = f"{pattern_type}_{i+1:03d}.csv"
            target_path = os.path.join(output_dir, filename)
            
            try:
                # Copy the file
                df = pd.read_csv(source_file)
                
                # Standardize column names if needed
                if 'ds' in df.columns and 'y' in df.columns:
                    df = df.rename(columns={'ds': 'timestamp', 'y': 'pod_count'})
                
                # Save with standardized format
                df.to_csv(target_path, index=False)
                
                # Generate visualization if needed
                self._generate_plot(df, pattern_type, i+1, output_dir)
                
                copied_files.append(target_path)
                logger.debug(f"Loaded {pattern_type} dataset: {os.path.basename(source_file)} -> {filename}")
                
            except Exception as e:
                logger.error(f"Error loading {source_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(copied_files)} existing {pattern_type} datasets")
        return copied_files
    
    def _generate_plot(self, df: pd.DataFrame, pattern_type: str, sample_id: int, output_dir: str):
        """Generate visualization for pattern data."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['pod_count'], linewidth=1.5, alpha=0.8)
            plt.title(f'{pattern_type.title()} Pattern - Sample {sample_id} (Existing Dataset)', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Pod Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{pattern_type}_{sample_id:03d}_plot.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate plot for {pattern_type} sample {sample_id}: {e}")
    
    def generate_all_patterns(self) -> Dict[str, List[str]]:
        """Load examples for all pattern types."""
        all_datasets = {}
        
        for pattern_type in self.config.pattern_types:
            try:
                datasets = self.generate_pattern_examples(pattern_type, self.config.samples_per_pattern)
                all_datasets[pattern_type] = datasets
            except Exception as e:
                logger.error(f"Failed to load {pattern_type} patterns: {e}")
                all_datasets[pattern_type] = []
        
        return all_datasets
    
    def get_existing_dataset_info(self) -> Dict[str, Dict]:
        """Get information about available existing datasets."""
        if not os.path.exists(self.existing_dataset_path):
            return {}
        
        info = {}
        
        for pattern_type in self.config.pattern_types:
            pattern_files = glob.glob(os.path.join(self.existing_dataset_path, f"{pattern_type}_*_full.csv"))
            
            info[pattern_type] = {
                'available_count': len(pattern_files),
                'files': [os.path.basename(f) for f in pattern_files[:5]]  # Show first 5 files
            }
        
        return info 