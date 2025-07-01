"""
Plotting Utilities

This module provides utilities for plotting and visualizing
generated patterns with consistent styling and annotations.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PatternPlotter:
    """Utility class for pattern plotting and visualization."""
    
    def __init__(self, style: str = 'seaborn', figsize: tuple = (12, 6)):
        """
        Initialize plotter with style settings.
        
        Args:
            style: Matplotlib style
            figsize: Figure size tuple
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib styling."""
        try:
            plt.style.use(self.style)
        except:
            logger.warning(f"Style {self.style} not available, using default")
    
    def plot_pattern(self, 
                    df: pd.DataFrame,
                    pattern_name: str,
                    output_path: str,
                    train_days: int = 28,
                    show_formula: bool = False,
                    formula: str = None) -> str:
        """
        Create a comprehensive pattern plot.
        
        Args:
            df: Pattern dataframe
            pattern_name: Pattern name for title
            output_path: Output file path
            train_days: Number of training days for split line
            show_formula: Whether to show mathematical formula
            formula: Mathematical formula string
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the pattern
        ax.plot(df['timestamp'], df['pod_count'], linewidth=2, color='steelblue')
        
        # Add train/test split line
        split_time = df['timestamp'].iloc[0] + pd.Timedelta(days=train_days)
        ax.axvline(x=split_time, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(split_time, ax.get_ylim()[1] * 0.9, 'Train/Test Split', 
               rotation=90, va='top', ha='right', color='red', fontsize=10, fontweight='bold')
        
        # Formatting
        self._format_axes(ax, df, pattern_name)
        
        # Add statistics
        self._add_statistics(ax, df)
        
        # Add formula if provided
        if show_formula and formula:
            self._add_formula(fig, formula)
        
        # Save plot
        plt.tight_layout()
        if show_formula and formula:
            plt.subplots_adjust(bottom=0.15)  # Make room for formula
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved plot to {output_path}")
        return output_path
    
    def plot_multiple_patterns(self,
                             patterns: Dict[str, pd.DataFrame],
                             output_path: str,
                             max_days: Optional[int] = None) -> str:
        """
        Plot multiple patterns in subplots.
        
        Args:
            patterns: Dictionary of pattern name -> dataframe
            output_path: Output file path
            max_days: Maximum days to show
            
        Returns:
            Path to saved plot
        """
        n_patterns = len(patterns)
        cols = min(3, n_patterns)
        rows = (n_patterns + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_patterns == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (name, df) in enumerate(patterns.items()):
            ax = axes[i]
            
            # Limit data if max_days specified
            if max_days:
                max_points = max_days * 24 * 4
                plot_df = df.iloc[:max_points] if len(df) > max_points else df
            else:
                plot_df = df
            
            ax.plot(plot_df['timestamp'], plot_df['pod_count'], linewidth=1.5)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Pod Count')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved multi-pattern plot to {output_path}")
        return output_path
    
    def plot_pattern_comparison(self,
                              patterns: Dict[str, pd.DataFrame],
                              output_path: str,
                              max_days: int = 7) -> str:
        """
        Plot patterns overlaid for comparison.
        
        Args:
            patterns: Dictionary of pattern name -> dataframe
            output_path: Output file path
            max_days: Number of days to show
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(patterns)))
        
        for i, (name, df) in enumerate(patterns.items()):
            # Limit to max_days
            max_points = max_days * 24 * 4
            plot_df = df.iloc[:max_points] if len(df) > max_points else df
            
            ax.plot(plot_df['timestamp'], plot_df['pod_count'], 
                   label=name, linewidth=2, color=colors[i])
        
        ax.set_title(f'Pattern Comparison ({max_days} days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Pod Count', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved comparison plot to {output_path}")
        return output_path
    
    def _format_axes(self, ax, df: pd.DataFrame, pattern_name: str):
        """Format plot axes with consistent styling."""
        ax.set_title(f'{pattern_name} Pattern', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Pod Count', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _add_statistics(self, ax, df: pd.DataFrame):
        """Add statistics text box to plot."""
        pod_counts = df['pod_count']
        stats_text = (
            f"Min: {pod_counts.min():.0f}  "
            f"Max: {pod_counts.max():.0f}  "
            f"Mean: {pod_counts.mean():.1f}  "
            f"StdDev: {pod_counts.std():.1f}  "
            f"CV: {pod_counts.std() / pod_counts.mean():.2f}"
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=10)
    
    def _add_formula(self, fig, formula: str):
        """Add mathematical formula to plot."""
        fig.text(0.5, 0.02, f'Formula: ${formula}$', ha='center', 
                fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    @staticmethod
    def create_pattern_summary_plot(pattern_stats: Dict[str, Dict],
                                  output_path: str) -> str:
        """
        Create summary plot showing pattern characteristics.
        
        Args:
            pattern_stats: Dictionary of pattern_name -> statistics
            output_path: Output file path
            
        Returns:
            Path to saved plot
        """
        pattern_names = list(pattern_stats.keys())
        means = [stats['mean'] for stats in pattern_stats.values()]
        stds = [stats['std'] for stats in pattern_stats.values()]
        cvs = [stats['coefficient_of_variation'] for stats in pattern_stats.values()]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Mean pod counts
        ax1.bar(pattern_names, means, color='steelblue', alpha=0.7)
        ax1.set_title('Mean Pod Count by Pattern', fontweight='bold')
        ax1.set_ylabel('Mean Pod Count')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Standard deviations
        ax2.bar(pattern_names, stds, color='orange', alpha=0.7)
        ax2.set_title('Variability by Pattern', fontweight='bold')
        ax2.set_ylabel('Standard Deviation')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Coefficient of variation
        ax3.bar(pattern_names, cvs, color='green', alpha=0.7)
        ax3.set_title('Relative Variability by Pattern', fontweight='bold')
        ax3.set_ylabel('Coefficient of Variation')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved summary plot to {output_path}")
        return output_path 