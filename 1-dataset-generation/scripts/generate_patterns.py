#!/usr/bin/env python3
"""
Pattern Generation Orchestrator

This is the main entry point for generating workload patterns for 
Kubernetes autoscaling research. This refactored version provides
a clean, modular approach to pattern generation.

Usage:
    python generate_patterns.py --output-dir ./output --days 35 --train-days 28
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Any
import traceback
from datetime import datetime

# Add patterns directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from patterns import (
    SeasonalPattern, GrowingPattern, BurstPattern,
    OnOffPattern, SteppedPattern, ChaoticPattern
)
from utils import DataProcessor, PatternPlotter
from config import PatternConfigurations, PATTERN_VARIATIONS


class PatternGenerator:
    """Main orchestrator for pattern generation."""
    
    def __init__(self, output_dir: str, days: int = 35, train_days: int = 28):
        """
        Initialize pattern generator.
        
        Args:
            output_dir: Output directory for generated data
            days: Total days to generate
            train_days: Days for training data
        """
        self.output_dir = output_dir
        self.days = days
        self.train_days = train_days
        
        # Initialize utilities
        self.data_processor = DataProcessor()
        self.plotter = PatternPlotter()
        self.config = PatternConfigurations()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Pattern registry
        self.pattern_classes = {
            'seasonal': SeasonalPattern,
            'growing': GrowingPattern,
            'burst': BurstPattern,
            'onoff': OnOffPattern,
            'stepped': SteppedPattern,
            'chaotic': ChaoticPattern
        }
        
        self.generated_patterns = {}
        self.pattern_metadata = {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, 'pattern_generation.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pattern generation started - Output: {self.output_dir}")
    
    def generate_all_patterns(self) -> Dict[str, Any]:
        """
        Generate all pattern types with variations.
        
        Returns:
            Summary of generated patterns
        """
        self.logger.info("Starting pattern generation for all types...")
        
        total_patterns = 0
        pattern_summary = {}
        
        for pattern_type in self.pattern_classes.keys():
            self.logger.info(f"Generating {pattern_type} patterns...")
            
            try:
                patterns = self._generate_pattern_type(pattern_type)
                total_patterns += len(patterns)
                pattern_summary[pattern_type] = {
                    'count': len(patterns),
                    'patterns': list(patterns.keys())
                }
                
                self.logger.info(f"Generated {len(patterns)} {pattern_type} patterns")
                
            except Exception as e:
                self.logger.error(f"Error generating {pattern_type} patterns: {e}")
                self.logger.error(traceback.format_exc())
                pattern_summary[pattern_type] = {'count': 0, 'error': str(e)}
        
        # Generate summary plots and reports
        self._generate_summary_outputs()
        
        summary = {
            'total_patterns': total_patterns,
            'pattern_types': pattern_summary,
            'output_directory': self.output_dir,
            'generation_time': datetime.now().isoformat()
        }
        
        self.logger.info(f"Pattern generation completed! Total patterns: {total_patterns}")
        return summary
    
    def _generate_pattern_type(self, pattern_type: str) -> Dict[str, str]:
        """
        Generate all variations for a specific pattern type.
        
        Args:
            pattern_type: Type of pattern to generate
            
        Returns:
            Dictionary mapping pattern names to file paths
        """
        pattern_class = self.pattern_classes[pattern_type]
        variations = PATTERN_VARIATIONS.get(pattern_type, [{}])
        
        generated_patterns = {}
        
        for i, variation in enumerate(variations, 1):
            pattern_name = f"{pattern_type}_{i:03d}"
            
            try:
                # Create pattern instance
                pattern = pattern_class(
                    days=self.days,
                    seed=42 + i,  # Unique seed for each variation
                    **variation
                )
                
                # Generate data
                df = pattern.generate()
                
                # Validate data
                validation = self.data_processor.validate_pattern_data(df)
                if not validation['is_valid']:
                    self.logger.warning(f"Validation failed for {pattern_name}: {validation['issues']}")
                    continue
                
                # Save data files
                file_paths = self.data_processor.save_with_train_test_split(
                    df, self.output_dir, pattern_name, train_ratio=0.8
                )
                
                # Generate plot
                plot_path = self.plotter.plot_pattern(
                    df, 
                    pattern_name,
                    os.path.join(self.output_dir, f"{pattern_name}_plot.png"),
                    train_days=self.train_days,
                    show_formula=False
                )
                
                # Calculate features
                features = self.data_processor.calculate_pattern_features(df)
                
                # Store metadata
                metadata = {
                    **pattern.get_pattern_info(),
                    'files': file_paths,
                    'plot': plot_path,
                    'features': features,
                    'validation': validation
                }
                
                self.pattern_metadata[pattern_name] = metadata
                generated_patterns[pattern_name] = file_paths['full_file']
                
                self.logger.debug(f"Generated {pattern_name}")
                
            except Exception as e:
                self.logger.error(f"Error generating {pattern_name}: {e}")
                continue
        
        return generated_patterns
    
    def generate_single_pattern(self, 
                               pattern_type: str, 
                               **kwargs) -> Dict[str, Any]:
        """
        Generate a single pattern with custom parameters.
        
        Args:
            pattern_type: Type of pattern to generate
            **kwargs: Pattern-specific parameters
            
        Returns:
            Pattern metadata and file paths
        """
        if pattern_type not in self.pattern_classes:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        pattern_class = self.pattern_classes[pattern_type]
        pattern_name = f"{pattern_type}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create pattern instance
        pattern = pattern_class(
            days=self.days,
            **kwargs
        )
        
        # Generate and process
        df = pattern.generate()
        
        # Save files
        file_paths = self.data_processor.save_with_train_test_split(
            df, self.output_dir, pattern_name, train_ratio=0.8
        )
        
        # Generate plot
        plot_path = self.plotter.plot_pattern(
            df,
            pattern_name,
            os.path.join(self.output_dir, f"{pattern_name}_plot.png"),
            train_days=self.train_days,
            show_formula=False
        )
        
        # Calculate features
        features = self.data_processor.calculate_pattern_features(df)
        
        return {
            **pattern.get_pattern_info(),
            'pattern_name': pattern_name,
            'files': file_paths,
            'plot': plot_path,
            'features': features
        }
    
    def _generate_summary_outputs(self):
        """Generate summary plots and reports."""
        try:
            # Create pattern statistics summary
            pattern_stats = {}
            for name, metadata in self.pattern_metadata.items():
                pattern_stats[name] = metadata['features']
            
            # Generate summary plot
            if pattern_stats:
                summary_plot_path = os.path.join(self.output_dir, 'pattern_summary.png')
                self.plotter.create_pattern_summary_plot(pattern_stats, summary_plot_path)
            
            # Save metadata
            metadata_path = os.path.join(self.output_dir, 'patterns_metadata.json')
            self.data_processor.save_metadata(self.pattern_metadata, metadata_path)
            
            self.logger.info("Generated summary outputs")
            
        except Exception as e:
            self.logger.error(f"Error generating summary outputs: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate workload patterns for Kubernetes autoscaling research"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./pattern_output',
        help='Output directory for generated patterns (default: ./pattern_output)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=35,
        help='Total days to generate (default: 35)'
    )
    
    parser.add_argument(
        '--train-days', '-t',
        type=int,
        default=28,
        help='Days for training data (default: 28)'
    )
    
    parser.add_argument(
        '--pattern-type', '-p',
        type=str,
        choices=['seasonal', 'growing', 'burst', 'onoff', 'stepped', 'chaotic'],
        help='Generate only specific pattern type (default: all types)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize generator
        generator = PatternGenerator(
            output_dir=args.output_dir,
            days=args.days,
            train_days=args.train_days
        )
        
        # Generate patterns
        if args.pattern_type:
            # Generate specific pattern type
            patterns = generator._generate_pattern_type(args.pattern_type)
            print(f"Generated {len(patterns)} {args.pattern_type} patterns")
        else:
            # Generate all pattern types
            summary = generator.generate_all_patterns()
            print(f"Generated {summary['total_patterns']} total patterns")
            print(f"Output directory: {summary['output_directory']}")
            
            # Print pattern type summary
            for pattern_type, info in summary['pattern_types'].items():
                if 'error' in info:
                    print(f"  {pattern_type}: ERROR - {info['error']}")
                else:
                    print(f"  {pattern_type}: {info['count']} patterns")
    
    except Exception as e:
        print(f"Error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 