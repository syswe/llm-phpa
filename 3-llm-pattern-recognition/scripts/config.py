"""
Configuration Module for Enhanced PHPA Benchmark

This module manages configuration settings for the comprehensive LLM benchmark
system, including API keys, model parameters, and evaluation settings.
"""

import os
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Comprehensive benchmark configuration."""
    
    # General settings
    enabled_llms: List[str]
    analysis_methods: List[str]
    pattern_types: List[str]
    samples_per_pattern: int
    max_rows_per_example: int
    output_dir: str
    
    # LLM configurations
    gemini_config: Dict[str, Any]
    qwen_config: Dict[str, Any]
    grok_config: Dict[str, Any]
    
    # Evaluation settings
    delay_between_calls: float
    timeout_seconds: int
    max_retries: int
    
    # Dataset generation settings
    dataset_dir: str
    temporal_resolution_minutes: int
    dataset_duration_days: int
    train_test_split: float
    
    @classmethod
    def load(cls, config_path: str, args=None) -> 'BenchmarkConfig':
        """Load configuration from file and command line arguments."""
        
        # Default configuration
        config_data = cls._get_default_config()
        
        # Load from YAML file if exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config_data.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Config file {config_path} not found, using defaults")
        
        # Override with command line arguments
        if args:
            config_data = cls._apply_args_overrides(config_data, args)
        
        return cls(**config_data)
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            # General settings
            'enabled_llms': ['gemini', 'qwen', 'grok'],
            'analysis_methods': ['text', 'visual'],
            'pattern_types': ['seasonal', 'growing', 'burst', 'onoff', 'chaotic', 'stepped'],
            'samples_per_pattern': 3,
            'max_rows_per_example': 500,
            'output_dir': 'benchmark_results',
            
            # LLM API configurations
            'gemini_config': {
                'api_key': os.getenv('GEMINI_API_KEY', ''),
                'model_name': 'gemini-2.0-flash',
                'base_url': '',  # Set by provider
                'max_tokens': 1000,
                'temperature': 0.1,
                'timeout': 60,
                'max_retries': 3,
                'retry_delay': 2.0
            },
            'qwen_config': {
                'api_key': os.getenv('QWEN_API_KEY', ''),
                'model_name': 'qwen-turbo',
                'base_url': '',  # Set by provider
                'max_tokens': 1000,
                'temperature': 0.1,
                'timeout': 60,
                'max_retries': 3,
                'retry_delay': 2.0
            },
            'grok_config': {
                'api_key': os.getenv('GROK_API_KEY', ''),
                'model_name': 'grok-beta',
                'base_url': '',  # Set by provider
                'max_tokens': 1000,
                'temperature': 0.1,
                'timeout': 60,
                'max_retries': 3,
                'retry_delay': 2.0
            },
            
            # Evaluation settings
            'delay_between_calls': 5.0,
            'timeout_seconds': 60,
            'max_retries': 3,
            
            # Dataset loading settings
            'dataset_dir': 'pattern-examples',
            'temporal_resolution_minutes': 15,
            'dataset_duration_days': 35,
            'train_test_split': 0.8
        }
    
    @staticmethod
    def _apply_args_overrides(config_data: Dict[str, Any], args) -> Dict[str, Any]:
        """Apply command line argument overrides to configuration."""
        
        # Handle LLM selection
        if hasattr(args, 'llm') and args.llm != 'all':
            config_data['enabled_llms'] = [args.llm]
        
        # Handle method selection
        if hasattr(args, 'method') and args.method != 'all':
            config_data['analysis_methods'] = [args.method]
        
        # Handle output directory
        if hasattr(args, 'output_dir') and args.output_dir:
            config_data['output_dir'] = args.output_dir
        
        # Handle samples per pattern
        if hasattr(args, 'samples_per_pattern') and args.samples_per_pattern:
            config_data['samples_per_pattern'] = args.samples_per_pattern
        
        return config_data
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        valid = True
        
        # Check enabled LLMs have API keys
        for llm in self.enabled_llms:
            config_attr = f'{llm}_config'
            if hasattr(self, config_attr):
                llm_config = getattr(self, config_attr)
                if not llm_config.get('api_key'):
                    logger.warning(f"No API key configured for {llm}")
                    valid = False
            else:
                logger.error(f"No configuration found for LLM: {llm}")
                valid = False
        
        # Validate pattern types
        expected_patterns = {'seasonal', 'growing', 'burst', 'onoff', 'chaotic', 'stepped'}
        if not set(self.pattern_types).issubset(expected_patterns):
            logger.error(f"Invalid pattern types. Expected subset of {expected_patterns}")
            valid = False
        
        # Validate analysis methods
        expected_methods = {'text', 'visual'}
        if not set(self.analysis_methods).issubset(expected_methods):
            logger.error(f"Invalid analysis methods. Expected subset of {expected_methods}")
            valid = False
        
        return valid
    
    def get_llm_config(self, llm_name: str) -> Dict[str, Any]:
        """Get configuration for specific LLM."""
        config_attr = f'{llm_name.lower()}_config'
        if hasattr(self, config_attr):
            return getattr(self, config_attr)
        else:
            raise ValueError(f"No configuration found for LLM: {llm_name}")
    
    def save(self, output_path: str):
        """Save current configuration to file."""
        try:
            config_dict = {
                'enabled_llms': self.enabled_llms,
                'analysis_methods': self.analysis_methods,
                'pattern_types': self.pattern_types,
                'samples_per_pattern': self.samples_per_pattern,
                'max_rows_per_example': self.max_rows_per_example,
                'output_dir': self.output_dir,
                'gemini_config': self.gemini_config,
                'qwen_config': self.qwen_config,
                'grok_config': self.grok_config,
                'delay_between_calls': self.delay_between_calls,
                'timeout_seconds': self.timeout_seconds,
                'max_retries': self.max_retries,
                'dataset_dir': self.dataset_dir,
                'temporal_resolution_minutes': self.temporal_resolution_minutes,
                'dataset_duration_days': self.dataset_duration_days,
                'train_test_split': self.train_test_split
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

def create_default_config_file(output_path: str = 'config.yaml'):
    """Create a default configuration file."""
    config = BenchmarkConfig._get_default_config()
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Default configuration file created: {output_path}")
        print(f"""
Default configuration file created: {output_path}

Please edit this file to add your API keys:
- GEMINI_API_KEY: Your Google Gemini API key
- QWEN_API_KEY: Your Alibaba Qwen API key  
- GROK_API_KEY: Your xAI Grok API key

You can also set these as environment variables.
""")
    except Exception as e:
        logger.error(f"Failed to create default config file: {e}")

if __name__ == "__main__":
    create_default_config_file() 