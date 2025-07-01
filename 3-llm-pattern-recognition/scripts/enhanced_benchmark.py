#!/usr/bin/env python
"""
Enhanced PHPA Pattern Detection and Model Recommendation Tool

This modular benchmark tool provides comprehensive LLM-based pattern recognition
for Kubernetes workload patterns, supporting multiple LLM providers and both
text-based and visual-based analysis methods.

Supported LLMs:
- Gemini 2.5 Pro (Google)
- Qwen3 (Alibaba)  
- Grok-3 (xAI)

Supported Analysis Methods:
- Text-based: CSV data analysis
- Visual-based: Plot/chart analysis

Usage:
  python enhanced_benchmark.py --llm gemini --method text
  python enhanced_benchmark.py --llm all --method visual --generate-only
  python enhanced_benchmark.py --test-file data.csv --llm qwen
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import modular components
from llm_providers import LLMProviderFactory, LLMProvider
from pattern_generator import PatternGenerator
from prompt_builder import PromptBuilder
from evaluator import BenchmarkEvaluator
from visualizer import ResultVisualizer
from config import BenchmarkConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EnhancedBenchmark:
    """Main benchmark orchestrator for enhanced LLM pattern recognition evaluation."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.pattern_generator = PatternGenerator(config)
        self.prompt_builder = PromptBuilder(config)
        self.evaluator = BenchmarkEvaluator(config)
        self.visualizer = ResultVisualizer(config)
        
        # Initialize LLM providers
        self.llm_providers: Dict[str, LLMProvider] = {}
        self._initialize_llm_providers()
    
    def _initialize_llm_providers(self):
        """Initialize configured LLM providers."""
        factory = LLMProviderFactory()
        
        for llm_name in self.config.enabled_llms:
            try:
                provider = factory.create_provider(llm_name, self.config)
                self.llm_providers[llm_name] = provider
                logger.info(f"Initialized {llm_name} provider")
            except Exception as e:
                logger.error(f"Failed to initialize {llm_name}: {e}")
    
    def generate_datasets(self) -> Dict[str, List[str]]:
        """Load existing datasets for all patterns."""
        logger.info("Loading existing pattern datasets...")
        
        # Check if existing datasets are available
        dataset_info = self.pattern_generator.get_existing_dataset_info()
        if not dataset_info:
            logger.error("No existing datasets found. Please run the 1-dataset-generation module first.")
            return {}
        
        # Log available datasets
        for pattern_type, info in dataset_info.items():
            logger.info(f"Available {pattern_type} datasets: {info['available_count']}")
        
        datasets = {}
        for pattern_type in self.config.pattern_types:
            pattern_files = self.pattern_generator.generate_pattern_examples(
                pattern_type, 
                self.config.samples_per_pattern
            )
            datasets[pattern_type] = pattern_files
            logger.info(f"Loaded {len(pattern_files)} examples for {pattern_type}")
        
        return datasets
    
    def run_benchmark(self, datasets: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Run the comprehensive benchmark evaluation."""
        if datasets is None:
            datasets = self.generate_datasets()
        
        results = {
            'overall_results': {},
            'pattern_specific': {},
            'llm_comparison': {},
            'method_comparison': {}
        }
        
        for llm_name, provider in self.llm_providers.items():
            logger.info(f"Evaluating {llm_name}...")
            
            llm_results = self._evaluate_llm(provider, datasets)
            results['overall_results'][llm_name] = llm_results
        
        # Aggregate and analyze results
        results['pattern_specific'] = self.evaluator.analyze_pattern_performance(
            results['overall_results']
        )
        results['llm_comparison'] = self.evaluator.compare_llm_performance(
            results['overall_results']
        )
        results['method_comparison'] = self.evaluator.compare_methods(
            results['overall_results']
        )
        
        return results
    
    def _evaluate_llm(self, provider: LLMProvider, datasets: Dict[str, List[str]]) -> Dict:
        """Evaluate a single LLM provider across all datasets and methods."""
        llm_results = {
            'text_based': {},
            'visual_based': {},
            'overall_accuracy': {},
            'pattern_accuracy': {}
        }
        
        for method in self.config.analysis_methods:
            logger.info(f"Running {method} analysis with {provider.name}...")
            
            method_results = {}
            for pattern_type, files in datasets.items():
                pattern_results = []
                
                for file_path in files:
                    result = self._analyze_single_file(provider, file_path, pattern_type, method)
                    if result:
                        pattern_results.append(result)
                
                method_results[pattern_type] = pattern_results
            
            llm_results[f'{method}_based'] = method_results
        
        # Calculate overall metrics
        llm_results['overall_accuracy'] = self.evaluator.calculate_overall_accuracy(llm_results)
        llm_results['pattern_accuracy'] = self.evaluator.calculate_pattern_accuracy(llm_results)
        
        return llm_results
    
    def _analyze_single_file(
        self, 
        provider: LLMProvider, 
        file_path: str, 
        expected_pattern: str,
        method: str
    ) -> Optional[Dict]:
        """Analyze a single file with the specified LLM and method."""
        try:
            # Generate appropriate prompt
            if method == 'text':
                prompt = self.prompt_builder.build_text_prompt(file_path, expected_pattern)
            else:  # visual
                prompt = self.prompt_builder.build_visual_prompt(file_path, expected_pattern)
            
            # Get LLM response
            response = provider.analyze(prompt)
            if not response:
                return None
            
            # Parse response
            parsed_result = self.evaluator.parse_llm_response(response)
            
            # Evaluate accuracy
            evaluation = self.evaluator.evaluate_prediction(
                parsed_result,
                expected_pattern,
                file_path
            )
            
            return {
                'file_path': file_path,
                'expected_pattern': expected_pattern,
                'method': method,
                'llm_response': response,
                'parsed_result': parsed_result,
                'evaluation': evaluation
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path} with {provider.name}: {e}")
            return None
    
    def test_single_file(self, file_path: str, llm_name: str, method: str) -> Dict:
        """Test a single file with specified LLM and method."""
        if llm_name not in self.llm_providers:
            raise ValueError(f"LLM {llm_name} not available")
        
        provider = self.llm_providers[llm_name]
        
        # Extract pattern from filename or use 'unknown'
        expected_pattern = self._extract_pattern_from_filename(file_path)
        
        result = self._analyze_single_file(provider, file_path, expected_pattern, method)
        
        if result:
            logger.info(f"Analysis complete for {file_path}")
            logger.info(f"Detected pattern: {result['parsed_result'].get('pattern', 'N/A')}")
            logger.info(f"Recommended model: {result['parsed_result'].get('model', 'N/A')}")
            logger.info(f"Reasoning: {result['parsed_result'].get('reasoning', 'N/A')}")
        
        return result or {}
    
    def _extract_pattern_from_filename(self, file_path: str) -> str:
        """Extract pattern type from filename."""
        basename = os.path.basename(file_path).lower()
        for pattern in self.config.pattern_types:
            if pattern in basename:
                return pattern
        return 'unknown'
    
    def save_results(self, results: Dict, output_dir: str):
        """Save benchmark results and generate reports."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate visualizations and reports
        self.visualizer.generate_comprehensive_report(results, output_dir, timestamp)
        
        return results_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced PHPA Pattern Detection Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with all LLMs
  python enhanced_benchmark.py --llm all --method all
  
  # Test specific LLM and method
  python enhanced_benchmark.py --llm gemini --method text
  
  # Generate datasets only
  python enhanced_benchmark.py --generate-only
  
  # Test single file
  python enhanced_benchmark.py --test-file data.csv --llm qwen --method visual
"""
    )
    
    parser.add_argument('--llm', choices=['gemini', 'qwen', 'grok', 'all'], 
                        default='all', help='LLM provider to use')
    parser.add_argument('--method', choices=['text', 'visual', 'all'], 
                        default='all', help='Analysis method')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only load existing datasets without running benchmark')
    parser.add_argument('--test-file', type=str,
                        help='Test a specific CSV file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--samples-per-pattern', type=int, default=3,
                        help='Number of samples to generate per pattern')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = BenchmarkConfig.load(args.config, args)
    
    # Initialize benchmark
    benchmark = EnhancedBenchmark(config)
    
    try:
        if args.test_file:
            # Test single file mode
            if not os.path.exists(args.test_file):
                logger.error(f"Test file not found: {args.test_file}")
                sys.exit(1)
            
            llm_name = args.llm if args.llm != 'all' else 'gemini'
            method = args.method if args.method != 'all' else 'text'
            
            result = benchmark.test_single_file(args.test_file, llm_name, method)
            
            # Save single result
            output_file = os.path.join(args.output_dir, 'single_test_result.json')
            os.makedirs(args.output_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"Single test result saved to {output_file}")
            
        elif args.generate_only:
            # Load datasets only (note: this option is kept for compatibility)
            datasets = benchmark.generate_datasets()
            logger.info(f"Loaded datasets for {len(datasets)} patterns")
            
        else:
            # Full benchmark mode
            logger.info("Starting comprehensive benchmark evaluation...")
            results = benchmark.run_benchmark()
            
            # Save results and generate reports
            results_file = benchmark.save_results(results, args.output_dir)
            logger.info(f"Benchmark complete. Results saved to {results_file}")
            
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 