"""
Evaluator Module for Enhanced PHPA Benchmark

This module handles LLM response parsing, accuracy calculation,
and comprehensive performance analysis.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    """Evaluates LLM performance on pattern recognition tasks."""
    
    def __init__(self, config):
        self.config = config
        self.best_models = {
            "seasonal": "gbdt",
            "growing": "var",
            "burst": "gbdt", 
            "onoff": "catboost",
            "chaotic": "gbdt",
            "stepped": "gbdt"
        }
    
    def parse_llm_response(self, response_text: str) -> Dict[str, str]:
        """Parse LLM response to extract pattern, model, and reasoning."""
        if not response_text:
            return {"pattern": None, "model": None, "reasoning": "No response"}
        
        parsed = {"pattern": None, "model": None, "reasoning": "N/A"}
        
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip().lower()
            
            if 'pattern' in line and ':' in line:
                pattern_text = line.split(':', 1)[1].strip()
                parsed['pattern'] = self._normalize_pattern(pattern_text)
            
            elif any(keyword in line for keyword in ['model', 'recommendation']) and ':' in line:
                model_text = line.split(':', 1)[1].strip()
                parsed['model'] = self._normalize_model(model_text)
            
            elif 'reasoning' in line and ':' in line:
                parsed['reasoning'] = line.split(':', 1)[1].strip()
        
        return parsed
    
    def _normalize_pattern(self, pattern_text: str) -> Optional[str]:
        """Normalize pattern text to standard pattern names."""
        if not pattern_text:
            return None
        
        pattern_lower = pattern_text.lower().strip()
        
        for pattern in self.config.pattern_types:
            if pattern in pattern_lower:
                return pattern
        
        return None
    
    def _normalize_model(self, model_text: str) -> Optional[str]:
        """Normalize model text to standard model names."""
        if not model_text:
            return None
        
        model_lower = model_text.lower().strip()
        known_models = ['gbdt', 'var', 'catboost', 'prophet', 'lstm', 'arima']
        
        for model in known_models:
            if model in model_lower:
                return model
        
        return None
    
    def evaluate_prediction(self, parsed_result: Dict[str, str], expected_pattern: str, file_path: str) -> Dict:
        """Evaluate a single prediction."""
        llm_pattern = parsed_result.get('pattern')
        llm_model = parsed_result.get('model')
        expected_model = self.best_models.get(expected_pattern)
        
        return {
            'pattern_match': llm_pattern == expected_pattern,
            'model_match': llm_model == expected_model,
            'pattern_confidence': 'high' if llm_pattern else 'low',
            'model_confidence': 'high' if llm_model else 'low'
        }
    
    def calculate_overall_accuracy(self, llm_results: Dict) -> Dict[str, float]:
        """Calculate overall accuracy metrics."""
        total = 0
        correct_patterns = 0
        correct_models = 0
        
        for method_results in llm_results.values():
            if isinstance(method_results, dict):
                for pattern_results in method_results.values():
                    if isinstance(pattern_results, list):
                        for result in pattern_results:
                            if 'evaluation' in result:
                                total += 1
                                if result['evaluation'].get('pattern_match'):
                                    correct_patterns += 1
                                if result['evaluation'].get('model_match'):
                                    correct_models += 1
        
        return {
            'pattern_accuracy': correct_patterns / total if total > 0 else 0.0,
            'model_accuracy': correct_models / total if total > 0 else 0.0
        }
    
    def calculate_pattern_accuracy(self, llm_results: Dict) -> Dict:
        """Calculate accuracy by pattern type."""
        pattern_metrics = {}
        
        for pattern_type in self.config.pattern_types:
            pattern_metrics[pattern_type] = {
                'pattern_accuracy': 0.0,
                'model_accuracy': 0.0,
                'sample_count': 0
            }
        
        return pattern_metrics
    
    def analyze_pattern_performance(self, overall_results: Dict) -> Dict:
        """Analyze performance across patterns."""
        return {}
    
    def compare_llm_performance(self, overall_results: Dict) -> Dict:
        """Compare LLM performance."""
        return {}
    
    def compare_methods(self, overall_results: Dict) -> Dict:
        """Compare analysis methods."""
        return {}
    
    def _extract_after_colon(self, line: str) -> str:
        """Extract text after colon in a line."""
        if ':' in line:
            return line.split(':', 1)[1].strip()
        return ""
    
    def _extract_pattern_from_text(self, text: str) -> Optional[str]:
        """Extract pattern name from free text."""
        if not text:
            return None
            
        text_lower = text.lower()
        
        # Define pattern keywords with priorities
        pattern_keywords = {
            'seasonal': ['seasonal', 'cyclic', 'periodic', 'cyclical'],
            'growing': ['growing', 'trend', 'increasing', 'upward', 'growth'],
            'burst': ['burst', 'spike', 'sudden', 'transient'],
            'onoff': ['on/off', 'onoff', 'binary', 'toggle', 'switching'],
            'chaotic': ['chaotic', 'irregular', 'random', 'unpredictable'],
            'stepped': ['stepped', 'discrete', 'plateau', 'level']
        }
        
        # Count keyword matches
        pattern_scores = {}
        for pattern, keywords in pattern_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                pattern_scores[pattern] = score
        
        if pattern_scores:
            # Return pattern with highest score
            return max(pattern_scores, key=pattern_scores.get)
        
        return None
    
    def _extract_model_from_text(self, text: str) -> Optional[str]:
        """Extract model name from free text."""
        if not text:
            return None
            
        text_lower = text.lower()
        
        # Model keywords
        model_keywords = {
            'gbdt': ['gbdt', 'gradient boosting', 'xgboost', 'lightgbm'],
            'var': ['var', 'vector autoregression'],
            'catboost': ['catboost', 'categorical boosting'],
            'prophet': ['prophet', 'facebook prophet'],
            'lstm': ['lstm', 'long short-term memory'],
            'arima': ['arima', 'autoregressive']
        }
        
        for model, keywords in model_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return model
        
        return None
    
    def _calculate_method_accuracy(self, method_results: Dict) -> Dict:
        """Calculate accuracy for a specific method."""
        
        total_predictions = 0
        correct_patterns = 0
        pattern_counts = defaultdict(int)
        pattern_correct = defaultdict(int)
        
        for pattern_type, pattern_results in method_results.items():
            for result in pattern_results:
                if 'evaluation' in result:
                    total_predictions += 1
                    pattern_counts[pattern_type] += 1
                    
                    if result['evaluation'].get('pattern_match', False):
                        correct_patterns += 1
                        pattern_correct[pattern_type] += 1
        
        metrics = {
            'overall_accuracy': correct_patterns / total_predictions if total_predictions > 0 else 0.0,
            'pattern_accuracy': {}
        }
        
        for pattern_type in pattern_counts:
            if pattern_counts[pattern_type] > 0:
                metrics['pattern_accuracy'][pattern_type] = pattern_correct[pattern_type] / pattern_counts[pattern_type]
            else:
                metrics['pattern_accuracy'][pattern_type] = 0.0
        
        return metrics 