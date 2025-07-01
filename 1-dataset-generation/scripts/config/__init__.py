"""
Configuration Package

This package provides configuration management for pattern generation,
including parameter variations and constants.
"""

from .pattern_configs import PatternConfigurations, PATTERN_VARIATIONS
from .constants import *

__all__ = [
    'PatternConfigurations',
    'PATTERN_VARIATIONS'
] 