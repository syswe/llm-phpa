"""
Constants and Configuration Values

This module defines shared constants used across the pattern generation system.
"""

# Time-related constants
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
INTERVALS_PER_HOUR = 4  # 15-minute intervals
INTERVALS_PER_DAY = HOURS_PER_DAY * INTERVALS_PER_HOUR

# Default pattern generation parameters
DEFAULT_DAYS = 35
DEFAULT_TRAIN_DAYS = 28
DEFAULT_FREQ = '15min'
DEFAULT_SEED = 42

# Pod count constraints
MIN_POD_COUNT = 0
MAX_POD_COUNT = 100

# Pattern type names
PATTERN_TYPES = [
    'seasonal',
    'growing', 
    'burst',
    'onoff',
    'stepped',
    'chaotic'
]

# Noise level categories
NOISE_LEVELS = {
    'low': 0.2,
    'medium': 0.5,
    'high': 0.8
}

# Mathematical constants
PI = 3.14159265359
TWO_PI = 2 * PI

# File extensions
CSV_EXTENSION = '.csv'
PNG_EXTENSION = '.png'
JSON_EXTENSION = '.json'
LOG_EXTENSION = '.log'

# Default plot styling
DEFAULT_FIGURE_SIZE = (12, 6)
DEFAULT_DPI = 150
DEFAULT_LINE_WIDTH = 2

# Statistical thresholds
VALIDATION_MIN_POINTS = 10
AUTOCORR_LAG_THRESHOLD = 0.8
CV_HIGH_THRESHOLD = 1.0 