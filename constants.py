# -*- coding: utf-8 -*-

"""
Project constants used by training scripts.
Only values that are intended to be stable across runs should live here.
"""

DEFAULT_DATA_FILE = "data.csv"
DEFAULT_THETA_FILE = "thetas.csv"

MIN_ROWS_FOR_TRAINING = 2
MIN_DISTINCT_MILEAGES = 2
LOW_DATA_WARNING_THRESHOLD = 10
CONVERGENCE_THRESHOLD = 0.0001

ANSI_RESET = "\033[0m"
ANSI_WARNING = "\033[1;33m"  # bold yellow
ANSI_ERROR = "\033[1;31m"    # bold red

#cli overrideble defaults
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_ITER = 10000