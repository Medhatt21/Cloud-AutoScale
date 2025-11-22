"""Cloud AutoScale - Baseline Autoscaling Simulator for Master's Project."""

__version__ = "1.0.0"

from .data import SyntheticLoader, GCP2019Loader
from .simulation import CloudSimulator, BaselineAutoscaler, calculate_metrics, format_metrics_table
from .visualization import create_all_plots
from .config import load_config

__all__ = [
    'SyntheticLoader',
    'GCP2019Loader',
    'CloudSimulator',
    'BaselineAutoscaler',
    'calculate_metrics',
    'format_metrics_table',
    'create_all_plots',
    'load_config'
]

