"""Configuration management for Cloud AutoScale."""

import yaml
from pathlib import Path
from typing import Dict, Any, Literal


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config
    validate_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['mode', 'simulation', 'autoscaler', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate mode
    valid_modes = ['synthetic', 'gcp_2019']
    if config['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode: {config['mode']}. Must be one of {valid_modes}")
    
    # Validate simulation config
    sim_config = config['simulation']
    if 'step_minutes' not in sim_config:
        raise ValueError("Missing simulation.step_minutes")
    
    # Validate autoscaler config
    auto_config = config['autoscaler']
    required_auto = ['upper_threshold', 'lower_threshold', 'max_scale_per_step']
    for key in required_auto:
        if key not in auto_config:
            raise ValueError(f"Missing autoscaler.{key}")
    
    # Validate thresholds
    if not (0 < auto_config['upper_threshold'] <= 1):
        raise ValueError("autoscaler.upper_threshold must be between 0 and 1")
    if not (0 < auto_config['lower_threshold'] < auto_config['upper_threshold']):
        raise ValueError("autoscaler.lower_threshold must be between 0 and upper_threshold")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'mode': 'synthetic',
        'data': {
            'path': 'data/raw/google',
            'synthetic_pattern': 'periodic',
            'duration_minutes': 60
        },
        'simulation': {
            'step_minutes': 5,
            'min_machines': 1,
            'max_machines': 20,
            'machine_capacity': 10,
            'cost_per_machine_per_hour': 0.1
        },
        'autoscaler': {
            'upper_threshold': 0.7,
            'lower_threshold': 0.3,
            'max_scale_per_step': 1,
            'cooldown_steps': 2
        },
        'output': {
            'directory': 'results'
        }
    }

