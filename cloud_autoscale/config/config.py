"""Configuration management for Cloud AutoScale."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Production requirement: All configuration must be explicit.
    No defaults are provided. Missing fields will raise clear errors.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML parsing fails or configuration is invalid
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config file: {e}")
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    # Validate and return
    return validate_config(config)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration structure and values.
    
    Production requirement: All configuration must be explicit.
    No defaults are provided. Missing fields will raise clear errors.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated configuration dictionary (same as input)
    
    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    # Check required top-level sections
    required_sections = ['mode', 'data', 'simulation', 'autoscaler', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")
    
    # Validate mode
    valid_modes = ['synthetic', 'gcp_2019']
    if config['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode: '{config['mode']}'. Must be one of {valid_modes}")
    
    # Validate data section (mode-specific)
    data_config = config['data']
    
    if config['mode'] == 'synthetic':
        # Synthetic mode requires pattern and duration
        if 'synthetic_pattern' not in data_config:
            raise ValueError("Missing required key 'data.synthetic_pattern' in config")
        if 'duration_minutes' not in data_config:
            raise ValueError("Missing required key 'data.duration_minutes' in config")
        
        valid_patterns = ['periodic', 'bursty', 'random_walk', 'spike']
        if data_config['synthetic_pattern'] not in valid_patterns:
            raise ValueError(
                f"Invalid synthetic_pattern: '{data_config['synthetic_pattern']}'. "
                f"Must be one of {valid_patterns}"
            )
        
        # Validate type
        if not isinstance(data_config['duration_minutes'], int):
            raise ValueError("data.duration_minutes must be an integer")
        if data_config['duration_minutes'] <= 0:
            raise ValueError("data.duration_minutes must be positive")
    
    elif config['mode'] == 'gcp_2019':
        # GCP mode requires processed_dir
        if 'processed_dir' not in data_config:
            raise ValueError("Missing required key 'data.processed_dir' in config")
        
        # Validate that processed_dir exists and contains required files
        processed_dir = Path(data_config['processed_dir'])
        if not processed_dir.exists():
            raise ValueError(f"Processed data directory does not exist: {processed_dir}")
        
        cluster_file = processed_dir / 'cluster_level.parquet'
        if not cluster_file.exists():
            raise ValueError(
                f"Required file not found: {cluster_file}\n"
                f"Please ensure data/processed/cluster_level.parquet exists"
            )
    
    # Validate simulation config - ALL fields required
    sim_config = config['simulation']
    required_sim = [
        'step_minutes',
        'min_machines',
        'max_machines',
        'machine_capacity',
        'cost_per_machine_per_hour'
    ]
    for key in required_sim:
        if key not in sim_config:
            raise ValueError(f"Missing required key 'simulation.{key}' in config")
    
    # Validate simulation values and types
    if not isinstance(sim_config['step_minutes'], (int, float)) or sim_config['step_minutes'] <= 0:
        raise ValueError("simulation.step_minutes must be a positive number")
    if not isinstance(sim_config['min_machines'], int) or sim_config['min_machines'] < 1:
        raise ValueError("simulation.min_machines must be an integer >= 1")
    if not isinstance(sim_config['max_machines'], int) or sim_config['max_machines'] < sim_config['min_machines']:
        raise ValueError("simulation.max_machines must be an integer >= min_machines")
    if not isinstance(sim_config['machine_capacity'], (int, float)) or sim_config['machine_capacity'] <= 0:
        raise ValueError("simulation.machine_capacity must be a positive number")
    if not isinstance(sim_config['cost_per_machine_per_hour'], (int, float)) or sim_config['cost_per_machine_per_hour'] < 0:
        raise ValueError("simulation.cost_per_machine_per_hour must be a non-negative number")
    
    # Validate autoscaler config - ALL fields required
    auto_config = config['autoscaler']
    required_auto = [
        'upper_threshold',
        'lower_threshold',
        'max_scale_per_step',
        'cooldown_steps'
    ]
    for key in required_auto:
        if key not in auto_config:
            raise ValueError(f"Missing required key 'autoscaler.{key}' in config")
    
    # Validate threshold values and types
    if not isinstance(auto_config['upper_threshold'], (int, float)) or not (0 < auto_config['upper_threshold'] <= 1):
        raise ValueError("autoscaler.upper_threshold must be a number between 0 and 1")
    if not isinstance(auto_config['lower_threshold'], (int, float)) or not (0 < auto_config['lower_threshold'] < auto_config['upper_threshold']):
        raise ValueError(
            "autoscaler.lower_threshold must be a number between 0 and upper_threshold"
        )
    if not isinstance(auto_config['max_scale_per_step'], int) or auto_config['max_scale_per_step'] < 1:
        raise ValueError("autoscaler.max_scale_per_step must be an integer >= 1")
    if not isinstance(auto_config['cooldown_steps'], int) or auto_config['cooldown_steps'] < 0:
        raise ValueError("autoscaler.cooldown_steps must be a non-negative integer")
    
    # Validate output section
    if 'directory' not in config['output']:
        raise ValueError("Missing required key 'output.directory' in config")
    if not isinstance(config['output']['directory'], str):
        raise ValueError("output.directory must be a string")
    
    return config

