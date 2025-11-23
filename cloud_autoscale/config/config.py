"""Configuration management for Cloud AutoScale."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
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
    
    Production requirement: All configuration must be explicit.
    No defaults are provided. Missing fields will raise clear errors.
    
    Args:
        config: Configuration dictionary
    
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
            raise ValueError("Missing data.synthetic_pattern for synthetic mode")
        if 'duration_minutes' not in data_config:
            raise ValueError("Missing data.duration_minutes for synthetic mode")
        
        valid_patterns = ['periodic', 'bursty', 'random_walk', 'spike']
        if data_config['synthetic_pattern'] not in valid_patterns:
            raise ValueError(
                f"Invalid synthetic_pattern: '{data_config['synthetic_pattern']}'. "
                f"Must be one of {valid_patterns}"
            )
    
    elif config['mode'] == 'gcp_2019':
        # GCP mode requires processed_dir
        if 'processed_dir' not in data_config:
            raise ValueError("Missing data.processed_dir for gcp_2019 mode")
        
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
            raise ValueError(f"Missing simulation.{key}")
    
    # Validate simulation values
    if sim_config['step_minutes'] <= 0:
        raise ValueError("simulation.step_minutes must be positive")
    if sim_config['min_machines'] < 1:
        raise ValueError("simulation.min_machines must be at least 1")
    if sim_config['max_machines'] < sim_config['min_machines']:
        raise ValueError("simulation.max_machines must be >= min_machines")
    if sim_config['machine_capacity'] <= 0:
        raise ValueError("simulation.machine_capacity must be positive")
    if sim_config['cost_per_machine_per_hour'] < 0:
        raise ValueError("simulation.cost_per_machine_per_hour must be non-negative")
    
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
            raise ValueError(f"Missing autoscaler.{key}")
    
    # Validate threshold values
    if not (0 < auto_config['upper_threshold'] <= 1):
        raise ValueError("autoscaler.upper_threshold must be between 0 and 1")
    if not (0 < auto_config['lower_threshold'] < auto_config['upper_threshold']):
        raise ValueError(
            "autoscaler.lower_threshold must be between 0 and upper_threshold"
        )
    if auto_config['max_scale_per_step'] < 1:
        raise ValueError("autoscaler.max_scale_per_step must be at least 1")
    if auto_config['cooldown_steps'] < 0:
        raise ValueError("autoscaler.cooldown_steps must be non-negative")
    
    # Validate output section
    if 'directory' not in config['output']:
        raise ValueError("Missing output.directory")


def get_default_config() -> Dict[str, Any]:
    """
    PRODUCTION MODE: Default configurations are not allowed.
    
    All configuration must be explicit and provided via YAML files.
    This prevents silent failures and hidden assumptions.
    
    Raises:
        RuntimeError: Always, as default configs are forbidden in production
    """
    raise RuntimeError(
        "Default configuration is not allowed in production mode.\n"
        "Please provide an explicit configuration file via --config flag.\n"
        "See config/baseline.yaml for an example."
    )

