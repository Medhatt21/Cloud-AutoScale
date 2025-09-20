"""Configuration management utilities."""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from pydantic import BaseModel, Field
from loguru import logger

from ..core.simulator import SimulationConfig
from ..scheduling.autoscaling import AutoscalingConfig


class Config(BaseModel):
    """Main configuration class."""
    
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    
    # Infrastructure configuration
    infrastructure: Dict[str, Any] = Field(default_factory=dict)
    
    # Experiment configuration
    experiment: Dict[str, Any] = Field(default_factory=dict)
    
    # Data configuration
    data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


def load_config(config_path: Path) -> SimulationConfig:
    """Load configuration from file."""
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    # Extract simulation config
    sim_config_data = config_data.get('simulation', {})
    
    # Create SimulationConfig with loaded values
    sim_config = SimulationConfig(
        simulation_duration=sim_config_data.get('duration', 3600.0),
        time_step=sim_config_data.get('time_step', 1.0),
        random_seed=sim_config_data.get('random_seed', 42),
        enable_failures=sim_config_data.get('enable_failures', True),
        enable_autoscaling=sim_config_data.get('enable_autoscaling', True),
        metrics_collection_interval=sim_config_data.get('metrics_collection_interval', 30.0),
        host_failure_rate=sim_config_data.get('host_failure_rate', 0.001),
        host_recovery_time=sim_config_data.get('host_recovery_time', 300.0),
        workload_arrival_rate=sim_config_data.get('workload_arrival_rate', 1.0),
    )
    
    logger.info(f"Configuration loaded: {sim_config.simulation_duration}s duration, "
               f"seed {sim_config.random_seed}")
    
    return sim_config


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to file."""
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = {
        'simulation': {
            'duration': config.simulation.simulation_duration,
            'time_step': config.simulation.time_step,
            'random_seed': config.simulation.random_seed,
            'enable_failures': config.simulation.enable_failures,
            'enable_autoscaling': config.simulation.enable_autoscaling,
            'metrics_collection_interval': config.simulation.metrics_collection_interval,
            'host_failure_rate': config.simulation.host_failure_rate,
            'host_recovery_time': config.simulation.host_recovery_time,
            'workload_arrival_rate': config.simulation.workload_arrival_rate,
        },
        'autoscaling': {
            'cpu_scale_up_threshold': config.autoscaling.cpu_scale_up_threshold,
            'cpu_scale_down_threshold': config.autoscaling.cpu_scale_down_threshold,
            'memory_scale_up_threshold': config.autoscaling.memory_scale_up_threshold,
            'memory_scale_down_threshold': config.autoscaling.memory_scale_down_threshold,
            'scale_up_cooldown': config.autoscaling.scale_up_cooldown,
            'scale_down_cooldown': config.autoscaling.scale_down_cooldown,
            'min_hosts': config.autoscaling.min_hosts,
            'max_hosts': config.autoscaling.max_hosts,
        },
        'infrastructure': config.infrastructure,
        'experiment': config.experiment,
        'data': config.data,
    }
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    logger.info(f"Configuration saved to {config_path}")


def save_results(analysis: Dict[str, Any], output_dir: Path) -> None:
    """Save simulation results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis results
    results_file = output_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save metrics history if available
    if 'metrics_history' in analysis:
        metrics_file = output_dir / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(analysis['metrics_history'], f, indent=2, default=str)
        
        logger.info(f"Metrics history saved to {metrics_file}")


def create_default_configs() -> None:
    """Create default configuration files."""
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Baseline configuration
    baseline_config = Config(
        simulation=SimulationConfig(
            simulation_duration=3600.0,
            random_seed=42,
            enable_failures=True,
            enable_autoscaling=True,
        ),
        autoscaling=AutoscalingConfig(
            cpu_scale_up_threshold=0.8,
            cpu_scale_down_threshold=0.3,
            min_hosts=5,
            max_hosts=50,
        ),
        experiment={
            'name': 'baseline_experiment',
            'description': 'Baseline scheduling with threshold autoscaling',
            'scheduler': 'first_fit',
            'autoscaler': 'threshold',
        }
    )
    
    save_config(baseline_config, configs_dir / "baseline.yaml")
    
    # ML configuration
    ml_config = Config(
        simulation=SimulationConfig(
            simulation_duration=7200.0,  # 2 hours
            random_seed=42,
            enable_failures=True,
            enable_autoscaling=True,
        ),
        autoscaling=AutoscalingConfig(
            cpu_scale_up_threshold=0.75,
            cpu_scale_down_threshold=0.25,
            enable_predictive_scaling=True,
            min_hosts=3,
            max_hosts=100,
        ),
        experiment={
            'name': 'ml_experiment',
            'description': 'ML-based scheduling with predictive autoscaling',
            'scheduler': 'ml_scheduler',
            'autoscaler': 'predictive',
            'ml_model': 'lstm_forecaster',
        }
    )
    
    save_config(ml_config, configs_dir / "ml_scheduling.yaml")
    
    # RL configuration
    rl_config = Config(
        simulation=SimulationConfig(
            simulation_duration=10800.0,  # 3 hours
            random_seed=42,
            enable_failures=True,
            enable_autoscaling=True,
        ),
        experiment={
            'name': 'rl_experiment',
            'description': 'Reinforcement learning scheduling',
            'scheduler': 'rl_agent',
            'autoscaler': 'rl_agent',
            'rl_algorithm': 'ppo',
            'training_episodes': 1000,
        }
    )
    
    save_config(rl_config, configs_dir / "rl_scheduling.yaml")
    
    logger.info("Default configuration files created in configs/ directory")
