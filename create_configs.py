#!/usr/bin/env python3
"""Script to create default configuration files."""

import os
from pathlib import Path
import yaml

def create_configs():
    """Create default configuration files."""
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Baseline configuration
    baseline_config = {
        'simulation': {
            'duration': 3600.0,
            'time_step': 1.0,
            'random_seed': 42,
            'enable_failures': True,
            'enable_autoscaling': True,
            'metrics_collection_interval': 30.0,
            'host_failure_rate': 0.001,
            'host_recovery_time': 300.0,
            'workload_arrival_rate': 1.0,
        },
        'autoscaling': {
            'cpu_scale_up_threshold': 0.8,
            'cpu_scale_down_threshold': 0.3,
            'memory_scale_up_threshold': 0.8,
            'memory_scale_down_threshold': 0.3,
            'scale_up_cooldown': 300.0,
            'scale_down_cooldown': 600.0,
            'min_hosts': 5,
            'max_hosts': 50,
        },
        'infrastructure': {
            'host_types': {
                'general_purpose': {'count': 10, 'cpu': 4, 'memory': 16},
                'cpu_optimized': {'count': 5, 'cpu': 8, 'memory': 16},
                'memory_optimized': {'count': 5, 'cpu': 4, 'memory': 32},
                'gpu_enabled': {'count': 2, 'cpu': 8, 'memory': 32, 'gpu': 2},
            }
        },
        'experiment': {
            'name': 'baseline_experiment',
            'description': 'Baseline scheduling with threshold autoscaling',
            'scheduler': 'first_fit',
            'autoscaler': 'threshold',
        }
    }
    
    with open(configs_dir / "baseline.yaml", 'w') as f:
        yaml.dump(baseline_config, f, default_flow_style=False, indent=2)
    
    # ML configuration
    ml_config = baseline_config.copy()
    ml_config['simulation']['duration'] = 7200.0  # 2 hours
    ml_config['autoscaling']['enable_predictive_scaling'] = True
    ml_config['autoscaling']['max_hosts'] = 100
    ml_config['experiment'] = {
        'name': 'ml_experiment',
        'description': 'ML-based scheduling with predictive autoscaling',
        'scheduler': 'ml_scheduler',
        'autoscaler': 'predictive',
        'ml_model': 'lstm_forecaster',
    }
    
    with open(configs_dir / "ml_scheduling.yaml", 'w') as f:
        yaml.dump(ml_config, f, default_flow_style=False, indent=2)
    
    # RL configuration
    rl_config = baseline_config.copy()
    rl_config['simulation']['duration'] = 10800.0  # 3 hours
    rl_config['experiment'] = {
        'name': 'rl_experiment',
        'description': 'Reinforcement learning scheduling',
        'scheduler': 'rl_agent',
        'autoscaler': 'rl_agent',
        'rl_algorithm': 'ppo',
        'training_episodes': 1000,
    }
    
    with open(configs_dir / "rl_scheduling.yaml", 'w') as f:
        yaml.dump(rl_config, f, default_flow_style=False, indent=2)
    
    print("Default configuration files created in configs/ directory")

if __name__ == "__main__":
    create_configs()
