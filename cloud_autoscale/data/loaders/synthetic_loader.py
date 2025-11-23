"""Synthetic data loader for testing and development."""

import numpy as np
import pandas as pd
from typing import Literal

PatternType = Literal["periodic", "bursty", "random_walk", "spike"]


class SyntheticLoader:
    """Generate synthetic demand patterns for autoscaling simulation."""
    
    def __init__(
        self,
        pattern: PatternType,
        duration_minutes: int,
        step_minutes: int,
        seed: int
    ):
        """
        Initialize synthetic data loader.
        
        Production requirement: All parameters must be explicitly provided.
        No default values allowed.
        
        Args:
            pattern: Type of demand pattern to generate
                     ('periodic', 'bursty', 'random_walk', 'spike')
            duration_minutes: Total duration of simulation in minutes
            step_minutes: Time step size in minutes
            seed: Random seed for reproducibility
        
        Raises:
            ValueError: If pattern is invalid or parameters are out of range
        """
        valid_patterns = ["periodic", "bursty", "random_walk", "spike"]
        if pattern not in valid_patterns:
            raise ValueError(
                f"Invalid pattern: '{pattern}'. Must be one of {valid_patterns}"
            )
        
        if duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        
        if step_minutes <= 0:
            raise ValueError("step_minutes must be positive")
        
        self.pattern = pattern
        self.duration_minutes = duration_minutes
        self.step_minutes = step_minutes
        self.seed = seed
        np.random.seed(seed)
    
    def load(self) -> pd.DataFrame:
        """
        Generate synthetic demand data for simulation.
        
        Returns simulation-ready DataFrame with minimal normalization only.
        NO ML feature engineering (lags, rolling, etc.) - those belong in modeling notebooks.
        
        Returns:
            DataFrame with columns:
            - step: int, 0..N-1 (sequential, no gaps)
            - time: float, minutes since start (step * step_minutes)
            - cpu_demand: float
            - mem_demand: float
            - new_instances: float (raw)
            - new_instances_norm: float, log1p(new_instances) for stable signal
            - machines_reporting: float (NaN for synthetic)
        """
        num_steps = self.duration_minutes // self.step_minutes
        steps = np.arange(num_steps, dtype=int)
        times = steps * self.step_minutes
        
        if self.pattern == "periodic":
            data = self._generate_periodic(steps, times)
        elif self.pattern == "bursty":
            data = self._generate_bursty(steps, times)
        elif self.pattern == "random_walk":
            data = self._generate_random_walk(steps, times)
        elif self.pattern == "spike":
            data = self._generate_spike(steps, times)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        df = pd.DataFrame(data)
        
        # Minimal normalization for simulation stability only
        df['new_instances_norm'] = np.log1p(df['new_instances'])
        
        # Add machines_reporting (NaN for synthetic data)
        df['machines_reporting'] = np.nan
        
        return df
    
    def _generate_periodic(self, steps: np.ndarray, times: np.ndarray) -> dict:
        """Generate periodic sinusoidal demand pattern."""
        # Sinusoidal pattern with period of 60 minutes
        cpu_demand = 30 + 20 * np.sin(2 * np.pi * times / 60)
        mem_demand = 25 + 15 * np.sin(2 * np.pi * times / 60 + np.pi / 4)
        
        # Add some noise
        cpu_demand += np.random.normal(0, 2, len(times))
        mem_demand += np.random.normal(0, 2, len(times))
        
        # New instances arrive periodically
        new_instances = np.random.poisson(2, len(times))
        
        return {
            'step': steps,
            'time': times,
            'cpu_demand': np.maximum(5, cpu_demand),
            'mem_demand': np.maximum(5, mem_demand),
            'new_instances': new_instances
        }
    
    def _generate_bursty(self, steps: np.ndarray, times: np.ndarray) -> dict:
        """Generate bursty demand pattern with sudden spikes."""
        base_cpu = 15
        base_mem = 15
        
        cpu_demand = np.full(len(times), base_cpu, dtype=float)
        mem_demand = np.full(len(times), base_mem, dtype=float)
        new_instances = np.random.poisson(1, len(times))
        
        # Add random bursts
        num_bursts = max(3, len(times) // 10)
        burst_indices = np.random.choice(len(times), num_bursts, replace=False)
        
        for idx in burst_indices:
            burst_duration = min(5, len(times) - idx)
            burst_magnitude = np.random.uniform(20, 40)
            
            for i in range(burst_duration):
                if idx + i < len(times):
                    cpu_demand[idx + i] += burst_magnitude * (1 - i / burst_duration)
                    mem_demand[idx + i] += burst_magnitude * 0.8 * (1 - i / burst_duration)
                    new_instances[idx + i] += np.random.poisson(5)
        
        # Add noise
        cpu_demand += np.random.normal(0, 2, len(times))
        mem_demand += np.random.normal(0, 2, len(times))
        
        return {
            'step': steps,
            'time': times,
            'cpu_demand': np.maximum(5, cpu_demand),
            'mem_demand': np.maximum(5, mem_demand),
            'new_instances': new_instances
        }
    
    def _generate_random_walk(self, steps: np.ndarray, times: np.ndarray) -> dict:
        """Generate random walk demand pattern."""
        cpu_demand = np.zeros(len(times))
        mem_demand = np.zeros(len(times))
        
        # Start at moderate demand
        cpu_demand[0] = 20
        mem_demand[0] = 20
        
        # Random walk with drift
        for i in range(1, len(times)):
            cpu_demand[i] = cpu_demand[i-1] + np.random.normal(0, 3)
            mem_demand[i] = mem_demand[i-1] + np.random.normal(0, 3)
            
            # Keep within bounds
            cpu_demand[i] = np.clip(cpu_demand[i], 5, 60)
            mem_demand[i] = np.clip(mem_demand[i], 5, 60)
        
        new_instances = np.random.poisson(2, len(times))
        
        return {
            'step': steps,
            'time': times,
            'cpu_demand': cpu_demand,
            'mem_demand': mem_demand,
            'new_instances': new_instances
        }
    
    def _generate_spike(self, steps: np.ndarray, times: np.ndarray) -> dict:
        """Generate demand pattern with sharp spikes."""
        base_cpu = 10
        base_mem = 10
        
        cpu_demand = np.full(len(times), base_cpu, dtype=float)
        mem_demand = np.full(len(times), base_mem, dtype=float)
        new_instances = np.random.poisson(1, len(times))
        
        # Add sharp spikes at regular intervals
        spike_interval = max(5, len(times) // 6)
        
        for i in range(0, len(times), spike_interval):
            if i < len(times):
                spike_height = np.random.uniform(40, 60)
                cpu_demand[i] += spike_height
                mem_demand[i] += spike_height * 0.9
                new_instances[i] += np.random.poisson(10)
        
        # Add noise
        cpu_demand += np.random.normal(0, 1, len(times))
        mem_demand += np.random.normal(0, 1, len(times))
        
        return {
            'step': steps,
            'time': times,
            'cpu_demand': np.maximum(5, cpu_demand),
            'mem_demand': np.maximum(5, mem_demand),
            'new_instances': new_instances
        }

