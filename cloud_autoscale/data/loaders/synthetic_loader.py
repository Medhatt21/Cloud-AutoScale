"""Synthetic data loader for testing and development."""

import numpy as np
import pandas as pd
from typing import Literal

PatternType = Literal["periodic", "bursty", "random_walk", "spike"]


class SyntheticLoader:
    """Generate synthetic demand patterns for autoscaling simulation."""
    
    def __init__(self, pattern: PatternType = "periodic", duration_minutes: int = 60, step_minutes: int = 5, seed: int = 42):
        """
        Initialize synthetic data loader.
        
        Args:
            pattern: Type of demand pattern to generate
            duration_minutes: Total duration of simulation in minutes
            step_minutes: Time step size in minutes
            seed: Random seed for reproducibility
        """
        self.pattern = pattern
        self.duration_minutes = duration_minutes
        self.step_minutes = step_minutes
        self.seed = seed
        np.random.seed(seed)
    
    def load(self) -> pd.DataFrame:
        """
        Generate synthetic demand data.
        
        Returns:
            DataFrame with columns: time, cpu_demand, mem_demand, new_instances
        """
        num_steps = self.duration_minutes // self.step_minutes
        times = np.arange(0, num_steps) * self.step_minutes
        
        if self.pattern == "periodic":
            data = self._generate_periodic(times)
        elif self.pattern == "bursty":
            data = self._generate_bursty(times)
        elif self.pattern == "random_walk":
            data = self._generate_random_walk(times)
        elif self.pattern == "spike":
            data = self._generate_spike(times)
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        return pd.DataFrame(data)
    
    def _generate_periodic(self, times: np.ndarray) -> dict:
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
            'time': times,
            'cpu_demand': np.maximum(5, cpu_demand),
            'mem_demand': np.maximum(5, mem_demand),
            'new_instances': new_instances
        }
    
    def _generate_bursty(self, times: np.ndarray) -> dict:
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
            'time': times,
            'cpu_demand': np.maximum(5, cpu_demand),
            'mem_demand': np.maximum(5, mem_demand),
            'new_instances': new_instances
        }
    
    def _generate_random_walk(self, times: np.ndarray) -> dict:
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
            'time': times,
            'cpu_demand': cpu_demand,
            'mem_demand': mem_demand,
            'new_instances': new_instances
        }
    
    def _generate_spike(self, times: np.ndarray) -> dict:
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
            'time': times,
            'cpu_demand': np.maximum(5, cpu_demand),
            'mem_demand': np.maximum(5, mem_demand),
            'new_instances': new_instances
        }

