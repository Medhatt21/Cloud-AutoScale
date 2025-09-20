"""Synthetic data generators."""

from typing import List, Dict, Optional
import numpy as np
from loguru import logger

from ..core.workload import Workload, WorkloadType, WorkloadPriority
from ..core.resources import ResourceSpecs


class SyntheticDataGenerator:
    """Generator for synthetic workload data."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.logger = logger.bind(component="SyntheticDataGenerator")
    
    def generate_workloads(
        self,
        num_workloads: int = 1000,
        time_span: float = 3600.0,
        workload_mix: Optional[Dict[WorkloadType, float]] = None
    ) -> List[Workload]:
        """Generate synthetic workloads."""
        
        if workload_mix is None:
            workload_mix = {
                WorkloadType.BATCH: 0.3,
                WorkloadType.SERVICE: 0.25,
                WorkloadType.INTERACTIVE: 0.2,
                WorkloadType.ML_TRAINING: 0.15,
                WorkloadType.WEB_SERVER: 0.1,
            }
        
        self.logger.info(f"Generating {num_workloads} synthetic workloads over {time_span}s")
        
        workloads = []
        
        # Generate arrival times
        arrival_times = self._generate_arrival_times(num_workloads, time_span)
        
        for i in range(num_workloads):
            # Choose workload type
            workload_type = np.random.choice(
                list(workload_mix.keys()),
                p=list(workload_mix.values())
            )
            
            # Generate workload
            workload = self._generate_single_workload(
                workload_id=f"synthetic_{i+1:06d}",
                workload_type=workload_type,
                arrival_time=arrival_times[i]
            )
            
            workloads.append(workload)
        
        self.logger.info(f"Generated {len(workloads)} synthetic workloads")
        return workloads
    
    def _generate_arrival_times(self, num_workloads: int, time_span: float) -> List[float]:
        """Generate arrival times using Poisson process."""
        arrival_rate = num_workloads / time_span
        inter_arrival_times = np.random.exponential(1.0 / arrival_rate, num_workloads)
        arrival_times = np.cumsum(inter_arrival_times)
        
        # Scale to fit within time span
        if arrival_times[-1] > time_span:
            arrival_times = arrival_times * (time_span / arrival_times[-1])
        
        return arrival_times.tolist()
    
    def _generate_single_workload(
        self,
        workload_id: str,
        workload_type: WorkloadType,
        arrival_time: float
    ) -> Workload:
        """Generate a single synthetic workload."""
        
        # Resource specs based on type
        specs = self._generate_resource_specs(workload_type)
        
        # Priority distribution
        priority = np.random.choice(
            list(WorkloadPriority),
            p=[0.1, 0.2, 0.4, 0.2, 0.1]  # Critical, High, Medium, Low, Best-effort
        )
        
        # Duration based on type
        duration = self._generate_duration(workload_type)
        
        return Workload(
            workload_id=workload_id,
            workload_type=workload_type,
            priority=priority,
            specs=specs,
            arrival_time=arrival_time,
            duration=duration,
            user_id=f"user_{np.random.randint(1, 101)}",
            job_id=f"job_{np.random.randint(1, 1001)}",
        )
    
    def _generate_resource_specs(self, workload_type: WorkloadType) -> ResourceSpecs:
        """Generate resource specifications for workload type."""
        
        base_configs = {
            WorkloadType.BATCH: {
                'cpu_cores': (2, 8, 1.5),  # (min, max, log_scale)
                'memory_gb': (4.0, 32.0, 1.5),
                'disk_gb': (20.0, 200.0, 1.3),
                'network_gbps': (1.0, 5.0, 1.2),
            },
            WorkloadType.SERVICE: {
                'cpu_cores': (1, 4, 1.3),
                'memory_gb': (2.0, 16.0, 1.4),
                'disk_gb': (10.0, 100.0, 1.3),
                'network_gbps': (0.5, 2.0, 1.2),
            },
            WorkloadType.INTERACTIVE: {
                'cpu_cores': (1, 2, 1.2),
                'memory_gb': (1.0, 8.0, 1.3),
                'disk_gb': (5.0, 50.0, 1.2),
                'network_gbps': (0.1, 1.0, 1.1),
            },
            WorkloadType.ML_TRAINING: {
                'cpu_cores': (4, 16, 1.4),
                'memory_gb': (16.0, 128.0, 1.6),
                'disk_gb': (100.0, 1000.0, 1.5),
                'network_gbps': (2.0, 10.0, 1.3),
                'gpu_count': (0, 4, 1.8),
                'gpu_memory_gb': (0.0, 32.0, 2.0),
            },
            WorkloadType.WEB_SERVER: {
                'cpu_cores': (1, 4, 1.3),
                'memory_gb': (2.0, 16.0, 1.4),
                'disk_gb': (10.0, 200.0, 1.4),
                'network_gbps': (1.0, 5.0, 1.3),
            },
        }
        
        config = base_configs[workload_type]
        
        # Generate CPU cores
        cpu_cores = self._generate_log_normal_int(
            config['cpu_cores'][0],
            config['cpu_cores'][1], 
            config['cpu_cores'][2]
        )
        
        # Generate memory
        memory_gb = self._generate_log_normal_float(
            config['memory_gb'][0],
            config['memory_gb'][1],
            config['memory_gb'][2]
        )
        
        # Generate disk
        disk_gb = self._generate_log_normal_float(
            config['disk_gb'][0],
            config['disk_gb'][1],
            config['disk_gb'][2]
        )
        
        # Generate network
        network_gbps = self._generate_log_normal_float(
            config['network_gbps'][0],
            config['network_gbps'][1],
            config['network_gbps'][2]
        )
        
        # Generate GPU specs (if applicable)
        gpu_count = 0
        gpu_memory_gb = 0.0
        
        if workload_type == WorkloadType.ML_TRAINING and 'gpu_count' in config:
            if np.random.random() < 0.3:  # 30% chance of GPU workload
                gpu_count = self._generate_log_normal_int(
                    config['gpu_count'][0],
                    config['gpu_count'][1],
                    config['gpu_count'][2]
                )
                if gpu_count > 0:
                    gpu_memory_gb = self._generate_log_normal_float(
                        config['gpu_memory_gb'][0],
                        config['gpu_memory_gb'][1],
                        config['gpu_memory_gb'][2]
                    )
        
        return ResourceSpecs(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            network_gbps=network_gbps,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
        )
    
    def _generate_duration(self, workload_type: WorkloadType) -> float:
        """Generate duration for workload type."""
        
        base_durations = {
            WorkloadType.BATCH: (300.0, 3600.0, 1.5),  # 5min - 1hr
            WorkloadType.SERVICE: (60.0, 1800.0, 1.4),  # 1min - 30min
            WorkloadType.INTERACTIVE: (30.0, 600.0, 1.3),  # 30s - 10min
            WorkloadType.ML_TRAINING: (1800.0, 14400.0, 1.6),  # 30min - 4hr
            WorkloadType.WEB_SERVER: (300.0, 7200.0, 1.4),  # 5min - 2hr
        }
        
        min_dur, max_dur, scale = base_durations[workload_type]
        return self._generate_log_normal_float(min_dur, max_dur, scale)
    
    def _generate_log_normal_int(self, min_val: int, max_val: int, scale: float) -> int:
        """Generate log-normal distributed integer."""
        log_min = np.log(max(1, min_val))
        log_max = np.log(max_val)
        log_mean = (log_min + log_max) / 2
        log_std = (log_max - log_min) / (2 * scale)
        
        value = np.random.lognormal(log_mean, log_std)
        return max(min_val, min(max_val, int(round(value))))
    
    def _generate_log_normal_float(self, min_val: float, max_val: float, scale: float) -> float:
        """Generate log-normal distributed float."""
        log_min = np.log(max(0.1, min_val))
        log_max = np.log(max_val)
        log_mean = (log_min + log_max) / 2
        log_std = (log_max - log_min) / (2 * scale)
        
        value = np.random.lognormal(log_mean, log_std)
        return max(min_val, min(max_val, value))
