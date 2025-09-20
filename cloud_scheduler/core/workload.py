"""Workload models and generation."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import numpy as np
from loguru import logger

from .resources import ResourceSpecs


class WorkloadType(Enum):
    """Types of workloads."""
    BATCH = "batch"
    SERVICE = "service"
    INTERACTIVE = "interactive"
    ML_TRAINING = "ml_training"
    WEB_SERVER = "web_server"


class WorkloadPriority(Enum):
    """Workload priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BEST_EFFORT = 4


@dataclass
class SLARequirements:
    """Service Level Agreement requirements."""
    max_latency_ms: float = 1000.0
    min_availability: float = 0.99
    max_response_time_ms: float = 500.0
    throughput_rps: float = 100.0


@dataclass
class WorkloadPattern:
    """Workload arrival and execution patterns."""
    arrival_rate: float  # requests per second
    duration_mean: float  # seconds
    duration_std: float  # seconds
    resource_variation: float = 0.1  # coefficient of variation
    burstiness: float = 1.0  # 1.0 = Poisson, >1.0 = bursty


class Workload:
    """A workload request with resource requirements and SLA."""
    
    def __init__(
        self,
        workload_id: str,
        workload_type: WorkloadType,
        priority: WorkloadPriority,
        specs: ResourceSpecs,
        sla: Optional[SLARequirements] = None,
        duration: float = 60.0,
        arrival_time: float = 0.0,
        user_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        self.workload_id = workload_id
        self.workload_type = workload_type
        self.priority = priority
        self.specs = specs
        self.sla = sla or SLARequirements()
        self.duration = duration
        self.arrival_time = arrival_time
        self.user_id = user_id
        self.job_id = job_id
        
        # Execution tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.assigned_container_id: Optional[str] = None
        self.assigned_host_id: Optional[str] = None
        
        # SLA tracking
        self.queue_time: float = 0.0
        self.execution_time: float = 0.0
        self.sla_violated: bool = False
        
        # Resource usage over time
        self.resource_usage_history: List[Tuple[float, Dict[str, float]]] = []
        
        logger.debug(f"Workload {workload_id} created: {workload_type.value}, "
                    f"priority {priority.value}, duration {duration:.1f}s")
    
    def start_execution(self, current_time: float) -> None:
        """Mark workload as started."""
        self.start_time = current_time
        self.queue_time = current_time - self.arrival_time
        
        # Check if queuing time violates SLA
        if self.queue_time > self.sla.max_response_time_ms / 1000.0:
            self.sla_violated = True
            logger.warning(f"Workload {self.workload_id} SLA violated: "
                          f"queue time {self.queue_time:.2f}s")
        
        logger.info(f"Workload {self.workload_id} started execution at {current_time:.2f}s "
                   f"(queued for {self.queue_time:.2f}s)")
    
    def complete_execution(self, current_time: float) -> None:
        """Mark workload as completed."""
        self.end_time = current_time
        if self.start_time:
            self.execution_time = current_time - self.start_time
        
        logger.info(f"Workload {self.workload_id} completed at {current_time:.2f}s "
                   f"(executed for {self.execution_time:.2f}s)")
    
    def update_resource_usage(self, timestamp: float, usage: Dict[str, float]) -> None:
        """Update resource usage history."""
        self.resource_usage_history.append((timestamp, usage.copy()))
    
    def get_total_time(self) -> float:
        """Get total time from arrival to completion."""
        if self.end_time and self.arrival_time:
            return self.end_time - self.arrival_time
        return 0.0
    
    def is_sla_violated(self) -> bool:
        """Check if SLA was violated."""
        if self.sla_violated:
            return True
        
        total_time = self.get_total_time()
        if total_time > self.sla.max_latency_ms / 1000.0:
            return True
        
        return False


class WorkloadGenerator:
    """Generates workloads based on patterns and traces."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Workload patterns for different types
        self.patterns: Dict[WorkloadType, WorkloadPattern] = {
            WorkloadType.BATCH: WorkloadPattern(
                arrival_rate=0.1, duration_mean=300.0, duration_std=100.0
            ),
            WorkloadType.SERVICE: WorkloadPattern(
                arrival_rate=10.0, duration_mean=60.0, duration_std=20.0
            ),
            WorkloadType.INTERACTIVE: WorkloadPattern(
                arrival_rate=5.0, duration_mean=30.0, duration_std=15.0, burstiness=2.0
            ),
            WorkloadType.ML_TRAINING: WorkloadPattern(
                arrival_rate=0.05, duration_mean=1800.0, duration_std=600.0
            ),
            WorkloadType.WEB_SERVER: WorkloadPattern(
                arrival_rate=20.0, duration_mean=10.0, duration_std=5.0, burstiness=3.0
            ),
        }
        
        # Resource templates for different workload types
        self.resource_templates: Dict[WorkloadType, ResourceSpecs] = {
            WorkloadType.BATCH: ResourceSpecs(cpu_cores=2, memory_gb=4.0, disk_gb=10.0, network_gbps=1.0),
            WorkloadType.SERVICE: ResourceSpecs(cpu_cores=1, memory_gb=2.0, disk_gb=5.0, network_gbps=1.0),
            WorkloadType.INTERACTIVE: ResourceSpecs(cpu_cores=1, memory_gb=1.0, disk_gb=2.0, network_gbps=0.5),
            WorkloadType.ML_TRAINING: ResourceSpecs(cpu_cores=8, memory_gb=32.0, disk_gb=100.0, network_gbps=10.0, gpu_count=1),
            WorkloadType.WEB_SERVER: ResourceSpecs(cpu_cores=2, memory_gb=4.0, disk_gb=20.0, network_gbps=2.0),
        }
        
        self.workload_counter = 0
        logger.info(f"WorkloadGenerator initialized with seed {random_seed}")
    
    def generate_workload(
        self,
        workload_type: WorkloadType,
        arrival_time: float,
        priority: Optional[WorkloadPriority] = None,
    ) -> Workload:
        """Generate a single workload."""
        self.workload_counter += 1
        workload_id = f"wl_{workload_type.value}_{self.workload_counter:06d}"
        
        # Get pattern and template
        pattern = self.patterns[workload_type]
        base_specs = self.resource_templates[workload_type]
        
        # Add resource variation
        variation = pattern.resource_variation
        cpu_mult = 1.0 + np.random.normal(0, variation)
        mem_mult = 1.0 + np.random.normal(0, variation)
        
        specs = ResourceSpecs(
            cpu_cores=max(1, int(base_specs.cpu_cores * cpu_mult)),
            memory_gb=max(0.5, base_specs.memory_gb * mem_mult),
            disk_gb=base_specs.disk_gb,
            network_gbps=base_specs.network_gbps,
            gpu_count=base_specs.gpu_count,
            gpu_memory_gb=base_specs.gpu_memory_gb,
        )
        
        # Generate duration
        duration = max(1.0, np.random.normal(pattern.duration_mean, pattern.duration_std))
        
        # Set priority
        if priority is None:
            priority = np.random.choice(list(WorkloadPriority), 
                                      p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Generate SLA based on priority and type
        sla = self._generate_sla(workload_type, priority)
        
        return Workload(
            workload_id=workload_id,
            workload_type=workload_type,
            priority=priority,
            specs=specs,
            sla=sla,
            duration=duration,
            arrival_time=arrival_time,
        )
    
    def _generate_sla(self, workload_type: WorkloadType, priority: WorkloadPriority) -> SLARequirements:
        """Generate SLA requirements based on workload type and priority."""
        base_sla = {
            WorkloadType.BATCH: SLARequirements(max_latency_ms=10000.0, max_response_time_ms=5000.0),
            WorkloadType.SERVICE: SLARequirements(max_latency_ms=2000.0, max_response_time_ms=1000.0),
            WorkloadType.INTERACTIVE: SLARequirements(max_latency_ms=500.0, max_response_time_ms=200.0),
            WorkloadType.ML_TRAINING: SLARequirements(max_latency_ms=30000.0, max_response_time_ms=10000.0),
            WorkloadType.WEB_SERVER: SLARequirements(max_latency_ms=1000.0, max_response_time_ms=300.0),
        }[workload_type]
        
        # Adjust based on priority
        priority_multiplier = {
            WorkloadPriority.CRITICAL: 0.5,
            WorkloadPriority.HIGH: 0.7,
            WorkloadPriority.MEDIUM: 1.0,
            WorkloadPriority.LOW: 1.5,
            WorkloadPriority.BEST_EFFORT: 3.0,
        }[priority]
        
        return SLARequirements(
            max_latency_ms=base_sla.max_latency_ms * priority_multiplier,
            max_response_time_ms=base_sla.max_response_time_ms * priority_multiplier,
            min_availability=max(0.9, base_sla.min_availability / priority_multiplier),
            throughput_rps=base_sla.throughput_rps / priority_multiplier,
        )
    
    def generate_workload_stream(
        self,
        duration: float,
        workload_types: Optional[List[WorkloadType]] = None,
        type_probabilities: Optional[List[float]] = None,
    ) -> List[Workload]:
        """Generate a stream of workloads over a time period."""
        if workload_types is None:
            workload_types = list(WorkloadType)
        
        if type_probabilities is None:
            type_probabilities = [1.0 / len(workload_types)] * len(workload_types)
        
        workloads = []
        current_time = 0.0
        
        while current_time < duration:
            # Choose workload type
            workload_type = np.random.choice(workload_types, p=type_probabilities)
            pattern = self.patterns[workload_type]
            
            # Generate inter-arrival time (exponential for Poisson, adjusted for burstiness)
            if pattern.burstiness > 1.0:
                # Use gamma distribution for burstiness
                inter_arrival = np.random.gamma(1.0/pattern.burstiness, pattern.burstiness/pattern.arrival_rate)
            else:
                # Standard exponential (Poisson arrivals)
                inter_arrival = np.random.exponential(1.0 / pattern.arrival_rate)
            
            current_time += inter_arrival
            
            if current_time < duration:
                workload = self.generate_workload(workload_type, current_time)
                workloads.append(workload)
        
        logger.info(f"Generated {len(workloads)} workloads over {duration:.1f}s")
        return workloads
