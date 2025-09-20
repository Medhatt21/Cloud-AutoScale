"""Main cloud simulator using SimPy."""

import simpy
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time
from loguru import logger

from .resources import Host, VirtualMachine, Container, ResourceSpecs, ResourceState
from .workload import Workload, WorkloadGenerator, WorkloadType
from .events import EventBus, SimulationEvent, EventType


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    simulation_duration: float = 3600.0  # 1 hour
    time_step: float = 1.0  # 1 second
    random_seed: int = 42
    enable_failures: bool = True
    enable_autoscaling: bool = True
    metrics_collection_interval: float = 30.0  # 30 seconds
    
    # Host failure parameters
    host_failure_rate: float = 0.001  # failures per hour per host
    host_recovery_time: float = 300.0  # 5 minutes
    
    # Workload generation
    workload_arrival_rate: float = 1.0  # workloads per second
    workload_types: List[WorkloadType] = None
    
    def __post_init__(self):
        if self.workload_types is None:
            self.workload_types = list(WorkloadType)


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    timestamp: float
    total_workloads: int = 0
    completed_workloads: int = 0
    failed_workloads: int = 0
    queued_workloads: int = 0
    running_workloads: int = 0
    
    # SLA metrics
    sla_violations: int = 0
    avg_queue_time: float = 0.0
    avg_execution_time: float = 0.0
    avg_total_time: float = 0.0
    
    # Resource metrics
    total_hosts: int = 0
    active_hosts: int = 0
    failed_hosts: int = 0
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    
    # Scaling metrics
    scale_up_events: int = 0
    scale_down_events: int = 0
    vm_starts: int = 0
    vm_stops: int = 0


class CloudSimulator:
    """Main cloud infrastructure simulator."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.env = simpy.Environment()
        self.event_bus = EventBus()
        
        # Infrastructure
        self.hosts: Dict[str, Host] = {}
        self.vms: Dict[str, VirtualMachine] = {}
        self.containers: Dict[str, Container] = {}
        
        # Workload management
        self.workload_generator = WorkloadGenerator(config.random_seed)
        self.workload_queue: List[Workload] = []
        self.running_workloads: Dict[str, Workload] = {}
        self.completed_workloads: List[Workload] = []
        self.failed_workloads: List[Workload] = {}
        
        # Scheduling policies (will be set by scheduler)
        self.scheduler: Optional[Any] = None
        self.autoscaler: Optional[Any] = None
        
        # Metrics collection
        self.metrics_history: List[SimulationMetrics] = []
        self.current_metrics = SimulationMetrics(timestamp=0.0)
        
        # Event subscriptions
        self._setup_event_handlers()
        
        logger.info(f"CloudSimulator initialized with {config.simulation_duration}s duration")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for simulation events."""
        self.event_bus.subscribe(EventType.WORKLOAD_ARRIVAL, self._handle_workload_arrival)
        self.event_bus.subscribe(EventType.WORKLOAD_COMPLETION, self._handle_workload_completion)
        self.event_bus.subscribe(EventType.WORKLOAD_FAILURE, self._handle_workload_failure)
        self.event_bus.subscribe(EventType.HOST_FAILURE, self._handle_host_failure)
        self.event_bus.subscribe(EventType.HOST_RECOVERY, self._handle_host_recovery)
        self.event_bus.subscribe(EventType.METRICS_COLLECTION, self._handle_metrics_collection)
        self.event_bus.subscribe(EventType.AUTOSCALE_CHECK, self._handle_autoscale_check)
    
    def add_host(self, host: Host) -> None:
        """Add a host to the infrastructure."""
        self.hosts[host.host_id] = host
        logger.info(f"Host {host.host_id} added to infrastructure")
    
    def add_hosts(self, hosts: List[Host]) -> None:
        """Add multiple hosts to the infrastructure."""
        for host in hosts:
            self.add_host(host)
    
    def set_scheduler(self, scheduler: Any) -> None:
        """Set the scheduling policy."""
        self.scheduler = scheduler
        logger.info(f"Scheduler set: {scheduler.__class__.__name__}")
    
    def set_autoscaler(self, autoscaler: Any) -> None:
        """Set the autoscaling policy."""
        self.autoscaler = autoscaler
        logger.info(f"Autoscaler set: {autoscaler.__class__.__name__}")
    
    def run(self) -> List[SimulationMetrics]:
        """Run the simulation."""
        logger.info("Starting cloud simulation")
        start_time = time.time()
        
        # Start simulation processes
        self.env.process(self._workload_arrival_process())
        self.env.process(self._metrics_collection_process())
        self.env.process(self._failure_simulation_process())
        
        if self.config.enable_autoscaling and self.autoscaler:
            self.env.process(self._autoscaling_process())
        
        # Run simulation
        self.env.run(until=self.config.simulation_duration)
        
        # Final metrics collection
        self._collect_metrics()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Simulation completed in {elapsed_time:.2f}s "
                   f"(simulated {self.config.simulation_duration}s)")
        
        return self.metrics_history
    
    def _workload_arrival_process(self):
        """Process for generating workload arrivals."""
        while True:
            # Generate next workload
            workload_type = self.workload_generator.patterns.keys().__iter__().__next__()
            workload = self.workload_generator.generate_workload(
                workload_type, self.env.now
            )
            
            # Schedule arrival event
            event = SimulationEvent(
                timestamp=self.env.now,
                event_type=EventType.WORKLOAD_ARRIVAL,
                resource_id=workload.workload_id,
                data={"workload": workload}
            )
            self.event_bus.publish(event)
            
            # Wait for next arrival
            inter_arrival_time = 1.0 / self.config.workload_arrival_rate
            yield self.env.timeout(inter_arrival_time)
    
    def _metrics_collection_process(self):
        """Process for collecting simulation metrics."""
        while True:
            event = SimulationEvent(
                timestamp=self.env.now,
                event_type=EventType.METRICS_COLLECTION,
                resource_id="simulator",
                data={}
            )
            self.event_bus.publish(event)
            
            yield self.env.timeout(self.config.metrics_collection_interval)
    
    def _failure_simulation_process(self):
        """Process for simulating host failures."""
        if not self.config.enable_failures:
            return
            
        while True:
            # Check each host for potential failure
            for host in self.hosts.values():
                if (host.state == ResourceState.AVAILABLE and 
                    self.env.now > 0):  # Don't fail immediately
                    
                    failure_prob = (self.config.host_failure_rate * 
                                  self.config.time_step / 3600.0)  # per hour to per second
                    
                    if self.workload_generator.random.random() < failure_prob:
                        event = SimulationEvent(
                            timestamp=self.env.now,
                            event_type=EventType.HOST_FAILURE,
                            resource_id=host.host_id,
                            data={"host": host}
                        )
                        self.event_bus.publish(event)
            
            yield self.env.timeout(self.config.time_step)
    
    def _autoscaling_process(self):
        """Process for autoscaling decisions."""
        while True:
            if self.autoscaler:
                event = SimulationEvent(
                    timestamp=self.env.now,
                    event_type=EventType.AUTOSCALE_CHECK,
                    resource_id="autoscaler",
                    data={"metrics": self.current_metrics}
                )
                self.event_bus.publish(event)
            
            yield self.env.timeout(60.0)  # Check every minute
    
    def _handle_workload_arrival(self, event: SimulationEvent) -> None:
        """Handle workload arrival event."""
        workload = event.data["workload"]
        self.workload_queue.append(workload)
        
        logger.debug(f"Workload {workload.workload_id} arrived at {event.timestamp:.2f}s")
        
        # Try to schedule immediately
        if self.scheduler:
            scheduled = self.scheduler.schedule_workload(workload, self.hosts)
            if scheduled:
                self._start_workload_execution(workload)
    
    def _handle_workload_completion(self, event: SimulationEvent) -> None:
        """Handle workload completion event."""
        workload_id = event.resource_id
        if workload_id in self.running_workloads:
            workload = self.running_workloads[workload_id]
            workload.complete_execution(event.timestamp)
            
            # Move to completed
            del self.running_workloads[workload_id]
            self.completed_workloads.append(workload)
            
            # Free up resources
            if workload.assigned_container_id:
                container = self.containers.get(workload.assigned_container_id)
                if container:
                    container.terminate()
                    del self.containers[workload.assigned_container_id]
            
            logger.info(f"Workload {workload_id} completed")
    
    def _handle_workload_failure(self, event: SimulationEvent) -> None:
        """Handle workload failure event."""
        workload_id = event.resource_id
        failure_reason = event.data.get("reason", "unknown")
        
        # Move from queue or running to failed
        workload = None
        if workload_id in self.running_workloads:
            workload = self.running_workloads[workload_id]
            del self.running_workloads[workload_id]
        else:
            # Find in queue
            for i, wl in enumerate(self.workload_queue):
                if wl.workload_id == workload_id:
                    workload = self.workload_queue.pop(i)
                    break
        
        if workload:
            self.failed_workloads[workload_id] = workload
            logger.warning(f"Workload {workload_id} failed: {failure_reason}")
    
    def _handle_host_failure(self, event: SimulationEvent) -> None:
        """Handle host failure event."""
        host = event.data["host"]
        host.fail()
        
        # Fail all workloads on this host
        failed_workloads = []
        for workload in self.running_workloads.values():
            if workload.assigned_host_id == host.host_id:
                failed_workloads.append(workload.workload_id)
        
        for workload_id in failed_workloads:
            failure_event = SimulationEvent(
                timestamp=event.timestamp,
                event_type=EventType.WORKLOAD_FAILURE,
                resource_id=workload_id,
                data={"reason": f"host_failure_{host.host_id}"}
            )
            self.event_bus.publish(failure_event)
        
        # Schedule recovery
        recovery_event = SimulationEvent(
            timestamp=event.timestamp + self.config.host_recovery_time,
            event_type=EventType.HOST_RECOVERY,
            resource_id=host.host_id,
            data={"host": host}
        )
        self.event_bus.schedule_event(recovery_event)
    
    def _handle_host_recovery(self, event: SimulationEvent) -> None:
        """Handle host recovery event."""
        host = event.data["host"]
        host.recover()
        logger.info(f"Host {host.host_id} recovered")
    
    def _handle_metrics_collection(self, event: SimulationEvent) -> None:
        """Handle metrics collection event."""
        self._collect_metrics()
    
    def _handle_autoscale_check(self, event: SimulationEvent) -> None:
        """Handle autoscaling check event."""
        if self.autoscaler:
            scaling_decision = self.autoscaler.make_scaling_decision(
                self.current_metrics, self.hosts, self.workload_queue
            )
            
            if scaling_decision:
                logger.info(f"Autoscaling decision: {scaling_decision}")
                # Implementation depends on specific autoscaler
    
    def _start_workload_execution(self, workload: Workload) -> None:
        """Start executing a workload."""
        workload.start_execution(self.env.now)
        self.running_workloads[workload.workload_id] = workload
        
        # Schedule completion
        completion_event = SimulationEvent(
            timestamp=self.env.now + workload.duration,
            event_type=EventType.WORKLOAD_COMPLETION,
            resource_id=workload.workload_id,
            data={"workload": workload}
        )
        self.event_bus.schedule_event(completion_event)
    
    def _collect_metrics(self) -> None:
        """Collect current simulation metrics."""
        metrics = SimulationMetrics(timestamp=self.env.now)
        
        # Workload metrics
        metrics.total_workloads = (len(self.workload_queue) + 
                                 len(self.running_workloads) + 
                                 len(self.completed_workloads) + 
                                 len(self.failed_workloads))
        metrics.completed_workloads = len(self.completed_workloads)
        metrics.failed_workloads = len(self.failed_workloads)
        metrics.queued_workloads = len(self.workload_queue)
        metrics.running_workloads = len(self.running_workloads)
        
        # SLA metrics
        if self.completed_workloads:
            metrics.sla_violations = sum(1 for wl in self.completed_workloads if wl.is_sla_violated())
            metrics.avg_queue_time = sum(wl.queue_time for wl in self.completed_workloads) / len(self.completed_workloads)
            metrics.avg_execution_time = sum(wl.execution_time for wl in self.completed_workloads) / len(self.completed_workloads)
            metrics.avg_total_time = sum(wl.get_total_time() for wl in self.completed_workloads) / len(self.completed_workloads)
        
        # Resource metrics
        metrics.total_hosts = len(self.hosts)
        metrics.active_hosts = sum(1 for host in self.hosts.values() 
                                 if host.state == ResourceState.AVAILABLE)
        metrics.failed_hosts = sum(1 for host in self.hosts.values() 
                                 if host.state == ResourceState.FAILED)
        
        if self.hosts:
            cpu_utils = [host.get_utilization().cpu_utilization for host in self.hosts.values()]
            mem_utils = [host.get_utilization().memory_utilization for host in self.hosts.values()]
            metrics.avg_cpu_utilization = sum(cpu_utils) / len(cpu_utils)
            metrics.avg_memory_utilization = sum(mem_utils) / len(mem_utils)
        
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        logger.debug(f"Metrics collected at {self.env.now:.2f}s: "
                    f"{metrics.running_workloads} running, "
                    f"{metrics.completed_workloads} completed, "
                    f"{metrics.avg_cpu_utilization:.2f} CPU util")
