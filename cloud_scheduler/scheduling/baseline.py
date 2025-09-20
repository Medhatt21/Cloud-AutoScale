"""Baseline scheduling algorithms similar to AWS/Azure."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import random
from loguru import logger

from ..core.resources import Host, Container, ResourceSpecs, ResourceState
from ..core.workload import Workload, WorkloadPriority


class PlacementPolicy(Enum):
    """Placement policies for workload scheduling."""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    SPREAD = "spread"
    COMPACT = "compact"
    AFFINITY = "affinity"
    ANTI_AFFINITY = "anti_affinity"


class BaselineScheduler(ABC):
    """Abstract base class for baseline schedulers."""
    
    def __init__(self, placement_policy: PlacementPolicy):
        self.placement_policy = placement_policy
        self.scheduled_workloads: Dict[str, str] = {}  # workload_id -> host_id
        logger.info(f"Scheduler initialized with {placement_policy.value} policy")
    
    @abstractmethod
    def schedule_workload(self, workload: Workload, hosts: Dict[str, Host]) -> bool:
        """Schedule a workload on available hosts."""
        pass
    
    def _can_schedule_on_host(self, workload: Workload, host: Host) -> bool:
        """Check if workload can be scheduled on host considering taints/tolerations."""
        # Check basic resource availability
        if not host.can_accommodate(workload.specs):
            return False
        
        # Check host state
        if host.state != ResourceState.AVAILABLE:
            return False
        
        # Check taints/tolerations (Kubernetes-style)
        container = Container(
            container_id=f"container_{workload.workload_id}",
            specs=workload.specs,
            tolerations=getattr(workload, 'tolerations', set())
        )
        
        # If host has taints, container must tolerate them
        if host.taints and not host.taints.issubset(container.tolerations):
            return False
        
        return True
    
    def _calculate_host_score(self, workload: Workload, host: Host) -> float:
        """Calculate scheduling score for host (higher is better)."""
        if not self._can_schedule_on_host(workload, host):
            return -1.0
        
        utilization = host.get_utilization()
        cpu_available = 1.0 - utilization.cpu_utilization
        mem_available = 1.0 - utilization.memory_utilization
        
        if self.placement_policy == PlacementPolicy.BEST_FIT:
            # Prefer hosts with just enough resources
            cpu_fit = 1.0 - abs(cpu_available - (workload.specs.cpu_cores / host.specs.cpu_cores))
            mem_fit = 1.0 - abs(mem_available - (workload.specs.memory_gb / host.specs.memory_gb))
            return (cpu_fit + mem_fit) / 2.0
        
        elif self.placement_policy == PlacementPolicy.WORST_FIT:
            # Prefer hosts with most available resources
            return (cpu_available + mem_available) / 2.0
        
        elif self.placement_policy == PlacementPolicy.SPREAD:
            # Prefer less utilized hosts
            return (cpu_available + mem_available) / 2.0
        
        elif self.placement_policy == PlacementPolicy.COMPACT:
            # Prefer more utilized hosts
            return 1.0 - (cpu_available + mem_available) / 2.0
        
        else:
            # Default: balance utilization
            return (cpu_available + mem_available) / 2.0


class FirstFitScheduler(BaselineScheduler):
    """First-fit scheduler - schedules on first available host."""
    
    def __init__(self):
        super().__init__(PlacementPolicy.FIRST_FIT)
    
    def schedule_workload(self, workload: Workload, hosts: Dict[str, Host]) -> bool:
        """Schedule workload on first available host."""
        # Sort hosts by ID for deterministic behavior
        sorted_hosts = sorted(hosts.values(), key=lambda h: h.host_id)
        
        for host in sorted_hosts:
            if self._can_schedule_on_host(workload, host):
                return self._place_workload_on_host(workload, host)
        
        logger.warning(f"Could not schedule workload {workload.workload_id} - no suitable hosts")
        return False
    
    def _place_workload_on_host(self, workload: Workload, host: Host) -> bool:
        """Place workload on the specified host."""
        # Create container for workload
        container = Container(
            container_id=f"container_{workload.workload_id}",
            specs=workload.specs,
            labels=getattr(workload, 'labels', {}),
            tolerations=getattr(workload, 'tolerations', set())
        )
        
        # Schedule container on host
        if container.schedule_on_host(host):
            workload.assigned_container_id = container.container_id
            workload.assigned_host_id = host.host_id
            self.scheduled_workloads[workload.workload_id] = host.host_id
            
            logger.info(f"Workload {workload.workload_id} scheduled on host {host.host_id}")
            return True
        
        return False


class BestFitScheduler(BaselineScheduler):
    """Best-fit scheduler - schedules on host with best resource fit."""
    
    def __init__(self):
        super().__init__(PlacementPolicy.BEST_FIT)
    
    def schedule_workload(self, workload: Workload, hosts: Dict[str, Host]) -> bool:
        """Schedule workload on best-fit host."""
        best_host = None
        best_score = -1.0
        
        for host in hosts.values():
            score = self._calculate_host_score(workload, host)
            if score > best_score:
                best_score = score
                best_host = host
        
        if best_host:
            return self._place_workload_on_host(workload, best_host)
        
        logger.warning(f"Could not schedule workload {workload.workload_id} - no suitable hosts")
        return False
    
    def _place_workload_on_host(self, workload: Workload, host: Host) -> bool:
        """Place workload on the specified host."""
        container = Container(
            container_id=f"container_{workload.workload_id}",
            specs=workload.specs
        )
        
        if container.schedule_on_host(host):
            workload.assigned_container_id = container.container_id
            workload.assigned_host_id = host.host_id
            self.scheduled_workloads[workload.workload_id] = host.host_id
            
            logger.info(f"Workload {workload.workload_id} scheduled on host {host.host_id} "
                       f"(best fit score: {self._calculate_host_score(workload, host):.3f})")
            return True
        
        return False


class SpreadScheduler(BaselineScheduler):
    """Spread scheduler - distributes workloads across hosts evenly."""
    
    def __init__(self):
        super().__init__(PlacementPolicy.SPREAD)
    
    def schedule_workload(self, workload: Workload, hosts: Dict[str, Host]) -> bool:
        """Schedule workload to spread load across hosts."""
        # Find host with lowest utilization
        best_host = None
        lowest_utilization = float('inf')
        
        for host in hosts.values():
            if self._can_schedule_on_host(workload, host):
                utilization = host.get_utilization()
                avg_util = (utilization.cpu_utilization + utilization.memory_utilization) / 2.0
                
                if avg_util < lowest_utilization:
                    lowest_utilization = avg_util
                    best_host = host
        
        if best_host:
            return self._place_workload_on_host(workload, best_host)
        
        logger.warning(f"Could not schedule workload {workload.workload_id} - no suitable hosts")
        return False
    
    def _place_workload_on_host(self, workload: Workload, host: Host) -> bool:
        """Place workload on the specified host."""
        container = Container(
            container_id=f"container_{workload.workload_id}",
            specs=workload.specs
        )
        
        if container.schedule_on_host(host):
            workload.assigned_container_id = container.container_id
            workload.assigned_host_id = host.host_id
            self.scheduled_workloads[workload.workload_id] = host.host_id
            
            logger.info(f"Workload {workload.workload_id} scheduled on host {host.host_id} "
                       f"(spread policy, utilization: {lowest_utilization:.3f})")
            return True
        
        return False


class AffinityScheduler(BaselineScheduler):
    """Affinity-based scheduler supporting node and pod affinity/anti-affinity."""
    
    def __init__(self):
        super().__init__(PlacementPolicy.AFFINITY)
        self.workload_to_host: Dict[str, str] = {}  # workload_id -> host_id
        self.host_to_workloads: Dict[str, Set[str]] = {}  # host_id -> set of workload_ids
    
    def schedule_workload(self, workload: Workload, hosts: Dict[str, Host]) -> bool:
        """Schedule workload considering affinity/anti-affinity rules."""
        suitable_hosts = []
        
        for host in hosts.values():
            if self._can_schedule_on_host(workload, host):
                affinity_score = self._calculate_affinity_score(workload, host)
                if affinity_score >= 0:  # Negative score means anti-affinity violation
                    suitable_hosts.append((host, affinity_score))
        
        if not suitable_hosts:
            logger.warning(f"Could not schedule workload {workload.workload_id} - no suitable hosts")
            return False
        
        # Sort by affinity score (higher is better)
        suitable_hosts.sort(key=lambda x: x[1], reverse=True)
        best_host = suitable_hosts[0][0]
        
        return self._place_workload_on_host(workload, best_host)
    
    def _calculate_affinity_score(self, workload: Workload, host: Host) -> float:
        """Calculate affinity score for workload-host pair."""
        score = 0.0
        
        # Node affinity (based on labels)
        node_affinity = getattr(workload, 'node_affinity', {})
        for label_key, label_value in node_affinity.items():
            if host.labels.get(label_key) == label_value:
                score += 1.0
            elif label_key in host.labels:
                score -= 0.5  # Partial penalty for wrong value
        
        # Pod affinity (co-locate with similar workloads)
        pod_affinity = getattr(workload, 'pod_affinity', {})
        if pod_affinity and host.host_id in self.host_to_workloads:
            for other_workload_id in self.host_to_workloads[host.host_id]:
                # Check if other workload matches affinity criteria
                if self._workloads_have_affinity(workload, other_workload_id):
                    score += 2.0
        
        # Pod anti-affinity (avoid co-location)
        pod_anti_affinity = getattr(workload, 'pod_anti_affinity', {})
        if pod_anti_affinity and host.host_id in self.host_to_workloads:
            for other_workload_id in self.host_to_workloads[host.host_id]:
                if self._workloads_have_anti_affinity(workload, other_workload_id):
                    return -1.0  # Hard anti-affinity violation
        
        return score
    
    def _workloads_have_affinity(self, workload1: Workload, workload2_id: str) -> bool:
        """Check if two workloads should be co-located."""
        # Simple implementation: same workload type or user
        affinity_labels = getattr(workload1, 'pod_affinity', {})
        if 'workload_type' in affinity_labels:
            # This would require looking up workload2's type
            return True  # Simplified
        return False
    
    def _workloads_have_anti_affinity(self, workload1: Workload, workload2_id: str) -> bool:
        """Check if two workloads should not be co-located."""
        anti_affinity_labels = getattr(workload1, 'pod_anti_affinity', {})
        if 'high_availability' in anti_affinity_labels:
            return True  # Don't co-locate HA workloads
        return False
    
    def _place_workload_on_host(self, workload: Workload, host: Host) -> bool:
        """Place workload on the specified host."""
        container = Container(
            container_id=f"container_{workload.workload_id}",
            specs=workload.specs
        )
        
        if container.schedule_on_host(host):
            workload.assigned_container_id = container.container_id
            workload.assigned_host_id = host.host_id
            self.scheduled_workloads[workload.workload_id] = host.host_id
            
            # Update affinity tracking
            self.workload_to_host[workload.workload_id] = host.host_id
            if host.host_id not in self.host_to_workloads:
                self.host_to_workloads[host.host_id] = set()
            self.host_to_workloads[host.host_id].add(workload.workload_id)
            
            affinity_score = self._calculate_affinity_score(workload, host)
            logger.info(f"Workload {workload.workload_id} scheduled on host {host.host_id} "
                       f"(affinity score: {affinity_score:.3f})")
            return True
        
        return False


class PriorityScheduler(BaselineScheduler):
    """Priority-based scheduler that considers workload priorities."""
    
    def __init__(self, base_scheduler: BaselineScheduler):
        super().__init__(base_scheduler.placement_policy)
        self.base_scheduler = base_scheduler
        self.priority_queue: Dict[WorkloadPriority, List[Workload]] = {
            priority: [] for priority in WorkloadPriority
        }
    
    def schedule_workload(self, workload: Workload, hosts: Dict[str, Host]) -> bool:
        """Schedule workload considering priority."""
        # Try to schedule immediately
        if self.base_scheduler.schedule_workload(workload, hosts):
            return True
        
        # If can't schedule, add to priority queue
        self.priority_queue[workload.priority].append(workload)
        logger.info(f"Workload {workload.workload_id} queued with priority {workload.priority.value}")
        return False
    
    def schedule_queued_workloads(self, hosts: Dict[str, Host]) -> int:
        """Schedule queued workloads in priority order."""
        scheduled_count = 0
        
        # Process workloads in priority order
        for priority in WorkloadPriority:
            queue = self.priority_queue[priority]
            remaining_workloads = []
            
            for workload in queue:
                if self.base_scheduler.schedule_workload(workload, hosts):
                    scheduled_count += 1
                    logger.info(f"Queued workload {workload.workload_id} scheduled "
                               f"(priority {priority.value})")
                else:
                    remaining_workloads.append(workload)
            
            self.priority_queue[priority] = remaining_workloads
        
        return scheduled_count
