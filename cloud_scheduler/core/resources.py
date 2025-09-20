"""Cloud resource models: Hosts, VMs, and Containers."""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from loguru import logger


class ResourceState(Enum):
    """Resource state enumeration."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class ResourceType(Enum):
    """Resource type enumeration."""
    CPU_OPTIMIZED = "cpu_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    GENERAL_PURPOSE = "general_purpose"
    GPU_ENABLED = "gpu_enabled"


@dataclass
class ResourceSpecs:
    """Resource specifications."""
    cpu_cores: int
    memory_gb: float
    disk_gb: float
    network_gbps: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0


@dataclass
class ResourceUsage:
    """Current resource usage."""
    cpu_utilization: float = 0.0  # 0-1
    memory_utilization: float = 0.0  # 0-1
    disk_utilization: float = 0.0  # 0-1
    network_utilization: float = 0.0  # 0-1
    gpu_utilization: float = 0.0  # 0-1


class Host:
    """Physical host/node in the cloud infrastructure."""
    
    def __init__(
        self,
        host_id: str,
        specs: ResourceSpecs,
        resource_type: ResourceType = ResourceType.GENERAL_PURPOSE,
        zone: str = "default",
        rack_id: Optional[str] = None,
    ):
        self.host_id = host_id
        self.specs = specs
        self.resource_type = resource_type
        self.zone = zone
        self.rack_id = rack_id
        self.state = ResourceState.AVAILABLE
        
        # Resource tracking
        self.allocated_specs = ResourceSpecs(0, 0.0, 0.0, 0.0)
        self.current_usage = ResourceUsage()
        
        # Containers running on this host
        self.vms: Dict[str, "VirtualMachine"] = {}
        self.containers: Dict[str, "Container"] = {}
        
        # Failure simulation
        self.failure_rate = 0.001  # failures per hour
        self.recovery_time = 300.0  # seconds
        
        # Labels and taints for Kubernetes-style scheduling
        self.labels: Dict[str, str] = {}
        self.taints: Set[str] = set()
        
        logger.info(f"Host {host_id} created with {specs.cpu_cores} cores, "
                   f"{specs.memory_gb}GB RAM in zone {zone}")
    
    def can_accommodate(self, required_specs: ResourceSpecs) -> bool:
        """Check if host can accommodate the required resources."""
        available_cpu = self.specs.cpu_cores - self.allocated_specs.cpu_cores
        available_memory = self.specs.memory_gb - self.allocated_specs.memory_gb
        available_disk = self.specs.disk_gb - self.allocated_specs.disk_gb
        available_gpu = self.specs.gpu_count - self.allocated_specs.gpu_count
        
        return (
            self.state == ResourceState.AVAILABLE and
            available_cpu >= required_specs.cpu_cores and
            available_memory >= required_specs.memory_gb and
            available_disk >= required_specs.disk_gb and
            available_gpu >= required_specs.gpu_count
        )
    
    def allocate_resources(self, specs: ResourceSpecs) -> bool:
        """Allocate resources on this host."""
        if not self.can_accommodate(specs):
            return False
            
        self.allocated_specs.cpu_cores += specs.cpu_cores
        self.allocated_specs.memory_gb += specs.memory_gb
        self.allocated_specs.disk_gb += specs.disk_gb
        self.allocated_specs.gpu_count += specs.gpu_count
        
        logger.debug(f"Allocated {specs.cpu_cores} cores, {specs.memory_gb}GB "
                    f"on host {self.host_id}")
        return True
    
    def deallocate_resources(self, specs: ResourceSpecs) -> None:
        """Deallocate resources from this host."""
        self.allocated_specs.cpu_cores -= specs.cpu_cores
        self.allocated_specs.memory_gb -= specs.memory_gb
        self.allocated_specs.disk_gb -= specs.disk_gb
        self.allocated_specs.gpu_count -= specs.gpu_count
        
        # Ensure non-negative values
        self.allocated_specs.cpu_cores = max(0, self.allocated_specs.cpu_cores)
        self.allocated_specs.memory_gb = max(0.0, self.allocated_specs.memory_gb)
        self.allocated_specs.disk_gb = max(0.0, self.allocated_specs.disk_gb)
        self.allocated_specs.gpu_count = max(0, self.allocated_specs.gpu_count)
        
        logger.debug(f"Deallocated {specs.cpu_cores} cores, {specs.memory_gb}GB "
                    f"from host {self.host_id}")
    
    def get_utilization(self) -> ResourceUsage:
        """Get current resource utilization."""
        if self.specs.cpu_cores > 0:
            self.current_usage.cpu_utilization = (
                self.allocated_specs.cpu_cores / self.specs.cpu_cores
            )
        if self.specs.memory_gb > 0:
            self.current_usage.memory_utilization = (
                self.allocated_specs.memory_gb / self.specs.memory_gb
            )
        return self.current_usage
    
    def fail(self) -> None:
        """Simulate host failure."""
        self.state = ResourceState.FAILED
        logger.warning(f"Host {self.host_id} failed")
    
    def recover(self) -> None:
        """Simulate host recovery."""
        self.state = ResourceState.AVAILABLE
        logger.info(f"Host {self.host_id} recovered")


class VirtualMachine:
    """Virtual Machine running on a host."""
    
    def __init__(
        self,
        vm_id: str,
        specs: ResourceSpecs,
        host: Host,
        vm_type: str = "standard",
    ):
        self.vm_id = vm_id
        self.specs = specs
        self.host = host
        self.vm_type = vm_type
        self.state = ResourceState.AVAILABLE
        
        # Resource tracking
        self.allocated_specs = ResourceSpecs(0, 0.0, 0.0, 0.0)
        self.current_usage = ResourceUsage()
        
        # Containers running on this VM
        self.containers: Dict[str, "Container"] = {}
        
        # Startup/shutdown times
        self.startup_time = 60.0  # seconds
        self.shutdown_time = 30.0  # seconds
        
        logger.info(f"VM {vm_id} created on host {host.host_id}")
    
    def can_accommodate(self, required_specs: ResourceSpecs) -> bool:
        """Check if VM can accommodate the required resources."""
        available_cpu = self.specs.cpu_cores - self.allocated_specs.cpu_cores
        available_memory = self.specs.memory_gb - self.allocated_specs.memory_gb
        
        return (
            self.state == ResourceState.AVAILABLE and
            available_cpu >= required_specs.cpu_cores and
            available_memory >= required_specs.memory_gb
        )
    
    def start(self) -> None:
        """Start the VM."""
        if self.host.allocate_resources(self.specs):
            self.state = ResourceState.AVAILABLE
            self.host.vms[self.vm_id] = self
            logger.info(f"VM {self.vm_id} started on host {self.host.host_id}")
        else:
            logger.error(f"Failed to start VM {self.vm_id} - insufficient resources")
    
    def stop(self) -> None:
        """Stop the VM."""
        self.state = ResourceState.FAILED
        self.host.deallocate_resources(self.specs)
        if self.vm_id in self.host.vms:
            del self.host.vms[self.vm_id]
        logger.info(f"VM {self.vm_id} stopped")


class Container:
    """Container running on a VM or host."""
    
    def __init__(
        self,
        container_id: str,
        specs: ResourceSpecs,
        image: str = "default",
        labels: Optional[Dict[str, str]] = None,
        tolerations: Optional[Set[str]] = None,
    ):
        self.container_id = container_id
        self.specs = specs
        self.image = image
        self.labels = labels or {}
        self.tolerations = tolerations or set()
        self.state = ResourceState.AVAILABLE
        
        # Placement
        self.host: Optional[Host] = None
        self.vm: Optional[VirtualMachine] = None
        
        # Resource usage
        self.current_usage = ResourceUsage()
        
        # Startup time
        self.startup_time = 10.0  # seconds
        self.shutdown_time = 5.0  # seconds
        
        logger.debug(f"Container {container_id} created with image {image}")
    
    def schedule_on_host(self, host: Host) -> bool:
        """Schedule container directly on host."""
        if host.allocate_resources(self.specs):
            self.host = host
            self.state = ResourceState.ALLOCATED
            host.containers[self.container_id] = self
            logger.info(f"Container {self.container_id} scheduled on host {host.host_id}")
            return True
        return False
    
    def schedule_on_vm(self, vm: VirtualMachine) -> bool:
        """Schedule container on VM."""
        if vm.can_accommodate(self.specs):
            vm.allocated_specs.cpu_cores += self.specs.cpu_cores
            vm.allocated_specs.memory_gb += self.specs.memory_gb
            
            self.vm = vm
            self.host = vm.host
            self.state = ResourceState.ALLOCATED
            vm.containers[self.container_id] = self
            
            logger.info(f"Container {self.container_id} scheduled on VM {vm.vm_id}")
            return True
        return False
    
    def terminate(self) -> None:
        """Terminate the container."""
        if self.host:
            if self.vm:
                # Remove from VM
                self.vm.allocated_specs.cpu_cores -= self.specs.cpu_cores
                self.vm.allocated_specs.memory_gb -= self.specs.memory_gb
                if self.container_id in self.vm.containers:
                    del self.vm.containers[self.container_id]
            else:
                # Remove from host
                self.host.deallocate_resources(self.specs)
                if self.container_id in self.host.containers:
                    del self.host.containers[self.container_id]
        
        self.state = ResourceState.FAILED
        logger.info(f"Container {self.container_id} terminated")
