"""Data loaders for cloud traces and datasets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
# import polars as pl  # Optional dependency
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from ..core.workload import Workload, WorkloadType, WorkloadPriority
from ..core.resources import ResourceSpecs


class CloudTraceLoader(ABC):
    """Abstract base class for cloud trace loaders."""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.logger = logger.bind(component=self.__class__.__name__)
    
    @abstractmethod
    def load_workloads(self, limit: Optional[int] = None) -> List[Workload]:
        """Load workloads from trace data."""
        pass
    
    @abstractmethod
    def load_resource_usage(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load resource usage data."""
        pass
    
    def validate_data_path(self) -> bool:
        """Validate that data path exists and contains expected files."""
        return self.data_path.exists()


class GoogleTraceLoader(CloudTraceLoader):
    """Loader for Google Cluster Trace 2019 data."""
    
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.expected_files = [
            "instance_events.csv",
            "instance_usage.csv", 
            "collection_events.csv",
            "machine_events.csv"
        ]
    
    def validate_data_path(self) -> bool:
        """Validate Google trace data structure."""
        if not self.data_path.exists():
            self.logger.error(f"Data path does not exist: {self.data_path}")
            return False
        
        missing_files = []
        for filename in self.expected_files:
            if not (self.data_path / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            self.logger.warning(f"Missing Google trace files: {missing_files}")
            # Try to find compressed versions
            for filename in missing_files[:]:
                compressed_file = self.data_path / f"{filename}.gz"
                if compressed_file.exists():
                    missing_files.remove(filename)
                    self.logger.info(f"Found compressed version: {compressed_file}")
        
        return len(missing_files) == 0
    
    def load_workloads(self, limit: Optional[int] = None) -> List[Workload]:
        """Load workloads from Google trace data (processed JSON or raw CSV)."""
        self.logger.info(f"Loading Google trace workloads (limit: {limit})")
        
        # First try to load processed JSON format (from our integration script)
        workloads_json = self.data_path / "workloads.json"
        if workloads_json.exists():
            return self._load_processed_workloads(workloads_json, limit)
        
        # Fallback to raw CSV format
        instance_events_file = self.data_path / "instance_events.csv"
        if not instance_events_file.exists():
            instance_events_file = self.data_path / "instance_events.csv.gz"
        
        if not instance_events_file.exists():
            self.logger.error("Neither workloads.json nor instance_events.csv found")
            return []
        
        try:
            # Use Pandas for loading (fallback without Polars)
            df = pd.read_csv(instance_events_file, nrows=limit)
            self.logger.info(f"Loaded {len(df)} instance events")
            
            workloads = []
            workload_counter = 0
            
            # Convert to workloads
            for _, row in df.iterrows():
                if row.get('type') == 'SUBMIT':  # Job submission
                    workload_counter += 1
                    
                    # Extract resource requirements
                    cpu_request = row.get('cpu_request', 0.1)
                    memory_request = row.get('memory_request', 0.1)
                    
                    # Convert normalized values to actual specs
                    specs = ResourceSpecs(
                        cpu_cores=max(1, int(cpu_request * 32)),  # Scale to reasonable range
                        memory_gb=max(0.5, memory_request * 64),  # Scale to GB
                        disk_gb=10.0,  # Default disk
                        network_gbps=1.0,  # Default network
                    )
                    
                    # Determine workload type based on resource pattern
                    workload_type = self._infer_workload_type(cpu_request, memory_request)
                    
                    # Determine priority
                    priority = self._infer_priority(row.get('priority', 0))
                    
                    # Create workload
                    workload = Workload(
                        workload_id=f"google_{workload_counter:08d}",
                        workload_type=workload_type,
                        priority=priority,
                        specs=specs,
                        arrival_time=row.get('timestamp', 0) / 1e6,  # Convert microseconds
                        duration=max(60.0, row.get('runtime', 300.0) / 1e6),  # Convert microseconds
                        user_id=str(row.get('user', 'unknown')),
                        job_id=str(row.get('collection_id', 'unknown')),
                    )
                    
                    workloads.append(workload)
            
            self.logger.info(f"Created {len(workloads)} workloads from Google trace")
            return workloads
            
        except Exception as e:
            self.logger.error(f"Error loading Google trace workloads: {e}")
            return []
    
    def load_resource_usage(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load resource usage data from Google trace."""
        self.logger.info(f"Loading Google trace resource usage (limit: {limit})")
        
        usage_file = self.data_path / "instance_usage.csv"
        if not usage_file.exists():
            usage_file = self.data_path / "instance_usage.csv.gz"
        
        if not usage_file.exists():
            self.logger.error("Instance usage file not found")
            return pd.DataFrame()
        
        try:
            # Load with Pandas
            df = pd.read_csv(usage_file, nrows=limit)
            self.logger.info(f"Loaded {len(df)} usage records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Google trace usage: {e}")
            return pd.DataFrame()
    
    def _load_processed_workloads(self, workloads_file: Path, limit: Optional[int] = None) -> List[Workload]:
        """Load workloads from processed JSON format."""
        import json
        
        self.logger.info(f"Loading processed workloads from {workloads_file}")
        
        try:
            with open(workloads_file, 'r') as f:
                workload_data = json.load(f)
            
            if limit:
                workload_data = workload_data[:limit]
            
            workloads = []
            for i, data in enumerate(workload_data):
                # Map workload types
                workload_type_map = {
                    'batch': WorkloadType.BATCH,
                    'best_effort_batch': WorkloadType.BATCH,
                    'mid_tier': WorkloadType.SERVICE,
                    'production': WorkloadType.SERVICE,
                    'monitoring': WorkloadType.INTERACTIVE
                }
                
                workload_type = workload_type_map.get(
                    data.get('workload_type', 'batch'), 
                    WorkloadType.BATCH
                )
                
                # Create resource specifications
                resource_specs = ResourceSpecs(
                    cpu_cores=max(1, int(data.get('cpu_request', 1))),
                    memory_gb=max(0.5, float(data.get('memory_request', 1))),
                    disk_gb=20.0,
                    network_gbps=1.0,
                    gpu_count=0,
                    gpu_memory_gb=0.0
                )
                
                # Map priority
                priority = self._infer_priority(data.get('priority', 100))
                
                # Create workload
                workload = Workload(
                    workload_id=data.get('id', f'workload_{i}'),
                    workload_type=workload_type,
                    arrival_time=float(data.get('arrival_time', i * 10)),
                    resource_specs=resource_specs,
                    duration=float(data.get('estimated_duration', 300)),
                    priority=priority,
                    user_id=str(data.get('user', 'unknown')),
                    constraints={}
                )
                
                workloads.append(workload)
            
            self.logger.info(f"Loaded {len(workloads)} processed workloads")
            return workloads
            
        except Exception as e:
            self.logger.error(f"Failed to load processed workloads: {e}")
            return []
    
    def load_machines(self) -> List[Dict]:
        """Load machine configurations from processed JSON format."""
        machines_json = self.data_path / "machines.json"
        if machines_json.exists():
            return self._load_processed_machines(machines_json)
        
        self.logger.warning("No machines.json found")
        return []
    
    def _load_processed_machines(self, machines_file: Path) -> List[Dict]:
        """Load machines from processed JSON format."""
        import json
        
        self.logger.info(f"Loading processed machines from {machines_file}")
        
        try:
            with open(machines_file, 'r') as f:
                machines_data = json.load(f)
            
            machines = []
            for data in machines_data:
                machine = {
                    'id': data.get('id', f"host_{len(machines)}"),
                    'cpu_cores': int(data.get('cpu_cores', 4)),
                    'memory_gb': float(data.get('memory_gb', 8)),
                    'zone': data.get('zone', 'us-east-1a'),
                    'instance_type': data.get('instance_type', 't3.medium'),
                    'platform_id': data.get('platform_id', 'unknown'),
                    'switch_id': data.get('switch_id', 'unknown')
                }
                machines.append(machine)
            
            self.logger.info(f"Loaded {len(machines)} machine configurations")
            return machines
            
        except Exception as e:
            self.logger.error(f"Failed to load processed machines: {e}")
            return []
    
    def _infer_workload_type(self, cpu_request: float, memory_request: float) -> WorkloadType:
        """Infer workload type from resource pattern."""
        cpu_to_memory_ratio = cpu_request / max(memory_request, 0.001)
        
        if cpu_to_memory_ratio > 2.0:
            return WorkloadType.BATCH  # CPU-intensive
        elif memory_request > 0.5:
            return WorkloadType.ML_TRAINING  # Memory-intensive
        elif cpu_request < 0.1 and memory_request < 0.1:
            return WorkloadType.INTERACTIVE  # Light workload
        else:
            return WorkloadType.SERVICE  # Balanced workload
    
    def _infer_priority(self, priority_value: int) -> WorkloadPriority:
        """Infer workload priority from trace priority value."""
        if priority_value >= 9:
            return WorkloadPriority.CRITICAL
        elif priority_value >= 7:
            return WorkloadPriority.HIGH
        elif priority_value >= 4:
            return WorkloadPriority.MEDIUM
        elif priority_value >= 1:
            return WorkloadPriority.LOW
        else:
            return WorkloadPriority.BEST_EFFORT


class AzureTraceLoader(CloudTraceLoader):
    """Loader for Azure Public Dataset V2 (2019) data."""
    
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.expected_files = [
            "vmtable.csv",
            "vm_cpu_utilization.csv",
            "vm_memory_utilization.csv"
        ]
    
    def validate_data_path(self) -> bool:
        """Validate Azure trace data structure."""
        if not self.data_path.exists():
            self.logger.error(f"Data path does not exist: {self.data_path}")
            return False
        
        missing_files = []
        for filename in self.expected_files:
            if not (self.data_path / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            self.logger.warning(f"Missing Azure trace files: {missing_files}")
        
        return len(missing_files) == 0
    
    def load_workloads(self, limit: Optional[int] = None) -> List[Workload]:
        """Load workloads from Azure trace VM data."""
        self.logger.info(f"Loading Azure trace workloads (limit: {limit})")
        
        vm_table_file = self.data_path / "vmtable.csv"
        if not vm_table_file.exists():
            self.logger.error("VM table file not found")
            return []
        
        try:
            # Load VM table
            df = pd.read_csv(vm_table_file, nrows=limit)
            self.logger.info(f"Loaded {len(df)} VM records")
            
            workloads = []
            
            for _, row in df.iterrows():
                # Extract VM specifications
                vm_category = row.get('vmCategory', 'Unknown')
                vm_cores = row.get('vmCores', 1)
                vm_memory = row.get('vmMemory', 1.0)
                
                # Create resource specs
                specs = ResourceSpecs(
                    cpu_cores=max(1, int(vm_cores)),
                    memory_gb=max(0.5, float(vm_memory)),
                    disk_gb=50.0,  # Default
                    network_gbps=1.0,  # Default
                )
                
                # Infer workload type from VM category
                workload_type = self._infer_workload_type_from_category(vm_category)
                
                # Create workload
                workload = Workload(
                    workload_id=f"azure_{row.get('vmId', 'unknown')}",
                    workload_type=workload_type,
                    priority=WorkloadPriority.MEDIUM,  # Default priority
                    specs=specs,
                    arrival_time=0.0,  # Will be set from usage data if available
                    duration=3600.0,  # Default 1 hour
                    user_id=str(row.get('subscriptionId', 'unknown')),
                    job_id=str(row.get('deploymentId', 'unknown')),
                )
                
                workloads.append(workload)
            
            self.logger.info(f"Created {len(workloads)} workloads from Azure trace")
            return workloads
            
        except Exception as e:
            self.logger.error(f"Error loading Azure trace workloads: {e}")
            return []
    
    def load_resource_usage(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load resource usage data from Azure trace."""
        self.logger.info(f"Loading Azure trace resource usage (limit: {limit})")
        
        # Load CPU utilization
        cpu_file = self.data_path / "vm_cpu_utilization.csv"
        memory_file = self.data_path / "vm_memory_utilization.csv"
        
        usage_data = []
        
        if cpu_file.exists():
            try:
                cpu_df = pd.read_csv(cpu_file, nrows=limit)
                cpu_df['metric_type'] = 'cpu'
                usage_data.append(cpu_df)
                self.logger.info(f"Loaded {len(cpu_df)} CPU usage records")
            except Exception as e:
                self.logger.error(f"Error loading CPU usage: {e}")
        
        if memory_file.exists():
            try:
                mem_df = pd.read_csv(memory_file, nrows=limit)
                mem_df['metric_type'] = 'memory'
                usage_data.append(mem_df)
                self.logger.info(f"Loaded {len(mem_df)} memory usage records")
            except Exception as e:
                self.logger.error(f"Error loading memory usage: {e}")
        
        if usage_data:
            combined_df = pd.concat(usage_data, ignore_index=True)
            self.logger.info(f"Combined {len(combined_df)} total usage records")
            return combined_df
        else:
            return pd.DataFrame()
    
    def _infer_workload_type_from_category(self, vm_category: str) -> WorkloadType:
        """Infer workload type from Azure VM category."""
        category_lower = vm_category.lower()
        
        if 'compute' in category_lower or 'cpu' in category_lower:
            return WorkloadType.BATCH
        elif 'memory' in category_lower or 'ram' in category_lower:
            return WorkloadType.ML_TRAINING
        elif 'web' in category_lower or 'frontend' in category_lower:
            return WorkloadType.WEB_SERVER
        elif 'general' in category_lower:
            return WorkloadType.SERVICE
        else:
            return WorkloadType.SERVICE  # Default


class SyntheticTraceLoader(CloudTraceLoader):
    """Loader for synthetic trace data (for testing and development)."""
    
    def __init__(self, data_path: Optional[Path] = None):
        super().__init__(data_path or Path("data/synthetic"))
        self.num_workloads = 1000
        self.time_span = 3600.0  # 1 hour
    
    def validate_data_path(self) -> bool:
        """Always valid for synthetic data."""
        return True
    
    def load_workloads(self, limit: Optional[int] = None) -> List[Workload]:
        """Generate synthetic workloads."""
        num_workloads = min(limit or self.num_workloads, self.num_workloads)
        self.logger.info(f"Generating {num_workloads} synthetic workloads")
        
        workloads = []
        np.random.seed(42)  # For reproducibility
        
        # Workload type probabilities
        workload_types = list(WorkloadType)
        type_probs = [0.3, 0.25, 0.2, 0.15, 0.1]  # Batch, Service, Interactive, ML, Web
        
        # Generate arrival times (Poisson process)
        arrival_rate = num_workloads / self.time_span
        inter_arrival_times = np.random.exponential(1.0 / arrival_rate, num_workloads)
        arrival_times = np.cumsum(inter_arrival_times)
        
        for i in range(num_workloads):
            # Choose workload type
            workload_type = np.random.choice(workload_types, p=type_probs)
            
            # Generate resource specs based on type
            specs = self._generate_specs_for_type(workload_type)
            
            # Generate priority
            priority = np.random.choice(list(WorkloadPriority), p=[0.1, 0.2, 0.4, 0.2, 0.1])
            
            # Generate duration
            duration = self._generate_duration_for_type(workload_type)
            
            workload = Workload(
                workload_id=f"synthetic_{i+1:06d}",
                workload_type=workload_type,
                priority=priority,
                specs=specs,
                arrival_time=arrival_times[i] if i < len(arrival_times) else i * 10.0,
                duration=duration,
                user_id=f"user_{np.random.randint(1, 101)}",
                job_id=f"job_{np.random.randint(1, 1001)}",
            )
            
            workloads.append(workload)
        
        self.logger.info(f"Generated {len(workloads)} synthetic workloads")
        return workloads
    
    def load_resource_usage(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic resource usage data."""
        self.logger.info("Generating synthetic resource usage data")
        
        # Generate time series data
        timestamps = np.arange(0, self.time_span, 60)  # Every minute
        num_hosts = 20
        
        usage_records = []
        
        for host_id in range(1, num_hosts + 1):
            for timestamp in timestamps:
                # Generate realistic usage patterns
                base_cpu = 0.3 + 0.4 * np.sin(2 * np.pi * timestamp / 3600)  # Hourly cycle
                base_memory = 0.4 + 0.3 * np.sin(2 * np.pi * timestamp / 1800)  # 30-min cycle
                
                # Add noise
                cpu_usage = max(0.05, min(0.95, base_cpu + np.random.normal(0, 0.1)))
                memory_usage = max(0.05, min(0.95, base_memory + np.random.normal(0, 0.1)))
                
                usage_records.append({
                    'timestamp': timestamp,
                    'host_id': f"host_{host_id:03d}",
                    'cpu_utilization': cpu_usage,
                    'memory_utilization': memory_usage,
                    'network_utilization': np.random.uniform(0.1, 0.5),
                    'disk_utilization': np.random.uniform(0.2, 0.8),
                })
        
        df = pd.DataFrame(usage_records)
        self.logger.info(f"Generated {len(df)} usage records")
        return df
    
    def _generate_specs_for_type(self, workload_type: WorkloadType) -> ResourceSpecs:
        """Generate resource specs for workload type."""
        base_specs = {
            WorkloadType.BATCH: (4, 8.0, 50.0, 1.0, 0, 0.0),
            WorkloadType.SERVICE: (2, 4.0, 20.0, 1.0, 0, 0.0),
            WorkloadType.INTERACTIVE: (1, 2.0, 10.0, 0.5, 0, 0.0),
            WorkloadType.ML_TRAINING: (8, 32.0, 100.0, 5.0, 1, 16.0),
            WorkloadType.WEB_SERVER: (2, 4.0, 30.0, 2.0, 0, 0.0),
        }
        
        base_cpu, base_mem, base_disk, base_net, base_gpu, base_gpu_mem = base_specs[workload_type]
        
        # Add variation
        cpu_mult = np.random.lognormal(0, 0.3)
        mem_mult = np.random.lognormal(0, 0.3)
        
        return ResourceSpecs(
            cpu_cores=max(1, int(base_cpu * cpu_mult)),
            memory_gb=max(0.5, base_mem * mem_mult),
            disk_gb=base_disk,
            network_gbps=base_net,
            gpu_count=base_gpu,
            gpu_memory_gb=base_gpu_mem,
        )
    
    def _generate_duration_for_type(self, workload_type: WorkloadType) -> float:
        """Generate duration for workload type."""
        base_durations = {
            WorkloadType.BATCH: 600.0,  # 10 minutes
            WorkloadType.SERVICE: 300.0,  # 5 minutes
            WorkloadType.INTERACTIVE: 120.0,  # 2 minutes
            WorkloadType.ML_TRAINING: 3600.0,  # 1 hour
            WorkloadType.WEB_SERVER: 1800.0,  # 30 minutes
        }
        
        base_duration = base_durations[workload_type]
        return max(30.0, np.random.lognormal(np.log(base_duration), 0.5))
