"""Data preprocessing utilities."""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from ..core.workload import Workload


class DataPreprocessor:
    """General data preprocessing utilities."""
    
    def __init__(self):
        self.logger = logger.bind(component="DataPreprocessor")
    
    def normalize_workloads(self, workloads: List[Workload]) -> List[Workload]:
        """Normalize workload resource requirements."""
        if not workloads:
            return workloads
        
        self.logger.info(f"Normalizing {len(workloads)} workloads")
        
        # Extract resource values
        cpu_values = [wl.specs.cpu_cores for wl in workloads]
        memory_values = [wl.specs.memory_gb for wl in workloads]
        
        # Calculate statistics
        cpu_stats = self._calculate_stats(cpu_values)
        memory_stats = self._calculate_stats(memory_values)
        
        # Normalize workloads
        normalized_workloads = []
        for workload in workloads:
            # Create normalized copy
            normalized_workload = workload
            
            # Apply normalization (remove extreme outliers)
            if workload.specs.cpu_cores > cpu_stats['q99']:
                normalized_workload.specs.cpu_cores = int(cpu_stats['q99'])
            if workload.specs.memory_gb > memory_stats['q99']:
                normalized_workload.specs.memory_gb = memory_stats['q99']
            
            normalized_workloads.append(normalized_workload)
        
        self.logger.info("Workload normalization completed")
        return normalized_workloads
    
    def filter_workloads(
        self,
        workloads: List[Workload],
        min_duration: float = 10.0,
        max_duration: float = 86400.0,
        min_resources: bool = True
    ) -> List[Workload]:
        """Filter workloads based on criteria."""
        self.logger.info(f"Filtering {len(workloads)} workloads")
        
        filtered = []
        for workload in workloads:
            # Duration filter
            if not (min_duration <= workload.duration <= max_duration):
                continue
            
            # Resource filter
            if min_resources:
                if workload.specs.cpu_cores < 1 or workload.specs.memory_gb < 0.5:
                    continue
            
            filtered.append(workload)
        
        self.logger.info(f"Filtered to {len(filtered)} workloads "
                        f"({len(workloads) - len(filtered)} removed)")
        return filtered
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'q95': 0, 'q99': 0}
        
        return {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'q95': np.percentile(values, 95),
            'q99': np.percentile(values, 99),
        }


class TracePreprocessor(DataPreprocessor):
    """Specialized preprocessor for cloud traces."""
    
    def __init__(self):
        super().__init__()
    
    def preprocess_google_trace(self, workloads: List[Workload]) -> List[Workload]:
        """Preprocess Google cluster trace data."""
        self.logger.info("Preprocessing Google trace data")
        
        # Apply general filtering
        filtered = self.filter_workloads(workloads)
        
        # Apply normalization
        normalized = self.normalize_workloads(filtered)
        
        # Sort by arrival time
        normalized.sort(key=lambda w: w.arrival_time)
        
        return normalized
    
    def preprocess_azure_trace(self, workloads: List[Workload]) -> List[Workload]:
        """Preprocess Azure trace data."""
        self.logger.info("Preprocessing Azure trace data")
        
        # Apply general filtering
        filtered = self.filter_workloads(workloads)
        
        # Apply normalization
        normalized = self.normalize_workloads(filtered)
        
        # Sort by arrival time
        normalized.sort(key=lambda w: w.arrival_time)
        
        return normalized
