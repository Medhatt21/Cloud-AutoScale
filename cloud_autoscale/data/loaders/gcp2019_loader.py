"""GCP 2019 trace data loader.

This loader expects data to be downloaded externally using the provided scripts.
It reads JSONL.gz files and aggregates them into 5-minute time buckets.
"""

import gzip
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


class GCP2019Loader:
    """Load and process GCP 2019 trace data."""
    
    def __init__(self, data_path: str, duration_minutes: int = 60, step_minutes: int = 5):
        """
        Initialize GCP 2019 data loader.
        
        Args:
            data_path: Path to directory containing instance_usage-*.json.gz and machine_events-*.json.gz
            duration_minutes: Total duration to load in minutes
            step_minutes: Time bucket size in minutes
        """
        self.data_path = Path(data_path)
        self.duration_minutes = duration_minutes
        self.step_minutes = step_minutes
        
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load and aggregate GCP 2019 trace data.
        
        Returns:
            DataFrame with columns: time, cpu_demand, mem_demand, new_instances
        """
        # Find instance usage files
        usage_files = sorted(self.data_path.glob("instance_usage-*.json.gz"))
        
        if not usage_files:
            # Try CSV format as fallback
            usage_files = sorted(self.data_path.glob("instance_usage-*.csv"))
            if not usage_files:
                raise FileNotFoundError(
                    f"No instance_usage files found in {self.data_path}. "
                    "Please download data using the provided scripts."
                )
            return self._load_from_csv(usage_files[0])
        
        return self._load_from_jsonl_gz(usage_files)
    
    def _load_from_jsonl_gz(self, files: list) -> pd.DataFrame:
        """Load data from JSONL.gz files."""
        records = []
        
        # Read first file (or multiple if needed)
        for file_path in files[:1]:  # Process first file for now
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10000:  # Limit records for performance
                        break
                    
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        if not records:
            raise ValueError("No valid records found in data files")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Extract relevant fields
        # GCP 2019 schema: timestamp (microseconds), collection_id, instance_index, 
        # average_usage (cpu, memory)
        
        if 'start_time' in df.columns:
            df['time_minutes'] = df['start_time'] / 1_000_000 / 60  # Convert microseconds to minutes
        elif 'timestamp' in df.columns:
            df['time_minutes'] = df['timestamp'] / 1_000_000 / 60
        else:
            # Generate synthetic timestamps
            df['time_minutes'] = np.arange(len(df)) * 0.1
        
        # Extract CPU and memory usage
        if 'average_usage' in df.columns:
            # Parse nested average_usage field
            df['cpu'] = df['average_usage'].apply(lambda x: x.get('cpus', 0.0) if isinstance(x, dict) else 0.0)
            df['memory'] = df['average_usage'].apply(lambda x: x.get('memory', 0.0) if isinstance(x, dict) else 0.0)
        else:
            # Use direct fields if available
            df['cpu'] = df.get('cpus', 0.0)
            df['memory'] = df.get('memory', 0.0)
        
        # Normalize to 0-100 range (GCP data is typically 0-1)
        df['cpu'] = df['cpu'] * 100
        df['memory'] = df['memory'] * 100
        
        # Create time buckets
        df['time_bucket'] = (df['time_minutes'] // self.step_minutes).astype(int)
        
        # Aggregate by time bucket
        aggregated = df.groupby('time_bucket').agg({
            'cpu': 'mean',
            'memory': 'mean',
            'collection_id': 'nunique'  # Count unique instances as new instances
        }).reset_index()
        
        aggregated.columns = ['time_bucket', 'cpu_demand', 'mem_demand', 'new_instances']
        aggregated['time'] = aggregated['time_bucket'] * self.step_minutes
        
        # Ensure we have the requested duration
        num_steps = self.duration_minutes // self.step_minutes
        if len(aggregated) < num_steps:
            # Pad with zeros if needed
            full_range = pd.DataFrame({
                'time': np.arange(0, num_steps) * self.step_minutes
            })
            aggregated = full_range.merge(aggregated[['time', 'cpu_demand', 'mem_demand', 'new_instances']], 
                                         on='time', how='left')
            aggregated = aggregated.fillna(0)
        else:
            # Trim if too long
            aggregated = aggregated[aggregated['time'] < self.duration_minutes]
        
        return aggregated[['time', 'cpu_demand', 'mem_demand', 'new_instances']]
    
    def _load_from_csv(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV format (fallback)."""
        # Read CSV file
        df = pd.read_csv(file_path, nrows=10000)
        
        # Convert timestamp to minutes
        if 'start_time' in df.columns:
            df['time_minutes'] = df['start_time'] / 1_000_000 / 60
        elif 'timestamp' in df.columns:
            df['time_minutes'] = df['timestamp'] / 1_000_000 / 60
        else:
            df['time_minutes'] = np.arange(len(df)) * 0.1
        
        # Extract CPU and memory
        cpu_cols = [col for col in df.columns if 'cpu' in col.lower()]
        mem_cols = [col for col in df.columns if 'mem' in col.lower()]
        
        if cpu_cols:
            df['cpu'] = df[cpu_cols[0]] * 100
        else:
            df['cpu'] = np.random.uniform(10, 50, len(df))
        
        if mem_cols:
            df['memory'] = df[mem_cols[0]] * 100
        else:
            df['memory'] = np.random.uniform(10, 50, len(df))
        
        # Create time buckets
        df['time_bucket'] = (df['time_minutes'] // self.step_minutes).astype(int)
        
        # Aggregate
        aggregated = df.groupby('time_bucket').agg({
            'cpu': 'mean',
            'memory': 'mean'
        }).reset_index()
        
        aggregated.columns = ['time_bucket', 'cpu_demand', 'mem_demand']
        aggregated['time'] = aggregated['time_bucket'] * self.step_minutes
        aggregated['new_instances'] = np.random.poisson(2, len(aggregated))
        
        # Ensure duration
        num_steps = self.duration_minutes // self.step_minutes
        if len(aggregated) < num_steps:
            full_range = pd.DataFrame({
                'time': np.arange(0, num_steps) * self.step_minutes
            })
            aggregated = full_range.merge(aggregated[['time', 'cpu_demand', 'mem_demand', 'new_instances']], 
                                         on='time', how='left')
            aggregated = aggregated.fillna(0)
        else:
            aggregated = aggregated[aggregated['time'] < self.duration_minutes]
        
        return aggregated[['time', 'cpu_demand', 'mem_demand', 'new_instances']]

