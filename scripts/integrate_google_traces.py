#!/usr/bin/env uv run python
"""
Google Trace Data Integration
Converts Google cluster trace data to the format expected by the cloud scheduler.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleTraceIntegrator:
    def __init__(self, trace_dir: str):
        """Initialize with path to downloaded Google trace data."""
        self.trace_dir = Path(trace_dir)
        self.metadata = self._load_metadata()
        
        # Load all tables
        self.tables = {}
        for table_name in ['machine_events', 'machine_attributes', 'collection_events', 'instance_events', 'instance_usage']:
            parquet_file = self.trace_dir / f"{table_name}.parquet"
            csv_file = self.trace_dir / f"{table_name}.csv"
            
            if parquet_file.exists():
                self.tables[table_name] = pd.read_parquet(parquet_file)
                logger.info(f"Loaded {table_name}: {len(self.tables[table_name]):,} records")
            elif csv_file.exists():
                self.tables[table_name] = pd.read_csv(csv_file)
                logger.info(f"Loaded {table_name}: {len(self.tables[table_name]):,} records")
            else:
                logger.warning(f"Table {table_name} not found")
                self.tables[table_name] = pd.DataFrame()
    
    def _load_metadata(self) -> Dict:
        """Load trace metadata."""
        metadata_file = self.trace_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def extract_machine_configs(self) -> List[Dict]:
        """Extract machine configurations from machine events."""
        logger.info("Extracting machine configurations...")
        
        if 'machine_events' not in self.tables or self.tables['machine_events'].empty:
            logger.warning("No machine events data available")
            return []
        
        df = self.tables['machine_events']
        
        # Get latest ADD/UPDATE event for each machine
        df = df[df['type'].isin([0, 2])]  # ADD=0, UPDATE=2
        df = df.sort_values(['machine_id', 'time']).groupby('machine_id').last().reset_index()
        
        machines = []
        for _, row in df.iterrows():
            # Parse capacity (assuming it's a JSON string or already parsed)
            capacity = row.get('capacity', {})
            if isinstance(capacity, str):
                try:
                    capacity = json.loads(capacity)
                except:
                    capacity = {'cpus': 0.1, 'memory': 0.1}  # Default fallback
            
            machine = {
                'id': f"host_{row['machine_id']}",
                'cpu_cores': max(1, int(capacity.get('cpus', 0.1) * 100)),  # Scale up from normalized
                'memory_gb': max(1, int(capacity.get('memory', 0.1) * 1000)),  # Scale up from normalized
                'platform_id': row.get('platform_id', 'unknown'),
                'switch_id': row.get('switch_id', 'unknown'),
                'zone': f"zone_{hash(str(row.get('switch_id', 'unknown'))) % 3}",  # Simulate zones
                'instance_type': self._infer_instance_type(capacity.get('cpus', 0.1), capacity.get('memory', 0.1))
            }
            machines.append(machine)
        
        logger.info(f"Extracted {len(machines)} machine configurations")
        return machines
    
    def _infer_instance_type(self, cpu: float, memory: float) -> str:
        """Infer AWS-like instance type from resources."""
        # Normalize to approximate instance types
        cpu_scaled = cpu * 100  # Scale from [0,1] to reasonable range
        mem_scaled = memory * 1000  # Scale from [0,1] to GB
        
        if cpu_scaled <= 2 and mem_scaled <= 4:
            return "t3.small"
        elif cpu_scaled <= 4 and mem_scaled <= 8:
            return "t3.medium"
        elif cpu_scaled <= 8 and mem_scaled <= 16:
            return "t3.large"
        elif cpu_scaled <= 16 and mem_scaled <= 32:
            return "t3.xlarge"
        else:
            return "t3.2xlarge"
    
    def extract_workload_patterns(self, duration_hours: int = 24) -> List[Dict]:
        """Extract workload patterns from instance events and usage."""
        logger.info(f"Extracting workload patterns for {duration_hours} hours...")
        
        if 'instance_events' not in self.tables or self.tables['instance_events'].empty:
            logger.warning("No instance events data available")
            return []
        
        df = self.tables['instance_events'].copy()
        
        # Convert timestamps (microseconds since 600s before trace start)
        df['time_seconds'] = (df['time'] - 600_000_000) / 1_000_000  # Convert to seconds from trace start
        
        # Filter to specified duration
        max_time = duration_hours * 3600
        df = df[df['time_seconds'] <= max_time]
        
        # Focus on SUBMIT events to understand workload arrival patterns
        submit_events = df[df['type'] == 0]  # SUBMIT=0
        
        workloads = []
        for _, row in submit_events.iterrows():
            # Parse resource request
            resource_request = row.get('resource_request', {})
            if isinstance(resource_request, str):
                try:
                    resource_request = json.loads(resource_request)
                except:
                    resource_request = {'cpus': 0.01, 'memory': 0.01}
            
            workload = {
                'id': f"workload_{row['collection_id']}_{row['instance_index']}",
                'collection_id': row['collection_id'],
                'arrival_time': max(0, row['time_seconds']),
                'cpu_request': max(0.1, resource_request.get('cpus', 0.01) * 100),  # Scale up
                'memory_request': max(0.1, resource_request.get('memory', 0.01) * 1000),  # Scale up to GB
                'priority': row.get('priority', 100),
                'scheduling_class': row.get('scheduling_class', 1),
                'workload_type': self._infer_workload_type(row.get('priority', 100)),
                'estimated_duration': self._estimate_duration(row['collection_id']),
                'user': row.get('user', 'unknown'),
                'collection_name': row.get('collection_name', 'unknown')
            }
            workloads.append(workload)
        
        # Sort by arrival time
        workloads.sort(key=lambda x: x['arrival_time'])
        
        logger.info(f"Extracted {len(workloads)} workload patterns")
        return workloads
    
    def _infer_workload_type(self, priority: int) -> str:
        """Map Google priority to workload type."""
        if priority <= 99:
            return "batch"
        elif priority <= 115:
            return "best_effort_batch"
        elif priority <= 119:
            return "mid_tier"
        elif priority <= 359:
            return "production"
        else:
            return "monitoring"
    
    def _estimate_duration(self, collection_id: str) -> float:
        """Estimate workload duration from usage data or use default."""
        if 'instance_usage' in self.tables and not self.tables['instance_usage'].empty:
            usage_data = self.tables['instance_usage'][
                self.tables['instance_usage']['collection_id'] == collection_id
            ]
            if not usage_data.empty:
                # Use the span of usage measurements as duration estimate
                start_times = usage_data['start_time'] / 1_000_000  # Convert to seconds
                end_times = usage_data['end_time'] / 1_000_000
                duration = (end_times.max() - start_times.min())
                return max(60, duration)  # At least 1 minute
        
        # Default durations based on workload type (in seconds)
        return np.random.lognormal(mean=6, sigma=1.5)  # Log-normal distribution, mean ~400s
    
    def generate_arrival_pattern(self, workloads: List[Dict], target_duration: int = 3600) -> List[Dict]:
        """Generate realistic arrival pattern over target duration."""
        logger.info(f"Generating arrival pattern over {target_duration} seconds...")
        
        if not workloads:
            return []
        
        # Scale arrival times to target duration
        original_max_time = max(w['arrival_time'] for w in workloads)
        if original_max_time > 0:
            time_scale = target_duration / original_max_time
        else:
            time_scale = 1.0
        
        scaled_workloads = []
        for workload in workloads:
            scaled_workload = workload.copy()
            scaled_workload['arrival_time'] = workload['arrival_time'] * time_scale
            scaled_workloads.append(scaled_workload)
        
        # Add some realistic noise to arrival times
        for workload in scaled_workloads:
            noise = np.random.exponential(scale=10)  # Exponential noise
            workload['arrival_time'] = max(0, workload['arrival_time'] + noise)
        
        # Re-sort by arrival time
        scaled_workloads.sort(key=lambda x: x['arrival_time'])
        
        logger.info(f"Generated arrival pattern with {len(scaled_workloads)} workloads")
        return scaled_workloads
    
    def save_processed_data(self, output_dir: str, machines: List[Dict], workloads: List[Dict]):
        """Save processed data in the format expected by the simulator."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save machine configurations
        machines_file = output_path / 'machines.json'
        with open(machines_file, 'w') as f:
            json.dump(machines, f, indent=2)
        logger.info(f"Saved {len(machines)} machines to {machines_file}")
        
        # Save workload patterns
        workloads_file = output_path / 'workloads.json'
        with open(workloads_file, 'w') as f:
            json.dump(workloads, f, indent=2)
        logger.info(f"Saved {len(workloads)} workloads to {workloads_file}")
        
        # Save summary statistics
        stats = {
            'trace_info': self.metadata,
            'machines': {
                'count': len(machines),
                'total_cpu_cores': sum(m['cpu_cores'] for m in machines),
                'total_memory_gb': sum(m['memory_gb'] for m in machines),
                'instance_types': list(set(m['instance_type'] for m in machines))
            },
            'workloads': {
                'count': len(workloads),
                'duration_span': max(w['arrival_time'] for w in workloads) if workloads else 0,
                'total_cpu_request': sum(w['cpu_request'] for w in workloads),
                'total_memory_request': sum(w['memory_request'] for w in workloads),
                'workload_types': list(set(w['workload_type'] for w in workloads))
            }
        }
        
        stats_file = output_path / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Integrate Google trace data with cloud scheduler')
    parser.add_argument('trace_dir', help='Directory containing downloaded Google trace data')
    parser.add_argument('--output-dir', default='data/processed/google', help='Output directory')
    parser.add_argument('--duration-hours', type=int, default=1, help='Simulation duration in hours')
    
    args = parser.parse_args()
    
    try:
        integrator = GoogleTraceIntegrator(args.trace_dir)
        
        # Extract data
        machines = integrator.extract_machine_configs()
        workloads = integrator.extract_workload_patterns(args.duration_hours)
        
        # Generate realistic arrival pattern
        target_duration = args.duration_hours * 3600
        workloads = integrator.generate_arrival_pattern(workloads, target_duration)
        
        # Save processed data
        stats = integrator.save_processed_data(args.output_dir, machines, workloads)
        
        # Print summary
        print("\n🎉 Data integration complete!")
        print(f"📊 Machines: {stats['machines']['count']:,}")
        print(f"📊 Total CPU cores: {stats['machines']['total_cpu_cores']:,}")
        print(f"📊 Total memory: {stats['machines']['total_memory_gb']:,.1f} GB")
        print(f"📊 Workloads: {stats['workloads']['count']:,}")
        print(f"📊 Simulation duration: {args.duration_hours} hours")
        print(f"📁 Output saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        raise

if __name__ == '__main__':
    main()
