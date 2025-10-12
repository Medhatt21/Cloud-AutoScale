#!/usr/bin/env uv run python
"""
Complete workflow example for using Google Cluster Trace data with the cloud scheduler.
This demonstrates the full process from data download to simulation.
"""

import sys
import os
from pathlib import Path
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cloud_scheduler.data.loaders import GoogleTraceLoader
from cloud_scheduler.core.simulator import CloudSimulator, SimulationConfig
from cloud_scheduler.scheduling.baseline import BaselineScheduler
from cloud_scheduler.evaluation.metrics import MetricsCollector

def main():
    """Run the complete Google trace workflow."""
    
    print("🌥️  Google Cluster Trace Workflow Example")
    print("=" * 50)
    
    # Step 1: Check if we have processed Google trace data
    trace_data_dir = project_root / "data" / "processed" / "google"
    
    if not trace_data_dir.exists():
        print("❌ No processed Google trace data found!")
        print("\nTo get the data, follow these steps:")
        print("1. Run setup: ./scripts/setup_google_traces.sh")
        print("2. Download trace: uv run python scripts/download_google_traces.py --trace 2019_05_a --sample-size 100000")
        print("3. Process data: uv run python scripts/integrate_google_traces.py data/raw/google/2019_05_a")
        print("\nOr run the quick demo with synthetic data instead:")
        print("  uv run python demo.py")
        sys.exit(1)
    
    # Step 2: Load the processed trace data
    print("\n📊 Loading Google trace data...")
    loader = GoogleTraceLoader(trace_data_dir)
    
    # Load workloads and machines
    workloads = loader.load_workloads(limit=1000)  # Limit for demo
    machines = loader.load_machines()
    
    if not workloads:
        print("❌ No workloads loaded!")
        sys.exit(1)
    
    if not machines:
        print("❌ No machines loaded!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(workloads)} workloads and {len(machines)} machines")
    
    # Step 3: Show data statistics
    print("\n📈 Data Statistics:")
    print(f"  Workloads: {len(workloads)}")
    print(f"  Time span: {max(w.arrival_time for w in workloads):.1f} seconds")
    print(f"  Total CPU requested: {sum(w.resource_specs.cpu_cores for w in workloads)} cores")
    print(f"  Total memory requested: {sum(w.resource_specs.memory_gb for w in workloads):.1f} GB")
    print()
    print(f"  Machines: {len(machines)}")
    print(f"  Total CPU capacity: {sum(m['cpu_cores'] for m in machines)} cores")
    print(f"  Total memory capacity: {sum(m['memory_gb'] for m in machines):.1f} GB")
    
    # Step 4: Create simulation configuration
    print("\n⚙️  Creating simulation configuration...")
    
    # Calculate simulation duration based on workload span
    max_arrival_time = max(w.arrival_time for w in workloads)
    simulation_duration = int(max_arrival_time + 1800)  # Add 30 minutes buffer
    
    config = SimulationConfig(
        simulation_duration=simulation_duration,
        workload_arrival_rate=len(workloads) / max_arrival_time if max_arrival_time > 0 else 1.0,
        enable_autoscaling=False,  # Start with baseline
        random_seed=42,
        log_level="INFO"
    )
    
    print(f"  Simulation duration: {simulation_duration} seconds ({simulation_duration/3600:.1f} hours)")
    print(f"  Workload arrival rate: {config.workload_arrival_rate:.3f} workloads/second")
    
    # Step 5: Create and configure the simulator
    print("\n🔧 Setting up simulator...")
    
    simulator = CloudSimulator(config)
    
    # Add machines (convert to Host objects)
    from cloud_scheduler.core.resources import Host, ResourceSpecs
    
    for machine_data in machines:
        capacity = ResourceSpecs(
            cpu_cores=machine_data['cpu_cores'],
            memory_gb=machine_data['memory_gb'],
            disk_gb=100.0,  # Default
            network_gbps=10.0,  # Default
            gpu_count=0,
            gpu_memory_gb=0.0
        )
        
        host = Host(
            host_id=machine_data['id'],
            capacity=capacity,
            zone=machine_data['zone'],
            instance_type=machine_data['instance_type'],
            cost_per_hour=0.05,  # Default cost
            attributes={
                'platform_id': machine_data.get('platform_id', 'unknown'),
                'switch_id': machine_data.get('switch_id', 'unknown')
            }
        )
        
        simulator.add_host(host)
    
    # Add workloads
    for workload in workloads:
        simulator.add_workload(workload)
    
    # Set up scheduler
    scheduler = BaselineScheduler()
    simulator.set_scheduler(scheduler)
    
    print(f"✅ Simulator ready with {len(machines)} hosts and {len(workloads)} workloads")
    
    # Step 6: Run the simulation
    print("\n🚀 Running simulation...")
    print("This may take a few minutes...")
    
    try:
        results = simulator.run()
        
        if results:
            metrics = results[-1]  # Get final metrics
            
            print("\n🎉 Simulation completed successfully!")
            print("\n📊 Results:")
            print(f"  Total workloads: {metrics.total_workloads}")
            print(f"  Completed: {metrics.completed_workloads}")
            print(f"  Failed: {metrics.failed_workloads}")
            print(f"  Success rate: {metrics.completed_workloads / max(1, metrics.total_workloads) * 100:.1f}%")
            print(f"  Average CPU utilization: {metrics.avg_cpu_utilization * 100:.1f}%")
            print(f"  Average memory utilization: {metrics.avg_memory_utilization * 100:.1f}%")
            print(f"  Resource efficiency: {metrics.resource_efficiency:.3f}")
            print(f"  Performance score: {metrics.performance_score:.3f}")
            
            # Save results
            results_file = project_root / "results" / "google_trace_results.json"
            results_file.parent.mkdir(exist_ok=True)
            
            results_data = {
                'simulation_config': {
                    'duration': config.simulation_duration,
                    'workloads': len(workloads),
                    'hosts': len(machines)
                },
                'metrics': {
                    'total_workloads': metrics.total_workloads,
                    'completed_workloads': metrics.completed_workloads,
                    'failed_workloads': metrics.failed_workloads,
                    'success_rate': metrics.completed_workloads / max(1, metrics.total_workloads),
                    'avg_cpu_utilization': metrics.avg_cpu_utilization,
                    'avg_memory_utilization': metrics.avg_memory_utilization,
                    'resource_efficiency': metrics.resource_efficiency,
                    'performance_score': metrics.performance_score
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
            
        else:
            print("❌ Simulation failed - no results returned")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Workflow completed successfully!")
    print("\nNext steps:")
    print("1. Try different scheduling algorithms (ML, RL)")
    print("2. Compare results with baseline")
    print("3. Experiment with different trace datasets")
    print("4. Run evaluation: uv run cloud-sim evaluate --config configs/baseline.yaml")

if __name__ == '__main__':
    main()
