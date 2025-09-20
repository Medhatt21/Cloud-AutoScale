#!/usr/bin/env python3
"""
Demonstration script for the Cloud Scheduler Simulator.

This script shows how to use the simulator with different configurations
and demonstrates the basic functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cloud_scheduler.core.simulator import CloudSimulator, SimulationConfig
from cloud_scheduler.core.resources import Host, ResourceSpecs, ResourceType
from cloud_scheduler.scheduling.baseline import FirstFitScheduler, BestFitScheduler
from cloud_scheduler.scheduling.autoscaling import ThresholdAutoscaler, AutoscalingConfig
from cloud_scheduler.evaluation.metrics import SimulationAnalyzer
from cloud_scheduler.data.loaders import SyntheticTraceLoader
from loguru import logger


def create_infrastructure() -> list[Host]:
    """Create a sample infrastructure for the demo."""
    hosts = []
    
    # Create different types of hosts
    host_configs = [
        (ResourceType.GENERAL_PURPOSE, ResourceSpecs(4, 16.0, 100.0, 1.0), 5),
        (ResourceType.CPU_OPTIMIZED, ResourceSpecs(8, 16.0, 100.0, 2.0), 3),
        (ResourceType.MEMORY_OPTIMIZED, ResourceSpecs(4, 32.0, 100.0, 1.0), 2),
    ]
    
    host_id = 1
    for resource_type, specs, count in host_configs:
        for i in range(count):
            host = Host(
                host_id=f"demo-host-{host_id:02d}",
                specs=specs,
                resource_type=resource_type,
                zone=f"zone-{(host_id - 1) % 2 + 1}",  # 2 zones
            )
            hosts.append(host)
            host_id += 1
    
    return hosts


def run_baseline_demo():
    """Run a baseline scheduling demonstration."""
    logger.info("🚀 Starting Baseline Scheduling Demo")
    
    # Configuration
    config = SimulationConfig(
        simulation_duration=1800.0,  # 30 minutes
        random_seed=42,
        enable_failures=False,  # Disable for demo simplicity
        enable_autoscaling=True,
        workload_arrival_rate=0.5,  # 0.5 workloads per second
    )
    
    # Create simulator
    simulator = CloudSimulator(config)
    
    # Setup infrastructure
    hosts = create_infrastructure()
    simulator.add_hosts(hosts)
    logger.info(f"📊 Created infrastructure with {len(hosts)} hosts")
    
    # Setup scheduler
    scheduler = FirstFitScheduler()
    simulator.set_scheduler(scheduler)
    logger.info("📋 Using First-Fit Scheduler")
    
    # Setup autoscaler
    autoscaler_config = AutoscalingConfig(
        cpu_scale_up_threshold=0.7,
        cpu_scale_down_threshold=0.3,
        min_hosts=3,
        max_hosts=15,
    )
    autoscaler = ThresholdAutoscaler(autoscaler_config)
    simulator.set_autoscaler(autoscaler)
    logger.info("⚡ Using Threshold Autoscaler")
    
    # Run simulation
    logger.info("🔄 Running simulation...")
    metrics_history = simulator.run()
    
    # Analyze results
    analyzer = SimulationAnalyzer()
    analysis = analyzer.analyze_simulation(
        metrics_history, 
        simulator.completed_workloads,
        simulator.failed_workloads
    )
    
    # Print results
    print_results(analysis)


def run_comparison_demo():
    """Run a comparison between different schedulers."""
    logger.info("🔬 Starting Scheduler Comparison Demo")
    
    schedulers = {
        "First-Fit": FirstFitScheduler(),
        "Best-Fit": BestFitScheduler(),
    }
    
    results = {}
    
    for name, scheduler in schedulers.items():
        logger.info(f"🧪 Testing {name} scheduler")
        
        # Configuration
        config = SimulationConfig(
            simulation_duration=900.0,  # 15 minutes
            random_seed=42,
            enable_failures=False,
            enable_autoscaling=False,  # Disable for fair comparison
            workload_arrival_rate=1.0,
        )
        
        # Create simulator
        simulator = CloudSimulator(config)
        hosts = create_infrastructure()
        simulator.add_hosts(hosts)
        simulator.set_scheduler(scheduler)
        
        # Run simulation
        metrics_history = simulator.run()
        
        # Analyze results
        analyzer = SimulationAnalyzer()
        analysis = analyzer.analyze_simulation(
            metrics_history,
            simulator.completed_workloads,
            simulator.failed_workloads
        )
        
        results[name] = analysis
    
    # Compare results
    print_comparison(results)


def print_results(analysis: dict):
    """Print simulation results in a formatted way."""
    print("\n" + "="*60)
    print("📊 SIMULATION RESULTS")
    print("="*60)
    
    summary = analysis['summary']
    sla_metrics = analysis['sla_metrics']
    resource_metrics = analysis['resource_metrics']
    
    print(f"🎯 Workloads:")
    print(f"   Total: {summary['total_workloads']}")
    print(f"   Completed: {summary['completed_workloads']}")
    print(f"   Failed: {summary['failed_workloads']}")
    
    print(f"\n📈 SLA Metrics:")
    print(f"   Violation Rate: {sla_metrics['sla_violation_rate']:.2%}")
    print(f"   Avg Queue Time: {sla_metrics['avg_queue_time']:.2f}s")
    print(f"   Avg Execution Time: {sla_metrics['avg_execution_time']:.2f}s")
    
    print(f"\n💾 Resource Metrics:")
    print(f"   Avg CPU Utilization: {resource_metrics['avg_cpu_utilization']:.2%}")
    print(f"   Avg Memory Utilization: {resource_metrics['avg_memory_utilization']:.2%}")
    print(f"   Resource Efficiency: {resource_metrics['resource_efficiency']:.2%}")
    
    print(f"\n⭐ Overall Performance Score: {analysis['performance_score']:.3f}")
    print("="*60)


def print_comparison(results: dict):
    """Print comparison results between different methods."""
    print("\n" + "="*70)
    print("🔬 SCHEDULER COMPARISON")
    print("="*70)
    
    print(f"{'Method':<15} {'SLA Violations':<15} {'CPU Util':<12} {'Performance':<12}")
    print("-" * 70)
    
    for method, analysis in results.items():
        sla_rate = analysis['sla_metrics']['sla_violation_rate']
        cpu_util = analysis['resource_metrics']['avg_cpu_utilization']
        performance = analysis['performance_score']
        
        print(f"{method:<15} {sla_rate:<15.2%} {cpu_util:<12.2%} {performance:<12.3f}")
    
    print("="*70)


def main():
    """Main demonstration function."""
    print("🌟 Cloud Scheduler Simulator Demo")
    print("This demo showcases the cloud scheduling and autoscaling simulator")
    print()
    
    try:
        # Run baseline demo
        run_baseline_demo()
        
        print("\n" + "⏳ Waiting before comparison demo...")
        import time
        time.sleep(2)
        
        # Run comparison demo
        run_comparison_demo()
        
        print("\n✅ Demo completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Try different configurations in configs/")
        print("   2. Implement ML-based scheduling")
        print("   3. Add reinforcement learning agents")
        print("   4. Load real cloud traces for evaluation")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
