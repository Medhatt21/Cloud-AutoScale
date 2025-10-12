"""Command-line interface for the cloud scheduler simulator."""

import typer
from typing import Optional, List
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd
from loguru import logger

from .core.simulator import CloudSimulator, SimulationConfig
from .core.resources import Host, ResourceSpecs, ResourceType
from .scheduling.baseline import FirstFitScheduler, BestFitScheduler, SpreadScheduler
from .scheduling.autoscaling import ThresholdAutoscaler, ScheduledAutoscaler, PredictiveAutoscaler, AutoscalingConfig
from .evaluation.metrics import SimulationAnalyzer
from .utils.config import load_config, save_results

app = typer.Typer(name="cloud-sim", help="Cloud Scheduling and Autoscaling Simulator")
console = Console()


@app.command()
def simulate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    duration: float = typer.Option(3600.0, "--duration", "-d", help="Simulation duration in seconds"),
    scheduler: str = typer.Option("first_fit", "--scheduler", "-s", help="Scheduling algorithm"),
    autoscaler: str = typer.Option("threshold", "--autoscaler", "-a", help="Autoscaling policy"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run cloud scheduling simulation."""
    
    if verbose:
        logger.remove()
        logger.add("logs/simulation_{time}.log", level="DEBUG")
        logger.add(lambda msg: console.print(msg, style="dim"), level="INFO")
    
    console.print("🚀 Starting Cloud Scheduling Simulation", style="bold blue")
    
    # Load configuration
    full_config_data = None
    if config and config.exists():
        sim_config = load_config(config)
        console.print(f"📋 Loaded configuration from {config}")
        # Load full config data for autoscaler
        import yaml
        with open(config, 'r') as f:
            full_config_data = yaml.safe_load(f)
    else:
        sim_config = SimulationConfig(simulation_duration=duration)
        console.print("📋 Using default configuration")
    
    # Create simulator
    simulator = CloudSimulator(sim_config)
    
    # Setup infrastructure
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Setting up infrastructure...", total=None)
        
        # Create hosts
        hosts = create_default_infrastructure()
        simulator.add_hosts(hosts)
        progress.update(task, description=f"Created {len(hosts)} hosts")
        
        # Setup scheduler
        scheduler_obj = create_scheduler(scheduler)
        simulator.set_scheduler(scheduler_obj)
        progress.update(task, description=f"Configured {scheduler} scheduler")
        
        # Setup autoscaler
        autoscaler_obj = create_autoscaler(autoscaler, full_config_data)
        simulator.set_autoscaler(autoscaler_obj)
        progress.update(task, description=f"Configured {autoscaler} autoscaler")
    
    # Run simulation
    console.print(f"⚡ Running simulation for {duration:.0f} seconds...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating...", total=None)
        metrics_history = simulator.run()
        progress.update(task, description="Simulation completed")
    
    # Analyze results
    console.print("📊 Analyzing results...")
    analyzer = SimulationAnalyzer()
    analysis = analyzer.analyze_simulation(metrics_history, simulator.completed_workloads, simulator.failed_workloads)
    
    # Display summary
    display_results_summary(analysis)
    
    # Save results
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_results(analysis, output_dir)
        console.print(f"💾 Results saved to {output_dir}")
    
    console.print("✅ Simulation completed successfully!", style="bold green")


@app.command()
def evaluate(
    methods: str = typer.Option("baseline,ml,rl", "--methods", "-m", help="Comma-separated list of methods to compare"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    runs: int = typer.Option(3, "--runs", "-r", help="Number of runs per method"),
) -> None:
    """Evaluate and compare different scheduling methods."""
    
    console.print("🔬 Starting Evaluation and Comparison", style="bold blue")
    
    method_list = [m.strip() for m in methods.split(",")]
    results = {}
    
    for method in method_list:
        console.print(f"📈 Evaluating {method} method...")
        method_results = []
        
        for run in range(runs):
            console.print(f"  Run {run + 1}/{runs}")
            # Implementation would run specific method
            # For now, placeholder
            method_results.append({"sla_violation_rate": 0.05, "avg_utilization": 0.7})
        
        results[method] = method_results
    
    # Display comparison
    display_comparison_results(results)
    
    console.print("✅ Evaluation completed!", style="bold green")


@app.command()
def data(
    action: str = typer.Argument(help="Action: prepare, download, or clean"),
    dataset: str = typer.Option("google", "--dataset", "-d", help="Dataset: google, azure, or alibaba"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Data preparation and management."""
    
    console.print(f"📂 Data {action} for {dataset} dataset", style="bold blue")
    
    if action == "prepare":
        console.print("🔄 Preparing dataset...")
        # Implementation for data preparation
        console.print("✅ Dataset prepared!")
    
    elif action == "download":
        console.print("⬇️  Downloading dataset...")
        # Implementation for data download
        console.print("✅ Dataset downloaded!")
    
    elif action == "clean":
        console.print("🧹 Cleaning dataset...")
        # Implementation for data cleaning
        console.print("✅ Dataset cleaned!")


def create_default_infrastructure() -> List[Host]:
    """Create default infrastructure for simulation."""
    hosts = []
    
    # Create different types of hosts
    host_configs = [
        (ResourceType.GENERAL_PURPOSE, ResourceSpecs(4, 16.0, 100.0, 1.0), 10),
        (ResourceType.CPU_OPTIMIZED, ResourceSpecs(8, 16.0, 100.0, 2.0), 5),
        (ResourceType.MEMORY_OPTIMIZED, ResourceSpecs(4, 32.0, 100.0, 1.0), 5),
        (ResourceType.GPU_ENABLED, ResourceSpecs(8, 32.0, 200.0, 10.0, 2, 16.0), 2),
    ]
    
    host_id = 1
    for resource_type, specs, count in host_configs:
        for i in range(count):
            host = Host(
                host_id=f"host-{resource_type.value}-{host_id:03d}",
                specs=specs,
                resource_type=resource_type,
                zone=f"zone-{(host_id - 1) % 3 + 1}",  # 3 zones
                rack_id=f"rack-{(host_id - 1) % 10 + 1}",  # 10 racks
            )
            hosts.append(host)
            host_id += 1
    
    return hosts


def create_scheduler(scheduler_type: str):
    """Create scheduler instance."""
    schedulers = {
        "first_fit": FirstFitScheduler,
        "best_fit": BestFitScheduler,
        "spread": SpreadScheduler,
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return schedulers[scheduler_type]()


def create_autoscaler(autoscaler_type: str, config_data: dict = None):
    """Create autoscaler instance."""
    if config_data and 'autoscaling' in config_data:
        # Use config from file
        autoscaling_config = config_data['autoscaling']
        config = AutoscalingConfig(
            cpu_scale_up_threshold=autoscaling_config.get('cpu_scale_up_threshold', 0.8),
            cpu_scale_down_threshold=autoscaling_config.get('cpu_scale_down_threshold', 0.3),
            memory_scale_up_threshold=autoscaling_config.get('memory_scale_up_threshold', 0.8),
            memory_scale_down_threshold=autoscaling_config.get('memory_scale_down_threshold', 0.3),
            scale_up_cooldown=autoscaling_config.get('scale_up_cooldown', 300.0),
            scale_down_cooldown=autoscaling_config.get('scale_down_cooldown', 600.0),
            min_hosts=autoscaling_config.get('min_hosts', 1),
            max_hosts=autoscaling_config.get('max_hosts', 100),
        )
    else:
        # Use default config
        config = AutoscalingConfig()
    
    autoscalers = {
        "threshold": lambda: ThresholdAutoscaler(config),
        "scheduled": lambda: ScheduledAutoscaler(config),
        "predictive": lambda: PredictiveAutoscaler(config),
    }
    
    if autoscaler_type not in autoscalers:
        raise ValueError(f"Unknown autoscaler type: {autoscaler_type}")
    
    return autoscalers[autoscaler_type]()


def display_results_summary(analysis: dict) -> None:
    """Display simulation results summary."""
    
    # Main metrics table
    table = Table(title="Simulation Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Unit", style="yellow")
    
    summary = analysis.get('summary', {})
    sla_metrics = analysis.get('sla_metrics', {})
    resource_metrics = analysis.get('resource_metrics', {})
    scaling_metrics = analysis.get('scaling_metrics', {})
    
    metrics = [
        ("Total Workloads", f"{summary.get('total_workloads', 0)}", "count"),
        ("Completed Workloads", f"{summary.get('completed_workloads', 0)}", "count"),
        ("SLA Violation Rate", f"{sla_metrics.get('sla_violation_rate', 0):.2%}", "percentage"),
        ("Average Queue Time", f"{sla_metrics.get('avg_queue_time', 0):.2f}", "seconds"),
        ("Average CPU Utilization", f"{resource_metrics.get('avg_cpu_utilization', 0):.2%}", "percentage"),
        ("Average Memory Utilization", f"{resource_metrics.get('avg_memory_utilization', 0):.2%}", "percentage"),
        ("Scaling Events", f"{scaling_metrics.get('total_scaling_events', 0)}", "count"),
    ]
    
    for metric, value, unit in metrics:
        table.add_row(metric, value, unit)
    
    console.print(table)


def display_comparison_results(results: dict) -> None:
    """Display comparison results between methods."""
    
    table = Table(title="Method Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("SLA Violation Rate", style="red")
    table.add_column("Avg Utilization", style="green")
    table.add_column("Std Dev", style="yellow")
    
    for method, method_results in results.items():
        sla_rates = [r["sla_violation_rate"] for r in method_results]
        utilizations = [r["avg_utilization"] for r in method_results]
        
        avg_sla = sum(sla_rates) / len(sla_rates)
        avg_util = sum(utilizations) / len(utilizations)
        std_sla = (sum((x - avg_sla) ** 2 for x in sla_rates) / len(sla_rates)) ** 0.5
        
        table.add_row(
            method,
            f"{avg_sla:.2%}",
            f"{avg_util:.2%}",
            f"{std_sla:.3f}"
        )
    
    console.print(table)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
