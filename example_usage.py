"""Example usage of Cloud AutoScale simulator."""

from pathlib import Path
from cloud_autoscale import (
    SyntheticLoader,
    CloudSimulator,
    BaselineAutoscaler,
    calculate_metrics,
    format_metrics_table,
    create_all_plots
)

def main():
    """Run a simple example simulation."""
    
    print("=" * 70)
    print("Cloud AutoScale - Example Usage")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        'step_minutes': 5,
        'min_machines': 1,
        'max_machines': 20,
        'machine_capacity': 10,
        'cost_per_machine_per_hour': 0.1
    }
    
    autoscaler_config = {
        'upper_threshold': 0.7,
        'lower_threshold': 0.3,
        'max_scale_per_step': 1,
        'cooldown_steps': 2
    }
    
    # Load synthetic data
    print("Loading synthetic demand data...")
    loader = SyntheticLoader(
        pattern="bursty",
        duration_minutes=60,
        step_minutes=5,
        seed=42
    )
    demand_df = loader.load()
    print(f"Loaded {len(demand_df)} time steps")
    print()
    
    # Initialize simulator and autoscaler
    print("Initializing simulator...")
    simulator = CloudSimulator(config)
    autoscaler = BaselineAutoscaler(
        autoscaler_config=autoscaler_config,
        step_minutes=config['step_minutes']
    )
    print()
    
    # Run simulation
    print("Running simulation...")
    results = simulator.run(demand_df, autoscaler)
    print("Simulation completed!")
    print()
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    print()
    
    # Display metrics
    print(format_metrics_table(metrics))
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    output_dir = Path("example_results")
    create_all_plots(results, metrics, output_dir)
    print(f"Plots saved to: {output_dir}")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

