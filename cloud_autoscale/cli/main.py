"""Command-line interface for Cloud AutoScale."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud_autoscale.config import load_config
from cloud_autoscale.data import SyntheticLoader, GCP2019Loader
from cloud_autoscale.simulation import CloudSimulator, BaselineAutoscaler, calculate_metrics, format_metrics_table
from cloud_autoscale.visualization import create_all_plots


def main():
    """Main entry point for Cloud AutoScale CLI."""
    parser = argparse.ArgumentParser(
        description='Cloud AutoScale - Baseline Autoscaling Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default synthetic data
  cloud-autoscale run --config config/baseline.yaml
  
  # Run with GCP 2019 data
  cloud-autoscale run --config config/gcp2019.yaml
  
  # Run with custom pattern
  cloud-autoscale run --config config/baseline.yaml --pattern bursty
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run simulation')
    run_parser.add_argument('--config', '-c', type=str, required=True,
                           help='Path to configuration file')
    run_parser.add_argument('--pattern', '-p', type=str, choices=['periodic', 'bursty', 'random_walk', 'spike'],
                           help='Override synthetic pattern (for synthetic mode)')
    run_parser.add_argument('--output', '-o', type=str,
                           help='Override output directory')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_simulation(args)
    else:
        parser.print_help()


def run_simulation(args):
    """Run the simulation."""
    print("=" * 70)
    print("CLOUD AUTOSCALE - Baseline Autoscaling Simulator")
    print("=" * 70)
    print()
    
    # Load configuration
    print(f"📋 Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    # Override pattern if specified
    if args.pattern:
        config['data']['synthetic_pattern'] = args.pattern
        print(f"   Override pattern: {args.pattern}")
    
    # Override output directory if specified
    if args.output:
        config['output']['directory'] = args.output
    
    print(f"   Mode: {config['mode']}")
    print()
    
    # Load data
    print("📊 Loading demand data...")
    try:
        if config['mode'] == 'synthetic':
            # Strict access - no defaults
            pattern = config['data']['synthetic_pattern']
            duration = config['data']['duration_minutes']
            step = config['simulation']['step_minutes']
            
            loader = SyntheticLoader(
                pattern=pattern,
                duration_minutes=duration,
                step_minutes=step
            )
            print(f"   Pattern: {pattern}")
            print(f"   Duration: {duration} minutes")
            
        elif config['mode'] == 'gcp_2019':
            # Use processed_dir instead of raw path
            processed_dir = config['data']['processed_dir']
            step = config['simulation']['step_minutes']
            
            # Duration is optional for GCP mode (use all available data if not specified)
            duration = config['data'].get('duration_minutes')  # Can be None
            
            loader = GCP2019Loader(
                processed_dir=processed_dir,
                step_minutes=step,
                duration_minutes=duration
            )
            print(f"   Processed data: {processed_dir}")
            if duration:
                print(f"   Duration limit: {duration} minutes")
            else:
                print(f"   Duration: Using all available data")
        
        else:
            # This should never happen if config validation works
            raise ValueError(f"Invalid mode: {config['mode']}")
        
        demand_df = loader.load()
        print(f"   ✓ Loaded {len(demand_df)} time steps")
        print()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize simulator
    print("🚀 Initializing simulator...")
    simulator = CloudSimulator(config['simulation'])
    print(f"   Step size: {config['simulation']['step_minutes']} minutes")
    print(f"   Machine capacity: {config['simulation']['machine_capacity']} units")
    print()
    
    # Initialize autoscaler
    print("⚙️  Initializing baseline autoscaler...")
    autoscaler_config = {**config['autoscaler'], 'step_minutes': config['simulation']['step_minutes']}
    autoscaler = BaselineAutoscaler(autoscaler_config)
    print(f"   Upper threshold: {config['autoscaler']['upper_threshold']}")
    print(f"   Lower threshold: {config['autoscaler']['lower_threshold']}")
    print()
    
    # Run simulation
    print("▶️  Running simulation...")
    try:
        results = simulator.run(demand_df, autoscaler)
        print("   ✓ Simulation completed")
        print()
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics = calculate_metrics(results)
    print()
    
    # Display metrics
    print(format_metrics_table(metrics))
    print()
    
    # Create output directory
    output_dir = Path(config['output']['directory'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"💾 Saving results to: {run_dir}")
    
    # Save timeline data
    timeline_path = run_dir / "timeline.csv"
    results['timeline'].to_csv(timeline_path, index=False)
    print(f"   ✓ Timeline data: {timeline_path}")
    
    # Save metrics
    import json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Metrics: {metrics_path}")
    
    # Save config
    import yaml
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   ✓ Configuration: {config_path}")
    
    print()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    try:
        plots_dir = run_dir / "plots"
        create_all_plots(results, metrics, plots_dir)
        print(f"   ✓ Plots saved to: {plots_dir}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create plots: {e}")
    
    print()
    print("=" * 70)
    print("✅ SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Results saved to: {run_dir}")
    print()


if __name__ == '__main__':
    main()

