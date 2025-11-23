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
from cloud_autoscale.visualization import create_all_plots, create_comparison_plots
from cloud_autoscale.forecasting import ForecastingModel
from cloud_autoscale.simulation.autoscaler_proactive import ProactiveAutoscaler
from cloud_autoscale.simulation.metrics import compare_metrics
import json
import pandas as pd


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
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare baseline vs proactive autoscaling')
    compare_parser.add_argument('--baseline', type=str, required=True,
                               help='Path to baseline configuration file')
    compare_parser.add_argument('--proactive', type=str, required=True,
                               help='Path to proactive configuration file')
    compare_parser.add_argument('--model', type=str, required=False,
                               help='Override model path for proactive autoscaler')
    compare_parser.add_argument('--output', '-o', type=str, required=False,
                               help='Override output directory for comparison results')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_simulation(args)
    elif args.command == 'compare':
        run_comparison(args)
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
            # Strict access - all parameters required
            pattern = config['data']['synthetic_pattern']
            duration = config['data']['duration_minutes']
            step = config['simulation']['step_minutes']
            seed = 42  # Fixed seed for reproducibility
            
            loader = SyntheticLoader(
                pattern=pattern,
                duration_minutes=duration,
                step_minutes=step,
                seed=seed
            )
            print(f"   Pattern: {pattern}")
            print(f"   Duration: {duration} minutes")
            print(f"   Seed: {seed}")
            
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
    try:
        simulator = CloudSimulator(config['simulation'])
        print(f"   Step size: {config['simulation']['step_minutes']} minutes")
        print(f"   Machine capacity: {config['simulation']['machine_capacity']} units")
        print(f"   Machine range: {config['simulation']['min_machines']}-{config['simulation']['max_machines']}")
        print()
    except Exception as e:
        print(f"❌ Error initializing simulator: {e}")
        sys.exit(1)
    
    # Initialize autoscaler
    autoscaler_type = config['autoscaler'].get('type', 'baseline')
    print(f"⚙️  Initializing {autoscaler_type} autoscaler...")
    try:
        if autoscaler_type == 'baseline':
            autoscaler = BaselineAutoscaler(
                autoscaler_config=config['autoscaler'],
                step_minutes=config['simulation']['step_minutes']
            )
            print(f"   Upper threshold: {config['autoscaler']['upper_threshold']}")
            print(f"   Lower threshold: {config['autoscaler']['lower_threshold']}")
            print(f"   Max scale per step: {config['autoscaler']['max_scale_per_step']}")
            print(f"   Cooldown steps: {config['autoscaler']['cooldown_steps']}")
        
        elif autoscaler_type == 'proactive':
            # Load forecasting model
            model_run_dir = config['autoscaler'].get('model_run_dir', 'latest')
            
            if model_run_dir == 'latest':
                # Auto-detect latest run with modeling artifacts
                results_base = Path(config['output']['directory'])
                run_dirs = sorted(results_base.glob('run_*'))
                
                # Find the latest run with modeling directory
                model_run_dir = None
                for run_dir in reversed(run_dirs):
                    if (run_dir / 'modeling').exists():
                        model_run_dir = run_dir
                        break
                
                if model_run_dir is None:
                    raise FileNotFoundError(
                        f"No trained model found in {results_base}\n"
                        f"Please run the modeling notebook first to train and save models."
                    )
                print(f"   Auto-detected model: {model_run_dir.name}")
            else:
                model_run_dir = Path(model_run_dir)
                if not model_run_dir.exists():
                    raise FileNotFoundError(f"Model run directory not found: {model_run_dir}")
                print(f"   Using model: {model_run_dir}")
            
            # Load forecasting model
            forecast_model = ForecastingModel(model_run_dir)
            print(f"   ✓ Loaded forecasting model")
            
            # Create proactive autoscaler
            autoscaler = ProactiveAutoscaler(
                forecast_model=forecast_model,
                autoscaler_config=config['autoscaler'],
                step_minutes=config['simulation']['step_minutes'],
                min_machines=config['simulation']['min_machines'],
                max_machines=config['simulation']['max_machines']
            )
            print(f"   Upper threshold: {config['autoscaler']['upper_threshold']}")
            print(f"   Lower threshold: {config['autoscaler']['lower_threshold']}")
            print(f"   Max scale per step: {config['autoscaler']['max_scale_per_step']}")
            print(f"   Cooldown steps: {config['autoscaler']['cooldown_steps']}")
            print(f"   Safety margin: {config['autoscaler'].get('safety_margin', 1.10)}")
            print(f"   History window: {config['autoscaler'].get('history_window', 200)}")
        
        else:
            raise ValueError(f"Invalid autoscaler type: {autoscaler_type}")
        
        print()
    except Exception as e:
        print(f"❌ Error initializing autoscaler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run simulation
    print("▶️  Running simulation...")
    try:
        results = simulator.run(
            demand_df,
            autoscaler,
            output_dir=config['output']['directory'],
            save_results=True
        )
        print("   ✓ Simulation completed")
        print()
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get run directory from results
    run_dir = Path(results.get('run_dir', config['output']['directory']))
    
    # Calculate metrics (already in results)
    metrics = results['metrics']
    
    # Display metrics summary
    print("📈 Simulation Results:")
    print()
    print(f"   Total Violations:    {metrics['total_violations']}")
    print(f"   Violation Rate:      {metrics['violation_rate']:.2%}")
    print(f"   Avg Utilization:     {metrics['avg_utilization']:.2%}")
    print(f"   Avg Machines:        {metrics['avg_machines']:.1f}")
    print(f"   Total Scale Events:  {metrics['total_scale_events']}")
    print(f"   Total Cost:          ${metrics['total_cost']:.2f}")
    print()
    
    # Save config
    import yaml
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 Configuration saved: {config_path}")
    
    # Create visualizations
    print("📊 Creating visualizations...")
    try:
        plots_dir = run_dir / "plots"
        create_all_plots(results, metrics, plots_dir, config.get('autoscaler'))
        print(f"   ✓ All plots saved to: {plots_dir}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create plots: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("✅ SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Results saved to: {run_dir}")
    print(f"  - Timeline: {run_dir}/timeline.csv")
    print(f"  - Metrics: {run_dir}/metrics.json")
    print(f"  - Events: {run_dir}/scale_events.csv")
    print(f"  - Plots: {run_dir}/plots/")
    print()


def run_comparison(args):
    """Run comparison between baseline and proactive autoscalers."""
    print("=" * 70)
    print("CLOUD AUTOSCALE - Baseline vs Proactive Comparison")
    print("=" * 70)
    print()
    
    # Create comparison output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        comparison_dir = Path(args.output)
    else:
        comparison_dir = Path("results") / f"comparison_{timestamp}"
    
    comparison_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = comparison_dir / "baseline"
    proactive_dir = comparison_dir / "proactive"
    
    print(f"📁 Comparison directory: {comparison_dir}")
    print()
    
    # ========== Run Baseline Simulation ==========
    print("🔵 Running BASELINE simulation...")
    print("-" * 70)
    
    try:
        baseline_config = load_config(args.baseline)
        # Override output directory
        baseline_config['output']['directory'] = str(baseline_dir)
        
        # Load data
        demand_df_baseline = load_demand_data(baseline_config)
        print(f"   ✓ Loaded {len(demand_df_baseline)} time steps")
        
        # Initialize simulator and autoscaler
        simulator_baseline = CloudSimulator(baseline_config['simulation'])
        autoscaler_baseline = BaselineAutoscaler(
            autoscaler_config=baseline_config['autoscaler'],
            step_minutes=baseline_config['simulation']['step_minutes']
        )
        
        # Run simulation
        print("   Running baseline simulation...")
        baseline_results = simulator_baseline.run(
            demand_df_baseline,
            autoscaler_baseline,
            output_dir=str(baseline_dir),
            save_results=True
        )
        
        baseline_run_dir = Path(baseline_results['run_dir'])
        baseline_metrics = baseline_results['metrics']
        
        print(f"   ✓ Baseline completed")
        print(f"      Violations: {baseline_metrics['total_violations']}")
        print(f"      Cost: ${baseline_metrics['total_cost']:.2f}")
        print(f"      Avg Utilization: {baseline_metrics['avg_utilization']:.2%}")
        print()
        
    except Exception as e:
        print(f"❌ Error running baseline simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== Run Proactive Simulation ==========
    print("🟣 Running PROACTIVE simulation...")
    print("-" * 70)
    
    try:
        proactive_config = load_config(args.proactive)
        # Override output directory
        proactive_config['output']['directory'] = str(proactive_dir)
        
        # Override model path if provided
        if args.model:
            proactive_config['autoscaler']['model_run_dir'] = args.model
            print(f"   Using model: {args.model}")
        
        # Load data (should be same as baseline)
        demand_df_proactive = load_demand_data(proactive_config)
        print(f"   ✓ Loaded {len(demand_df_proactive)} time steps")
        
        # Initialize simulator
        simulator_proactive = CloudSimulator(proactive_config['simulation'])
        
        # Initialize proactive autoscaler
        model_run_dir = proactive_config['autoscaler'].get('model_run_dir', 'latest')
        
        if model_run_dir == 'latest':
            # Auto-detect latest run with modeling artifacts
            results_base = Path("results")
            run_dirs = sorted(results_base.glob('run_*'))
            
            model_run_dir = None
            for run_dir in reversed(run_dirs):
                if (run_dir / 'modeling').exists():
                    model_run_dir = run_dir
                    break
            
            if model_run_dir is None:
                raise FileNotFoundError(
                    f"No trained model found in {results_base}\n"
                    f"Please run the modeling notebook first."
                )
            print(f"   Auto-detected model: {model_run_dir.name}")
        else:
            model_run_dir = Path(model_run_dir)
            if not model_run_dir.exists():
                raise FileNotFoundError(f"Model run directory not found: {model_run_dir}")
        
        forecast_model = ForecastingModel(model_run_dir)
        autoscaler_proactive = ProactiveAutoscaler(
            forecast_model=forecast_model,
            autoscaler_config=proactive_config['autoscaler'],
            step_minutes=proactive_config['simulation']['step_minutes'],
            min_machines=proactive_config['simulation']['min_machines'],
            max_machines=proactive_config['simulation']['max_machines']
        )
        
        # Run simulation
        print("   Running proactive simulation...")
        proactive_results = simulator_proactive.run(
            demand_df_proactive,
            autoscaler_proactive,
            output_dir=str(proactive_dir),
            save_results=True
        )
        
        proactive_run_dir = Path(proactive_results['run_dir'])
        proactive_metrics = proactive_results['metrics']
        
        print(f"   ✓ Proactive completed")
        print(f"      Violations: {proactive_metrics['total_violations']}")
        print(f"      Cost: ${proactive_metrics['total_cost']:.2f}")
        print(f"      Avg Utilization: {proactive_metrics['avg_utilization']:.2%}")
        print()
        
    except Exception as e:
        print(f"❌ Error running proactive simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== Merge Timelines ==========
    print("📊 Merging timelines and computing comparison metrics...")
    
    try:
        baseline_timeline = baseline_results['timeline']
        proactive_timeline = proactive_results['timeline']
        
        # Merge on step
        merged_df = pd.merge(
            baseline_timeline,
            proactive_timeline,
            on='step',
            suffixes=('_baseline', '_proactive')
        )
        
        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'time_baseline': 'time',
            'demand_baseline': 'baseline_demand',
            'capacity_baseline': 'baseline_capacity',
            'machines_baseline': 'baseline_machines',
            'utilization_baseline': 'baseline_utilization',
            'violation_baseline': 'baseline_violation',
            'demand_proactive': 'proactive_demand',
            'capacity_proactive': 'proactive_capacity',
            'machines_proactive': 'proactive_machines',
            'utilization_proactive': 'proactive_utilization',
            'violation_proactive': 'proactive_violation'
        })
        
        # Select relevant columns
        merged_df = merged_df[[
            'step', 'time',
            'baseline_demand', 'baseline_capacity', 'baseline_machines',
            'baseline_utilization', 'baseline_violation',
            'proactive_capacity', 'proactive_machines',
            'proactive_utilization', 'proactive_violation'
        ]]
        
        # Save merged timeline
        merged_csv_path = comparison_dir / "comparison_timeline.csv"
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"   ✓ Merged timeline saved: {merged_csv_path}")
        
    except Exception as e:
        print(f"❌ Error merging timelines: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== Compute Comparison Metrics ==========
    try:
        comparison_metrics_dict = compare_metrics(baseline_metrics, proactive_metrics)
        
        # Save comparison metrics
        metrics_json_path = comparison_dir / "comparison_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(comparison_metrics_dict, f, indent=4)
        print(f"   ✓ Comparison metrics saved: {metrics_json_path}")
        print()
        
    except Exception as e:
        print(f"❌ Error computing comparison metrics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== Create Comparison Plots ==========
    print("📈 Creating comparison visualizations...")
    
    try:
        plots_dir = comparison_dir / "plots"
        create_comparison_plots(
            baseline_results,
            proactive_results,
            merged_df,
            comparison_metrics_dict,
            plots_dir
        )
        print()
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create comparison plots: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== Display Summary ==========
    print("=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    print()
    print(f"SLA Performance:")
    print(f"  Violation Reduction:     {comparison_metrics_dict['violation_reduction_percent']:>6.1f}%")
    print(f"  Baseline Violations:     {comparison_metrics_dict['baseline']['violations']:>6d}")
    print(f"  Proactive Violations:    {comparison_metrics_dict['proactive']['violations']:>6d}")
    print()
    print(f"Cost Efficiency:")
    print(f"  Cost Savings:            {comparison_metrics_dict['cost_savings_percent']:>6.1f}%")
    print(f"  Baseline Cost:          ${comparison_metrics_dict['baseline']['cost']:>7.2f}")
    print(f"  Proactive Cost:         ${comparison_metrics_dict['proactive']['cost']:>7.2f}")
    print()
    print(f"Resource Utilization:")
    print(f"  Utilization Gain:        {comparison_metrics_dict['avg_utilization_gain']:>6.1f} pp")
    print(f"  Baseline Utilization:    {comparison_metrics_dict['baseline']['avg_utilization']*100:>6.1f}%")
    print(f"  Proactive Utilization:   {comparison_metrics_dict['proactive']['avg_utilization']*100:>6.1f}%")
    print()
    print(f"System Stability:")
    print(f"  Scaling Event Change:    {comparison_metrics_dict['stability_change']:>6d}")
    print(f"  Baseline Events:         {comparison_metrics_dict['baseline']['scale_events']:>6d}")
    print(f"  Proactive Events:        {comparison_metrics_dict['proactive']['scale_events']:>6d}")
    print()
    print("=" * 70)
    print("✅ COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Results saved to: {comparison_dir}")
    print(f"  - Baseline run: {baseline_run_dir}")
    print(f"  - Proactive run: {proactive_run_dir}")
    print(f"  - Merged timeline: {merged_csv_path}")
    print(f"  - Comparison metrics: {metrics_json_path}")
    print(f"  - Comparison plots: {plots_dir}/")
    print()


def load_demand_data(config: dict) -> pd.DataFrame:
    """
    Load demand data based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Demand DataFrame
    """
    if config['mode'] == 'synthetic':
        pattern = config['data']['synthetic_pattern']
        duration = config['data']['duration_minutes']
        step = config['simulation']['step_minutes']
        seed = 42
        
        loader = SyntheticLoader(
            pattern=pattern,
            duration_minutes=duration,
            step_minutes=step,
            seed=seed
        )
    
    elif config['mode'] == 'gcp_2019':
        processed_dir = config['data']['processed_dir']
        step = config['simulation']['step_minutes']
        duration = config['data'].get('duration_minutes')
        
        loader = GCP2019Loader(
            processed_dir=processed_dir,
            step_minutes=step,
            duration_minutes=duration
        )
    
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")
    
    return loader.load()


if __name__ == '__main__':
    main()

