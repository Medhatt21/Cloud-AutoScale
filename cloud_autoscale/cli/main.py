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
from cloud_autoscale.rl.train import train_rl
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

    # Run RL Training command
    run_rl_parser = subparsers.add_parser('run-rl', help='Run RL training')
    run_rl_parser.add_argument('--config', '-c', type=str, required=True,
                           help='Path to RL configuration file')
    run_rl_parser.add_argument('--output', '-o', type=str,
                           help='Output directory for model and logs')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare baseline vs proactive vs RL autoscaling')
    compare_parser.add_argument('--baseline', type=str, required=True,
                               help='Path to baseline configuration file')
    compare_parser.add_argument('--proactive', type=str, required=False,
                               help='Path to proactive configuration file')
    compare_parser.add_argument('--rl', type=str, required=False,
                               help='Path to RL configuration file')
    compare_parser.add_argument('--model', type=str, required=False,
                               help='Override model path for proactive autoscaler')
    compare_parser.add_argument('--output', '-o', type=str, required=False,
                               help='Override output directory for comparison results')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_simulation(args)
    elif args.command == 'run-rl':
        run_rl_training(args)
    elif args.command == 'compare':
        run_comparison(args)
    else:
        parser.print_help()


def run_rl_training(args):
    """Run RL training."""
    print("=" * 70)
    print("CLOUD AUTOSCALE - RL Training")
    print("=" * 70)
    print()
    
    # Load configuration
    print(f"📋 Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
        
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to results/run_rl_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"run_rl_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()
    
    try:
        train_rl(config, output_dir)
        
        # Save config
        import yaml
        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"💾 Configuration saved: {config_path}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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

        elif autoscaler_type == "rl":
            from cloud_autoscale.rl.autoscaler_rl import RLAutoscaler
            autoscaler = RLAutoscaler(
                autoscaler_config=config['autoscaler'],
                step_minutes=config['simulation']['step_minutes'],
                min_machines=config['simulation']['min_machines'],
                max_machines=config['simulation']['max_machines']
            )
            print(f"   RL Model: {config['autoscaler'].get('model_rl_dir', 'latest')}")

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
    """Run comparison between baseline, proactive, and RL autoscalers."""
    print("=" * 70)
    print("CLOUD AUTOSCALE - Multi-Strategy Comparison")
    print("=" * 70)
    print()
    
    # Create comparison output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        comparison_dir = Path(args.output)
    else:
        comparison_dir = Path("results") / f"comparison_{timestamp}"
    
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Comparison directory: {comparison_dir}")
    print()
    
    results_map = {}
    
    # Function to run a single simulation
    def run_single_sim(name, config_path, subdir):
        print(f"🔵 Running {name.upper()} simulation...")
        print("-" * 70)
        
        try:
            cfg = load_config(config_path)
            cfg['output']['directory'] = str(subdir)
            
            # Special handling for proactive model override
            if name == 'proactive' and args.model:
                 cfg['autoscaler']['model_run_dir'] = args.model
                 print(f"   Using model: {args.model}")

            # Load data
            demand_df = load_demand_data(cfg)
            print(f"   ✓ Loaded {len(demand_df)} time steps")
            
            # Init simulator
            sim = CloudSimulator(cfg['simulation'])
            
            # Init autoscaler
            atype = cfg['autoscaler'].get('type', 'baseline')
            if atype == 'baseline':
                autoscaler = BaselineAutoscaler(
                    autoscaler_config=cfg['autoscaler'],
                    step_minutes=cfg['simulation']['step_minutes']
                )
            elif atype == 'proactive':
                 # Replicate proactive init logic roughly or assume config is correct
                 # We need forecasting model
                 model_run_dir = cfg['autoscaler'].get('model_run_dir', 'latest')
                 # ... logic to find model ...
                 if model_run_dir == 'latest':
                    # Find latest run
                    results_base = Path("results") # Should look in results/
                    run_dirs = sorted(results_base.glob('run_*'))
                    model_run_dir = None
                    for rd in reversed(run_dirs):
                        if (rd / 'modeling').exists():
                            model_run_dir = rd
                            break
                    if not model_run_dir:
                        raise FileNotFoundError("No trained model found.")
                 else:
                    model_run_dir = Path(model_run_dir)
                    
                 forecast_model = ForecastingModel(model_run_dir)
                 autoscaler = ProactiveAutoscaler(
                    forecast_model=forecast_model,
                    autoscaler_config=cfg['autoscaler'],
                    step_minutes=cfg['simulation']['step_minutes'],
                    min_machines=cfg['simulation']['min_machines'],
                    max_machines=cfg['simulation']['max_machines']
                 )
            elif atype == 'rl':
                from cloud_autoscale.rl.autoscaler_rl import RLAutoscaler
                autoscaler = RLAutoscaler(
                    autoscaler_config=cfg['autoscaler'],
                    step_minutes=cfg['simulation']['step_minutes'],
                    min_machines=cfg['simulation']['min_machines'],
                    max_machines=cfg['simulation']['max_machines']
                )
            else:
                 raise ValueError(f"Unknown type {atype}")

            # Run
            res = sim.run(demand_df, autoscaler, output_dir=str(subdir), save_results=True)
            print(f"   ✓ {name} completed")
            return res

        except Exception as e:
            print(f"❌ Error running {name} simulation: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 1. Run Baseline
    baseline_dir = comparison_dir / "baseline"
    baseline_res = run_single_sim("baseline", args.baseline, baseline_dir)
    if baseline_res:
        results_map['baseline'] = baseline_res

    # 2. Run Proactive (if provided)
    if args.proactive:
        proactive_dir = comparison_dir / "proactive"
        proactive_res = run_single_sim("proactive", args.proactive, proactive_dir)
        if proactive_res:
             results_map['proactive'] = proactive_res

    # 3. Run RL (if provided)
    if args.rl:
        rl_dir = comparison_dir / "rl"
        rl_res = run_single_sim("rl", args.rl, rl_dir)
        if rl_res:
             results_map['rl'] = rl_res

    # Only proceed if we have results
    if not results_map:
        print("No simulations completed successfully.")
        return

    # ========== Merge Timelines ==========
    print("📊 Merging timelines and computing comparison metrics...")
    
    try:
        # Start with baseline timeline
        if 'baseline' not in results_map:
             print("Error: Baseline required for comparison.")
             return
             
        merged_df = results_map['baseline']['timeline'].copy()
        suffix_map = {'baseline': '_baseline'}
        
        # Rename baseline columns
        cols_to_rename = {c: f"baseline_{c}" for c in ['demand', 'capacity', 'machines', 'utilization', 'violation']}
        merged_df = merged_df.rename(columns=cols_to_rename)
        
        # Merge others
        for name, res in results_map.items():
            if name == 'baseline': continue
            
            timeline = res['timeline']
            cols_to_rename = {c: f"{name}_{c}" for c in ['capacity', 'machines', 'utilization', 'violation', 'demand']}
            temp_df = timeline.rename(columns=cols_to_rename)
            
            # Merge
            merged_df = pd.merge(
                merged_df,
                temp_df[['step'] + list(cols_to_rename.values())],
                on='step',
                how='inner',
                suffixes=('', '')
            )
        
        # Save merged
        merged_csv_path = comparison_dir / "comparison_timeline.csv"
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"   ✓ Merged timeline saved: {merged_csv_path}")
        
    except Exception as e:
        print(f"❌ Error merging timelines: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== Comparison Metrics & Plots ==========
    from cloud_autoscale.visualization.plots import (
        plot_baseline_vs_rl, 
        plot_rl_vs_proactive, 
        plot_rl_summary,
        plot_three_way_comparison,
        plot_three_way_metrics_table
    )
    
    try:
        plots_dir = comparison_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # If all three strategies are present, create comprehensive 3-way comparison
        if 'baseline' in results_map and 'proactive' in results_map and 'rl' in results_map:
            print("📊 Creating 3-way comparison plots...")
            plot_three_way_comparison(
                results_map['baseline'],
                results_map['proactive'],
                results_map['rl'],
                plots_dir / "three_way_comparison.png"
            )
            plot_three_way_metrics_table(
                results_map['baseline'],
                results_map['proactive'],
                results_map['rl'],
                plots_dir / "three_way_metrics.png"
            )
        
        # Individual pairwise comparisons
        if 'rl' in results_map:
            # Baseline vs RL
            plot_baseline_vs_rl(
                results_map['baseline'],
                results_map['rl'],
                plots_dir / "baseline_vs_rl.png"
            )
            
            plot_rl_summary(
                results_map['rl'],
                plots_dir / "rl_summary.png"
            )
            
            if 'proactive' in results_map:
                plot_rl_vs_proactive(
                    results_map['rl'],
                    results_map['proactive'],
                    plots_dir / "rl_vs_proactive.png"
                )

    except ImportError as e:
        print(f"⚠️  Visualization functions not ready: {e}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create comparison plots: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)
    print("✅ COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 70)
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
