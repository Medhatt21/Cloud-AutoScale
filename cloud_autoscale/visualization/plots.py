"""Production-grade visualization module with storytelling mode."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')


def create_all_plots(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path,
    autoscaler_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create all visualization plots in storytelling mode.
    
    Args:
        results: Simulation results with timeline, events, metrics
        metrics: Calculated metrics dictionary
        output_dir: Directory to save plots
        autoscaler_config: Optional autoscaler configuration for threshold lines
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timeline = results['timeline']
    events = results.get('events', [])
    
    # Create individual plots
    plot_demand_vs_capacity(timeline, events, output_dir / "demand_vs_capacity.png")
    plot_utilization(timeline, autoscaler_config, output_dir / "utilization.png")
    plot_machines(timeline, events, output_dir / "machines.png")
    plot_violations(timeline, metrics, output_dir / "violations.png")
    plot_cost(timeline, metrics, output_dir / "cost.png")
    plot_scale_events_overlay(timeline, events, output_dir / "events_overlay.png")
    plot_metrics_summary(metrics, output_dir / "metrics_summary.png")
    
    print(f"✓ All plots saved to {output_dir}")


def plot_demand_vs_capacity(
    timeline: pd.DataFrame,
    events: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Plot demand vs capacity with SLA violation shading.
    
    Args:
        timeline: Timeline DataFrame
        events: List of scaling events
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot demand and capacity
    ax.plot(timeline['time'], timeline['demand'], 
            label='Demand', color='#E74C3C', linewidth=2, alpha=0.8)
    ax.plot(timeline['time'], timeline['capacity'], 
            label='Capacity', color='#3498DB', linewidth=2.5, alpha=0.9)
    
    # Shade SLA violation regions
    violations = timeline['violation'] > 0
    if violations.any():
        y_max = max(timeline['demand'].max(), timeline['capacity'].max()) * 1.1
        ax.fill_between(timeline['time'], 0, y_max,
                        where=violations, alpha=0.15, color='red', 
                        label='SLA Violation Zone')
    
    ax.set_xlabel('Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Resource Units', fontsize=13, fontweight='bold')
    ax.set_title('Demand vs Capacity Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_utilization(
    timeline: pd.DataFrame,
    autoscaler_config: Optional[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Plot utilization with autoscaler threshold lines.
    
    Args:
        timeline: Timeline DataFrame
        autoscaler_config: Autoscaler configuration (for thresholds)
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot utilization
    ax.plot(timeline['time'], timeline['utilization'], 
            label='Utilization', color='#9B59B6', linewidth=2, alpha=0.8)
    
    # Add threshold lines if config provided
    if autoscaler_config:
        upper = autoscaler_config.get('upper_threshold', 0.8)
        lower = autoscaler_config.get('lower_threshold', 0.4)
        
        ax.axhline(y=upper, color='#E74C3C', linestyle='--', linewidth=2, 
                   label=f'Upper Threshold ({upper:.0%})', alpha=0.7)
        ax.axhline(y=lower, color='#2ECC71', linestyle='--', linewidth=2, 
                   label=f'Lower Threshold ({lower:.0%})', alpha=0.7)
    
    # Add 100% line
    ax.axhline(y=1.0, color='orange', linestyle=':', linewidth=2, 
               label='100% Utilization', alpha=0.6)
    
    ax.set_xlabel('Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Utilization', fontsize=13, fontweight='bold')
    ax.set_title('Resource Utilization Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_machines(
    timeline: pd.DataFrame,
    events: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Plot machine count over time with scaling events.
    
    Args:
        timeline: Timeline DataFrame
        events: List of scaling events
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot machine count
    ax.plot(timeline['time'], timeline['machines'], 
            label='Active Machines', color='#16A085', linewidth=2.5, alpha=0.9)
    ax.fill_between(timeline['time'], 0, timeline['machines'], 
                     alpha=0.2, color='#16A085')
    
    # Mark scaling events
    if events:
        scale_up_times = [e['time'] for e in events if e['action'] == 'scale_up']
        scale_down_times = [e['time'] for e in events if e['action'] == 'scale_down']
        
        if scale_up_times:
            scale_up_machines = [timeline[timeline['time'] == t]['machines'].iloc[0] 
                                for t in scale_up_times if t in timeline['time'].values]
            ax.scatter(scale_up_times[:len(scale_up_machines)], scale_up_machines, 
                      color='green', s=100, marker='^', label='Scale Up', 
                      zorder=5, edgecolors='black', linewidths=1.5)
        
        if scale_down_times:
            scale_down_machines = [timeline[timeline['time'] == t]['machines'].iloc[0] 
                                  for t in scale_down_times if t in timeline['time'].values]
            ax.scatter(scale_down_times[:len(scale_down_machines)], scale_down_machines, 
                      color='red', s=100, marker='v', label='Scale Down', 
                      zorder=5, edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Machines', fontsize=13, fontweight='bold')
    ax.set_title('Machine Count and Scaling Events', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_violations(
    timeline: pd.DataFrame,
    metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Plot SLA violations histogram and rate.
    
    Args:
        timeline: Timeline DataFrame
        metrics: Metrics dictionary
        output_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram of violations over time
    violation_counts = timeline.groupby(timeline.index // 10)['violation'].sum()
    ax1.bar(range(len(violation_counts)), violation_counts, 
            color='#E74C3C', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Time Window', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Violation Count', fontsize=12, fontweight='bold')
    ax1.set_title('SLA Violations Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Violation rate bar
    total_violations = metrics.get('total_violations', 0)
    violation_rate = metrics.get('violation_rate', 0)
    total_steps = len(timeline)
    
    categories = ['Compliant', 'Violated']
    values = [total_steps - total_violations, total_violations]
    colors = ['#2ECC71', '#E74C3C']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Steps', fontsize=12, fontweight='bold')
    ax2.set_title(f'SLA Compliance (Violation Rate: {violation_rate:.1%})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cost(
    timeline: pd.DataFrame,
    metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Plot cumulative cost over time.
    
    Args:
        timeline: Timeline DataFrame
        metrics: Metrics dictionary
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate cumulative cost (approximate from machines)
    # Assuming cost_per_machine_per_hour is available in metrics
    cost_per_step = metrics.get('cost_per_step', 0)
    cumulative_cost = np.cumsum([cost_per_step] * len(timeline))
    
    ax.plot(timeline['time'], cumulative_cost, 
            label='Cumulative Cost', color='#F39C12', linewidth=2.5, alpha=0.9)
    ax.fill_between(timeline['time'], 0, cumulative_cost, 
                     alpha=0.2, color='#F39C12')
    
    # Add total cost annotation
    total_cost = metrics.get('total_cost', 0)
    ax.text(timeline['time'].iloc[-1], cumulative_cost[-1], 
            f'  Total: ${total_cost:.2f}',
            fontsize=12, fontweight='bold', va='center')
    
    ax.set_xlabel('Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost ($)', fontsize=13, fontweight='bold')
    ax.set_title('Cumulative Cost Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scale_events_overlay(
    timeline: pd.DataFrame,
    events: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Plot demand vs capacity with annotated scaling events.
    
    Args:
        timeline: Timeline DataFrame
        events: List of scaling events
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot demand and capacity
    ax.plot(timeline['time'], timeline['demand'], 
            label='Demand', color='#E74C3C', linewidth=2, alpha=0.7)
    ax.plot(timeline['time'], timeline['capacity'], 
            label='Capacity', color='#3498DB', linewidth=2.5, alpha=0.8)
    
    # Annotate scaling events
    if events:
        for event in events:
            time = event['time']
            action = event['action']
            delta = event.get('delta', 0)
            
            # Find corresponding capacity
            matching_rows = timeline[timeline['time'] == time]
            if not matching_rows.empty:
                capacity = matching_rows['capacity'].iloc[0]
                
                color = 'green' if action == 'scale_up' else 'red'
                marker = '^' if action == 'scale_up' else 'v'
                
                ax.scatter([time], [capacity], color=color, s=150, 
                          marker=marker, zorder=5, edgecolors='black', linewidths=1.5)
                
                # Add annotation for significant events
                if abs(delta) > 5:  # Only annotate large changes
                    ax.annotate(f'{delta:+d}', 
                               xy=(time, capacity), 
                               xytext=(0, 15 if action == 'scale_up' else -15),
                               textcoords='offset points',
                               fontsize=9, fontweight='bold',
                               ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Resource Units', fontsize=13, fontweight='bold')
    ax.set_title('Scaling Events Overlay', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(
    metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Create a comprehensive metrics summary dashboard.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save plot
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle('Simulation Metrics Summary', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. SLA Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    sla_data = [
        metrics.get('total_violations', 0),
        metrics.get('violation_rate', 0) * 100
    ]
    ax1.bar(['Total\nViolations', 'Violation\nRate (%)'], sla_data, 
            color=['#E74C3C', '#E67E22'], alpha=0.7, edgecolor='black')
    ax1.set_title('SLA Metrics', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Utilization
    ax2 = fig.add_subplot(gs[0, 1])
    util = metrics.get('avg_utilization', 0) * 100
    ax2.barh(['Avg Utilization'], [util], color='#9B59B6', alpha=0.7, edgecolor='black')
    ax2.set_xlim(0, 150)
    ax2.axvline(x=100, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Utilization (%)', fontsize=10)
    ax2.set_title('Average Utilization', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Machine Range
    ax3 = fig.add_subplot(gs[0, 2])
    machine_data = [
        metrics.get('min_machines', 0),
        metrics.get('avg_machines', 0),
        metrics.get('max_machines', 0)
    ]
    ax3.bar(['Min', 'Avg', 'Max'], machine_data, 
            color=['#2ECC71', '#3498DB', '#E74C3C'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Machines', fontsize=10)
    ax3.set_title('Machine Count Range', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Scaling Events
    ax4 = fig.add_subplot(gs[1, 0])
    scale_data = [
        metrics.get('scale_up_events', 0),
        metrics.get('scale_down_events', 0)
    ]
    ax4.bar(['Scale Up', 'Scale Down'], scale_data, 
            color=['#2ECC71', '#E74C3C'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Scaling Events', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Cost Breakdown
    ax5 = fig.add_subplot(gs[1, 1:])
    cost_labels = ['Total Cost', 'Cost/Hour', 'Cost/Step']
    cost_values = [
        metrics.get('total_cost', 0),
        metrics.get('cost_per_hour', 0),
        metrics.get('cost_per_step', 0)
    ]
    bars = ax5.bar(cost_labels, cost_values, 
                   color=['#F39C12', '#E67E22', '#D35400'], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Cost ($)', fontsize=10)
    ax5.set_title('Cost Metrics', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Summary Text
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    • Total Simulation Steps: {metrics.get('total_violations', 0) + int((1 - metrics.get('violation_rate', 0)) * 100)}
    • SLA Compliance: {(1 - metrics.get('violation_rate', 0)) * 100:.1f}%
    • Average Utilization: {metrics.get('avg_utilization', 0) * 100:.1f}%
    • Total Scaling Actions: {metrics.get('total_scale_events', 0)}
    • Total Cost: ${metrics.get('total_cost', 0):.2f}
    • Stability Score: {metrics.get('total_scale_events', 0)} (lower is better)
    """
    
    ax6.text(0.5, 0.5, summary_text, 
            ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_forecast_comparison(
    run_dir: Path,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot actual demand vs ML predictions from modeling results.
    
    This visualization helps compare baseline (reactive) vs proactive (ML-driven)
    autoscaling performance by showing how well the ML model forecasts demand.
    
    Args:
        run_dir: Path to simulation run directory containing modeling/predictions.csv
        output_path: Optional path to save plot (defaults to run_dir/plots/forecast_comparison.png)
    """
    # Load predictions
    predictions_path = run_dir / "modeling" / "predictions.csv"
    
    if not predictions_path.exists():
        print(f"⚠️  Predictions file not found: {predictions_path}")
        print("   Run the modeling notebook first to generate predictions.")
        return
    
    predictions = pd.read_csv(predictions_path)
    
    # Set default output path
    if output_path is None:
        output_path = run_dir / "plots" / "forecast_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Full time series comparison
    window = min(1000, len(predictions))
    times = predictions['time'].values[:window]
    actual = predictions['actual'].values[:window]
    lgb_pred = predictions['lgb_pred'].values[:window]
    
    axes[0].plot(times, actual, label='Actual Demand', linewidth=2, color='black', alpha=0.8)
    axes[0].plot(times, lgb_pred, label='LightGBM Forecast', linewidth=2, 
                linestyle='--', color='#9B59B6', alpha=0.8)
    axes[0].fill_between(times, actual, lgb_pred, alpha=0.2, color='orange')
    axes[0].set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('CPU Demand', fontweight='bold', fontsize=12)
    axes[0].set_title('ML Forecast vs Actual Demand (Test Set)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Forecast errors over time
    errors = predictions['lgb_error'].values[:window]
    axes[1].plot(times, errors, linewidth=1.5, color='#E74C3C', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1].fill_between(times, 0, errors, where=(errors > 0), 
                        alpha=0.3, color='red', label='Over-prediction')
    axes[1].fill_between(times, 0, errors, where=(errors < 0), 
                        alpha=0.3, color='blue', label='Under-prediction')
    axes[1].set_xlabel('Time (minutes)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Forecast Error', fontweight='bold', fontsize=12)
    axes[1].set_title('Forecast Error Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Add metrics text
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    metrics_text = f'MAE: {mae:.4f}  |  RMSE: {rmse:.4f}'
    axes[1].text(0.5, 0.95, metrics_text, transform=axes[1].transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Forecast comparison plot saved: {output_path}")


def plot_baseline_vs_proactive_comparison(
    baseline_run_dir: Path,
    proactive_run_dir: Path,
    output_path: Optional[Path] = None
) -> None:
    """
    Compare baseline (reactive) vs proactive (ML-driven) autoscaling performance.
    
    This visualization shows side-by-side comparison of:
    - Machine count over time
    - Utilization patterns
    - SLA violations
    - Total cost
    
    Args:
        baseline_run_dir: Path to baseline autoscaler run directory
        proactive_run_dir: Path to proactive autoscaler run directory
        output_path: Optional path to save plot (defaults to proactive_run_dir/plots/comparison.png)
    """
    # Load timelines
    baseline_timeline = pd.read_csv(baseline_run_dir / "timeline.csv")
    proactive_timeline = pd.read_csv(proactive_run_dir / "timeline.csv")
    
    # Load metrics
    import json
    with open(baseline_run_dir / "metrics.json", 'r') as f:
        baseline_metrics = json.load(f)
    with open(proactive_run_dir / "metrics.json", 'r') as f:
        proactive_metrics = json.load(f)
    
    # Set default output path
    if output_path is None:
        output_path = proactive_run_dir / "plots" / "baseline_vs_proactive.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Machine count comparison
    axes[0, 0].plot(baseline_timeline['time'], baseline_timeline['machines'], 
                   label='Baseline (Reactive)', linewidth=2, color='#3498DB', alpha=0.8)
    axes[0, 0].plot(proactive_timeline['time'], proactive_timeline['machines'], 
                   label='Proactive (ML)', linewidth=2, color='#9B59B6', alpha=0.8)
    axes[0, 0].set_xlabel('Time (minutes)', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Machines', fontweight='bold')
    axes[0, 0].set_title('Machine Count Over Time', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Utilization comparison
    axes[0, 1].plot(baseline_timeline['time'], baseline_timeline['utilization'], 
                   label='Baseline', linewidth=1.5, color='#3498DB', alpha=0.7)
    axes[0, 1].plot(proactive_timeline['time'], proactive_timeline['utilization'], 
                   label='Proactive', linewidth=1.5, color='#9B59B6', alpha=0.7)
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='SLA Threshold')
    axes[0, 1].set_xlabel('Time (minutes)', fontweight='bold')
    axes[0, 1].set_ylabel('Utilization', fontweight='bold')
    axes[0, 1].set_title('Utilization Over Time', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Violations comparison (bar chart)
    categories = ['Total\nViolations', 'Violation\nRate (%)', 'Scale\nEvents']
    baseline_values = [
        baseline_metrics['total_violations'],
        baseline_metrics['violation_rate'] * 100,
        baseline_metrics['total_scale_events']
    ]
    proactive_values = [
        proactive_metrics['total_violations'],
        proactive_metrics['violation_rate'] * 100,
        proactive_metrics['total_scale_events']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, baseline_values, width, label='Baseline', 
                  color='#3498DB', alpha=0.8)
    axes[1, 0].bar(x + width/2, proactive_values, width, label='Proactive', 
                  color='#9B59B6', alpha=0.8)
    axes[1, 0].set_ylabel('Count / Rate', fontweight='bold')
    axes[1, 0].set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cost comparison
    baseline_cost = baseline_metrics['total_cost']
    proactive_cost = proactive_metrics['total_cost']
    cost_savings = ((baseline_cost - proactive_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
    
    axes[1, 1].bar(['Baseline', 'Proactive'], [baseline_cost, proactive_cost], 
                  color=['#3498DB', '#9B59B6'], alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Total Cost ($)', fontweight='bold')
    axes[1, 1].set_title('Total Cost Comparison', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add cost savings text
    savings_text = f'Cost Savings: {cost_savings:+.1f}%'
    savings_color = 'green' if cost_savings > 0 else 'red'
    axes[1, 1].text(0.5, 0.95, savings_text, transform=axes[1, 1].transAxes,
                   ha='center', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=savings_color, alpha=0.3))
    
    plt.suptitle('Baseline vs Proactive Autoscaling Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved: {output_path}")
