"""Visualization module for simulation results."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def create_all_plots(results: Dict[str, Any], metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Create all visualization plots.
    
    Args:
        results: Simulation results
        metrics: Calculated metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual plots
    plot_demand_vs_capacity(results, output_dir / "demand_vs_capacity.png")
    plot_utilization(results, output_dir / "utilization.png")
    plot_machines(results, output_dir / "machines.png")
    plot_violations(results, output_dir / "violations.png")
    plot_metrics_summary(metrics, output_dir / "metrics_summary.png")
    
    print(f"✓ All plots saved to {output_dir}")


def plot_demand_vs_capacity(results: Dict[str, Any], output_path: Path) -> None:
    """Plot demand vs capacity over time."""
    timeline = results['timeline']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot demand and capacity
    ax.plot(timeline['time'], timeline['demand'], label='Demand', color='#E74C3C', linewidth=2)
    ax.plot(timeline['time'], timeline['capacity'], label='Capacity', color='#3498DB', linewidth=2)
    
    # Shade violations
    violations = timeline['violation'] > 0
    if violations.any():
        ax.fill_between(timeline['time'], 0, timeline['demand'].max() * 1.1, 
                        where=violations, alpha=0.2, color='red', label='SLA Violation')
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Units', fontsize=12)
    ax.set_title('Demand vs Capacity Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_utilization(results: Dict[str, Any], output_path: Path) -> None:
    """Plot utilization over time with threshold lines."""
    timeline = results['timeline']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot utilization
    ax.plot(timeline['time'], timeline['utilization'], label='Utilization', 
            color='#9B59B6', linewidth=2)
    
    # Add threshold lines (typical values)
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1.5, 
              label='Upper Threshold (0.7)', alpha=0.7)
    ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1.5, 
              label='Lower Threshold (0.3)', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, 
              label='Max Capacity (1.0)', alpha=0.7)
    
    # Shade over-utilization
    over_util = timeline['utilization'] > 1.0
    if over_util.any():
        ax.fill_between(timeline['time'], 1.0, timeline['utilization'], 
                        where=over_util, alpha=0.3, color='red')
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Utilization', fontsize=12)
    ax.set_title('Resource Utilization Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.2, timeline['utilization'].max() * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_machines(results: Dict[str, Any], output_path: Path) -> None:
    """Plot machine count over time with scaling events."""
    timeline = results['timeline']
    events = results['events']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot machine count
    ax.plot(timeline['time'], timeline['machines'], label='Active Machines', 
            color='#2ECC71', linewidth=2, drawstyle='steps-post')
    
    # Mark scaling events
    if events:
        scale_up_times = [e['time'] for e in events if e['action'] == 'scale_up']
        scale_down_times = [e['time'] for e in events if e['action'] == 'scale_down']
        
        if scale_up_times:
            scale_up_machines = [timeline[timeline['time'] == t]['machines'].values[0] 
                               for t in scale_up_times if len(timeline[timeline['time'] == t]) > 0]
            ax.scatter(scale_up_times, scale_up_machines, color='green', marker='^', 
                      s=100, label='Scale Up', zorder=5)
        
        if scale_down_times:
            scale_down_machines = [timeline[timeline['time'] == t]['machines'].values[0] 
                                  for t in scale_down_times if len(timeline[timeline['time'] == t]) > 0]
            ax.scatter(scale_down_times, scale_down_machines, color='red', marker='v', 
                      s=100, label='Scale Down', zorder=5)
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Number of Machines', fontsize=12)
    ax.set_title('Machine Count and Scaling Events', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_violations(results: Dict[str, Any], output_path: Path) -> None:
    """Plot SLA violations over time."""
    timeline = results['timeline']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create cumulative violations
    cumulative_violations = timeline['violation'].cumsum()
    
    # Plot cumulative violations
    ax.plot(timeline['time'], cumulative_violations, label='Cumulative Violations', 
            color='#E74C3C', linewidth=2)
    
    # Mark individual violations
    violation_times = timeline[timeline['violation'] > 0]['time']
    violation_counts = cumulative_violations[timeline['violation'] > 0]
    
    if len(violation_times) > 0:
        ax.scatter(violation_times, violation_counts, color='red', marker='x', 
                  s=100, label='Violation Event', zorder=5)
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Cumulative Violations', fontsize=12)
    ax.set_title('SLA Violations Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_summary(metrics: Dict[str, Any], output_path: Path) -> None:
    """Create a summary bar chart of key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Violation Rate
    ax = axes[0, 0]
    violation_pct = metrics['sla']['violation_percentage']
    colors = ['#E74C3C' if violation_pct > 5 else '#2ECC71']
    ax.bar(['Violation Rate'], [violation_pct], color=colors, alpha=0.7)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('SLA Violation Rate', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(10, violation_pct * 1.2))
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Average Utilization
    ax = axes[0, 1]
    avg_util = metrics['utilization']['average'] * 100
    colors = ['#F39C12' if avg_util < 50 else '#2ECC71']
    ax.bar(['Avg Utilization'], [avg_util], color=colors, alpha=0.7)
    ax.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Average Resource Utilization', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Scaling Events
    ax = axes[1, 0]
    scale_up = metrics['scaling']['scale_up_events']
    scale_down = metrics['scaling']['scale_down_events']
    x = np.arange(2)
    ax.bar(x, [scale_up, scale_down], color=['#2ECC71', '#E74C3C'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Scale Up', 'Scale Down'])
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Scaling Events', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Cost and Stability
    ax = axes[1, 1]
    cost = metrics['cost']['total_cost']
    stability = metrics['scaling']['stability_score']
    
    # Normalize for visualization
    max_val = max(cost, stability)
    if max_val > 0:
        cost_norm = (cost / max_val) * 100
        stability_norm = (stability / max_val) * 100
    else:
        cost_norm = stability_norm = 0
    
    x = np.arange(2)
    ax.bar(x, [cost_norm, stability_norm], color=['#3498DB', '#9B59B6'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Cost\n(normalized)', 'Stability\n(normalized)'])
    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('Cost & Stability Metrics', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add actual values as text
    ax.text(0, cost_norm + 5, f'${cost:.2f}', ha='center', fontsize=9)
    ax.text(1, stability_norm + 5, f'{stability} events', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

