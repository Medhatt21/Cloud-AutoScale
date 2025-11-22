"""Metrics calculation for simulation results."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics from simulation results.
    
    Args:
        results: Simulation results dictionary
    
    Returns:
        Dictionary with calculated metrics
    """
    timeline = results['timeline']
    events = results['events']
    
    # SLA metrics
    total_violations = results['metrics']['total_violations']
    violation_rate = results['metrics']['violation_rate']
    
    # Utilization metrics
    utilization = timeline['utilization'].values
    avg_utilization = np.mean(utilization)
    p50_utilization = np.percentile(utilization, 50)
    p95_utilization = np.percentile(utilization, 95)
    p99_utilization = np.percentile(utilization, 99)
    
    # Capacity metrics
    machines = timeline['machines'].values
    avg_machines = np.mean(machines)
    min_machines = np.min(machines)
    max_machines = np.max(machines)
    
    # Scaling events
    scale_up_events = len([e for e in events if e['action'] == 'scale_up'])
    scale_down_events = len([e for e in events if e['action'] == 'scale_down'])
    total_scale_events = scale_up_events + scale_down_events
    
    # Cost
    total_cost = results['metrics']['total_cost']
    
    # Stability (lower is better)
    stability_score = total_scale_events
    
    # Overall efficiency score (0-100)
    # Penalize violations, reward high utilization, penalize excessive scaling
    violation_penalty = violation_rate * 100
    utilization_reward = avg_utilization * 100
    scaling_penalty = min(total_scale_events / len(timeline) * 100, 20)
    
    efficiency_score = max(0, utilization_reward - violation_penalty - scaling_penalty)
    
    metrics = {
        'sla': {
            'total_violations': int(total_violations),
            'violation_rate': float(violation_rate),
            'violation_percentage': float(violation_rate * 100)
        },
        'utilization': {
            'average': float(avg_utilization),
            'p50': float(p50_utilization),
            'p95': float(p95_utilization),
            'p99': float(p99_utilization)
        },
        'capacity': {
            'avg_machines': float(avg_machines),
            'min_machines': int(min_machines),
            'max_machines': int(max_machines)
        },
        'scaling': {
            'scale_up_events': int(scale_up_events),
            'scale_down_events': int(scale_down_events),
            'total_events': int(total_scale_events),
            'stability_score': int(stability_score)
        },
        'cost': {
            'total_cost': float(total_cost),
            'cost_per_step': float(total_cost / len(timeline)) if len(timeline) > 0 else 0
        },
        'overall': {
            'efficiency_score': float(efficiency_score)
        }
    }
    
    return metrics


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as a readable table.
    
    Args:
        metrics: Metrics dictionary
    
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION METRICS")
    lines.append("=" * 60)
    
    # SLA Metrics
    lines.append("\nSLA Metrics:")
    lines.append(f"  Total Violations:    {metrics['sla']['total_violations']}")
    lines.append(f"  Violation Rate:      {metrics['sla']['violation_percentage']:.2f}%")
    
    # Utilization Metrics
    lines.append("\nUtilization Metrics:")
    lines.append(f"  Average:             {metrics['utilization']['average']:.2%}")
    lines.append(f"  50th Percentile:     {metrics['utilization']['p50']:.2%}")
    lines.append(f"  95th Percentile:     {metrics['utilization']['p95']:.2%}")
    lines.append(f"  99th Percentile:     {metrics['utilization']['p99']:.2%}")
    
    # Capacity Metrics
    lines.append("\nCapacity Metrics:")
    lines.append(f"  Average Machines:    {metrics['capacity']['avg_machines']:.1f}")
    lines.append(f"  Min Machines:        {metrics['capacity']['min_machines']}")
    lines.append(f"  Max Machines:        {metrics['capacity']['max_machines']}")
    
    # Scaling Metrics
    lines.append("\nScaling Events:")
    lines.append(f"  Scale Up:            {metrics['scaling']['scale_up_events']}")
    lines.append(f"  Scale Down:          {metrics['scaling']['scale_down_events']}")
    lines.append(f"  Total Events:        {metrics['scaling']['total_events']}")
    lines.append(f"  Stability Score:     {metrics['scaling']['stability_score']} (lower is better)")
    
    # Cost Metrics
    lines.append("\nCost Metrics:")
    lines.append(f"  Total Cost:          ${metrics['cost']['total_cost']:.2f}")
    lines.append(f"  Cost per Step:       ${metrics['cost']['cost_per_step']:.4f}")
    
    # Overall
    lines.append("\nOverall Performance:")
    lines.append(f"  Efficiency Score:    {metrics['overall']['efficiency_score']:.2f}/100")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

