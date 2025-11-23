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


def compare_metrics(baseline_metrics: Dict[str, Any], proactive_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare baseline vs proactive autoscaler metrics.
    
    Args:
        baseline_metrics: Metrics from baseline autoscaler run
        proactive_metrics: Metrics from proactive autoscaler run
    
    Returns:
        Dictionary with comparison metrics
    """
    # Extract key metrics
    baseline_violations = baseline_metrics['total_violations']
    proactive_violations = proactive_metrics['total_violations']
    
    baseline_viol_rate = baseline_metrics['violation_rate']
    proactive_viol_rate = proactive_metrics['violation_rate']
    
    baseline_cost = baseline_metrics['total_cost']
    proactive_cost = proactive_metrics['total_cost']
    
    baseline_util = baseline_metrics['avg_utilization']
    proactive_util = proactive_metrics['avg_utilization']
    
    baseline_stability = baseline_metrics['total_scale_events']
    proactive_stability = proactive_metrics['total_scale_events']
    
    # Calculate improvements (positive = proactive is better)
    sla_improvement = (
        ((baseline_viol_rate - proactive_viol_rate) / baseline_viol_rate * 100)
        if baseline_viol_rate > 0 else 0
    )
    
    cost_change_percent = (
        ((proactive_cost - baseline_cost) / baseline_cost * 100)
        if baseline_cost > 0 else 0
    )
    
    violation_reduction_percent = (
        ((baseline_violations - proactive_violations) / baseline_violations * 100)
        if baseline_violations > 0 else 0
    )
    
    avg_utilization_gain = (proactive_util - baseline_util) * 100  # percentage points
    
    stability_change = proactive_stability - baseline_stability  # negative = more stable
    
    comparison = {
        'sla_improvement': float(sla_improvement),
        'cost_change_percent': float(cost_change_percent),
        'cost_savings_percent': float(-cost_change_percent),  # Positive = savings
        'violation_reduction_percent': float(violation_reduction_percent),
        'avg_utilization_gain': float(avg_utilization_gain),
        'stability_change': int(stability_change),
        'baseline': {
            'violations': int(baseline_violations),
            'violation_rate': float(baseline_viol_rate),
            'cost': float(baseline_cost),
            'avg_utilization': float(baseline_util),
            'scale_events': int(baseline_stability)
        },
        'proactive': {
            'violations': int(proactive_violations),
            'violation_rate': float(proactive_viol_rate),
            'cost': float(proactive_cost),
            'avg_utilization': float(proactive_util),
            'scale_events': int(proactive_stability)
        }
    }
    
    return comparison

