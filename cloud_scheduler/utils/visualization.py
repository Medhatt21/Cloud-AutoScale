"""Visualization utilities for simulation results."""

from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from ..core.simulator import SimulationMetrics


def create_plots(analysis: Dict[str, Any], output_dir: Path) -> None:
    """Create visualization plots for simulation analysis."""
    
    logger.info(f"Creating plots in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics history
    metrics_history = analysis.get('metrics_history', [])
    if not metrics_history:
        logger.warning("No metrics history found for plotting")
        return
    
    # Create timeline plots
    plot_metrics_timeline(metrics_history, output_dir / "metrics_timeline.html")
    
    # Create resource utilization plots
    plot_resource_utilization(metrics_history, output_dir / "resource_utilization.html")
    
    # Create SLA metrics plots
    plot_sla_metrics(analysis, output_dir / "sla_metrics.html")
    
    # Create scaling events plot
    plot_scaling_events(metrics_history, output_dir / "scaling_events.html")
    
    logger.info("Plots created successfully")


def plot_metrics_timeline(metrics_history: List[Dict[str, Any]], output_file: Path) -> None:
    """Plot timeline of key metrics."""
    
    if not metrics_history:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_history)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Workload Status', 'Resource Utilization',
            'SLA Violations', 'Host Status',
            'Queue Length', 'Scaling Events'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Workload status
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['running_workloads'], 
                  name='Running', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['queued_workloads'], 
                  name='Queued', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['completed_workloads'], 
                  name='Completed', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Resource utilization
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['avg_cpu_utilization'], 
                  name='CPU', line=dict(color='red')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['avg_memory_utilization'], 
                  name='Memory', line=dict(color='purple')),
        row=1, col=2
    )
    
    # SLA violations
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['sla_violations'], 
                  name='SLA Violations', line=dict(color='red')),
        row=2, col=1
    )
    
    # Host status
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['active_hosts'], 
                  name='Active Hosts', line=dict(color='green')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['failed_hosts'], 
                  name='Failed Hosts', line=dict(color='red')),
        row=2, col=2
    )
    
    # Queue length
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['queued_workloads'], 
                  name='Queue Length', line=dict(color='orange')),
        row=3, col=1
    )
    
    # Scaling events
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['scale_up_events'], 
                  name='Scale Up', line=dict(color='green')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['scale_down_events'], 
                  name='Scale Down', line=dict(color='red')),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Simulation Metrics Timeline",
        showlegend=True
    )
    
    # Update x-axes
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Time (seconds)", row=i, col=j)
    
    # Save plot
    fig.write_html(str(output_file))
    logger.info(f"Timeline plot saved to {output_file}")


def plot_resource_utilization(metrics_history: List[Dict[str, Any]], output_file: Path) -> None:
    """Plot resource utilization over time."""
    
    if not metrics_history:
        return
    
    df = pd.DataFrame(metrics_history)
    
    fig = go.Figure()
    
    # CPU utilization
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['avg_cpu_utilization'],
        mode='lines',
        name='CPU Utilization',
        line=dict(color='red', width=2)
    ))
    
    # Memory utilization
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['avg_memory_utilization'],
        mode='lines',
        name='Memory Utilization',
        line=dict(color='blue', width=2)
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="Scale-up threshold")
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                  annotation_text="Scale-down threshold")
    
    fig.update_layout(
        title="Resource Utilization Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Utilization",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    
    fig.write_html(str(output_file))
    logger.info(f"Resource utilization plot saved to {output_file}")


def plot_sla_metrics(analysis: Dict[str, Any], output_file: Path) -> None:
    """Plot SLA-related metrics."""
    
    sla_metrics = analysis.get('sla_metrics', {})
    if not sla_metrics:
        return
    
    # Create bar chart of SLA metrics
    metrics_names = list(sla_metrics.keys())
    metrics_values = list(sla_metrics.values())
    
    fig = go.Figure(data=[
        go.Bar(x=metrics_names, y=metrics_values, 
               marker_color=['red' if 'violation' in name else 'blue' for name in metrics_names])
    ])
    
    fig.update_layout(
        title="SLA Metrics Summary",
        xaxis_title="Metric",
        yaxis_title="Value",
        xaxis_tickangle=-45
    )
    
    fig.write_html(str(output_file))
    logger.info(f"SLA metrics plot saved to {output_file}")


def plot_scaling_events(metrics_history: List[Dict[str, Any]], output_file: Path) -> None:
    """Plot scaling events over time."""
    
    if not metrics_history:
        return
    
    df = pd.DataFrame(metrics_history)
    
    fig = go.Figure()
    
    # Cumulative scaling events
    df['cumulative_scale_up'] = df['scale_up_events'].cumsum()
    df['cumulative_scale_down'] = df['scale_down_events'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_scale_up'],
        mode='lines',
        name='Cumulative Scale Up',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_scale_down'],
        mode='lines',
        name='Cumulative Scale Down',
        line=dict(color='red', width=2)
    ))
    
    # Add individual scaling events as markers
    scale_up_times = df[df['scale_up_events'] > 0]['timestamp']
    scale_up_values = df[df['scale_up_events'] > 0]['cumulative_scale_up']
    
    fig.add_trace(go.Scatter(
        x=scale_up_times,
        y=scale_up_values,
        mode='markers',
        name='Scale Up Events',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))
    
    scale_down_times = df[df['scale_down_events'] > 0]['timestamp']
    scale_down_values = df[df['scale_down_events'] > 0]['cumulative_scale_down']
    
    fig.add_trace(go.Scatter(
        x=scale_down_times,
        y=scale_down_values,
        mode='markers',
        name='Scale Down Events',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))
    
    fig.update_layout(
        title="Scaling Events Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Cumulative Events",
        hovermode='x unified'
    )
    
    fig.write_html(str(output_file))
    logger.info(f"Scaling events plot saved to {output_file}")


def plot_comparison_results(results: Dict[str, List[Dict[str, Any]]], output_file: Path) -> None:
    """Plot comparison results between different methods."""
    
    # Prepare data for plotting
    methods = []
    performance_scores = []
    sla_violation_rates = []
    resource_efficiencies = []
    
    for method, method_results in results.items():
        for result in method_results:
            methods.append(method)
            performance_scores.append(result.get('performance_score', 0))
            sla_violation_rates.append(result['sla_metrics']['sla_violation_rate'])
            resource_efficiencies.append(result['resource_metrics']['resource_efficiency'])
    
    # Create comparison plots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Performance Score', 'SLA Violation Rate', 'Resource Efficiency']
    )
    
    # Performance score box plot
    fig.add_trace(
        go.Box(x=methods, y=performance_scores, name='Performance Score'),
        row=1, col=1
    )
    
    # SLA violation rate box plot
    fig.add_trace(
        go.Box(x=methods, y=sla_violation_rates, name='SLA Violation Rate'),
        row=1, col=2
    )
    
    # Resource efficiency box plot
    fig.add_trace(
        go.Box(x=methods, y=resource_efficiencies, name='Resource Efficiency'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Method Comparison",
        height=400,
        showlegend=False
    )
    
    fig.write_html(str(output_file))
    logger.info(f"Comparison plot saved to {output_file}")
