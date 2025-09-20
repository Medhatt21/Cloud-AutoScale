"""Simulation analysis and metrics calculation."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from ..core.simulator import SimulationMetrics
from ..core.workload import Workload


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation."""
    
    # SLA metrics
    sla_violation_rate: float
    avg_queue_time: float
    avg_execution_time: float
    avg_total_time: float
    p95_queue_time: float
    p99_queue_time: float
    
    # Resource utilization metrics
    avg_cpu_utilization: float
    avg_memory_utilization: float
    peak_cpu_utilization: float
    peak_memory_utilization: float
    resource_efficiency: float
    
    # Scaling metrics
    total_scaling_events: int
    scale_up_events: int
    scale_down_events: int
    scaling_efficiency: float
    
    # Cost and energy metrics
    cost_proxy: float
    energy_proxy: float
    resource_waste: float
    
    # Reliability metrics
    workload_success_rate: float
    host_availability: float
    mean_time_to_failure: float
    mean_time_to_recovery: float


class MetricsCalculator:
    """Calculator for various performance metrics."""
    
    def __init__(self):
        self.logger = logger.bind(component="MetricsCalculator")
    
    def calculate_sla_metrics(self, completed_workloads: List[Workload]) -> Dict[str, float]:
        """Calculate SLA-related metrics."""
        if not completed_workloads:
            return {
                'sla_violation_rate': 0.0,
                'avg_queue_time': 0.0,
                'avg_execution_time': 0.0,
                'avg_total_time': 0.0,
                'p95_queue_time': 0.0,
                'p99_queue_time': 0.0,
            }
        
        # Extract timing data
        queue_times = [wl.queue_time for wl in completed_workloads]
        execution_times = [wl.execution_time for wl in completed_workloads]
        total_times = [wl.get_total_time() for wl in completed_workloads]
        
        # SLA violations
        sla_violations = sum(1 for wl in completed_workloads if wl.is_sla_violated())
        sla_violation_rate = sla_violations / len(completed_workloads)
        
        # Percentiles
        p95_queue = np.percentile(queue_times, 95) if queue_times else 0.0
        p99_queue = np.percentile(queue_times, 99) if queue_times else 0.0
        
        return {
            'sla_violation_rate': sla_violation_rate,
            'avg_queue_time': np.mean(queue_times) if queue_times else 0.0,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0.0,
            'avg_total_time': np.mean(total_times) if total_times else 0.0,
            'p95_queue_time': p95_queue,
            'p99_queue_time': p99_queue,
        }
    
    def calculate_resource_metrics(self, metrics_history: List[SimulationMetrics]) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        if not metrics_history:
            return {
                'avg_cpu_utilization': 0.0,
                'avg_memory_utilization': 0.0,
                'peak_cpu_utilization': 0.0,
                'peak_memory_utilization': 0.0,
                'resource_efficiency': 0.0,
            }
        
        cpu_utils = [m.avg_cpu_utilization for m in metrics_history]
        mem_utils = [m.avg_memory_utilization for m in metrics_history]
        
        # Filter out zero values for efficiency calculation
        non_zero_cpu = [u for u in cpu_utils if u > 0]
        non_zero_mem = [u for u in mem_utils if u > 0]
        
        # Resource efficiency (how well resources are utilized when allocated)
        avg_efficiency = 0.0
        if non_zero_cpu and non_zero_mem:
            avg_efficiency = (np.mean(non_zero_cpu) + np.mean(non_zero_mem)) / 2.0
        
        return {
            'avg_cpu_utilization': np.mean(cpu_utils) if cpu_utils else 0.0,
            'avg_memory_utilization': np.mean(mem_utils) if mem_utils else 0.0,
            'peak_cpu_utilization': np.max(cpu_utils) if cpu_utils else 0.0,
            'peak_memory_utilization': np.max(mem_utils) if mem_utils else 0.0,
            'resource_efficiency': avg_efficiency,
        }
    
    def calculate_scaling_metrics(self, metrics_history: List[SimulationMetrics]) -> Dict[str, float]:
        """Calculate scaling-related metrics."""
        if not metrics_history:
            return {
                'total_scaling_events': 0,
                'scale_up_events': 0,
                'scale_down_events': 0,
                'scaling_efficiency': 0.0,
            }
        
        total_scale_up = sum(m.scale_up_events for m in metrics_history)
        total_scale_down = sum(m.scale_down_events for m in metrics_history)
        total_scaling = total_scale_up + total_scale_down
        
        # Scaling efficiency: ratio of useful scaling actions
        # (simplified: assume all scaling actions are useful for now)
        scaling_efficiency = 1.0 if total_scaling > 0 else 0.0
        
        return {
            'total_scaling_events': total_scaling,
            'scale_up_events': total_scale_up,
            'scale_down_events': total_scale_down,
            'scaling_efficiency': scaling_efficiency,
        }
    
    def calculate_cost_metrics(
        self, 
        metrics_history: List[SimulationMetrics],
        host_cost_per_hour: float = 0.1,
        energy_cost_per_kwh: float = 0.12
    ) -> Dict[str, float]:
        """Calculate cost and energy proxy metrics."""
        if not metrics_history:
            return {
                'cost_proxy': 0.0,
                'energy_proxy': 0.0,
                'resource_waste': 0.0,
            }
        
        total_cost = 0.0
        total_energy = 0.0
        total_waste = 0.0
        
        for i in range(1, len(metrics_history)):
            prev_metrics = metrics_history[i-1]
            curr_metrics = metrics_history[i]
            time_delta = (curr_metrics.timestamp - prev_metrics.timestamp) / 3600.0  # hours
            
            # Cost calculation
            active_hosts = curr_metrics.active_hosts
            cost = active_hosts * host_cost_per_hour * time_delta
            total_cost += cost
            
            # Energy calculation (simplified: proportional to host count and utilization)
            avg_utilization = (curr_metrics.avg_cpu_utilization + curr_metrics.avg_memory_utilization) / 2.0
            energy_kwh = active_hosts * (0.2 + 0.3 * avg_utilization) * time_delta  # Base + utilization
            total_energy += energy_kwh * energy_cost_per_kwh
            
            # Waste calculation (underutilized resources)
            waste = active_hosts * (1.0 - avg_utilization) * time_delta
            total_waste += waste
        
        return {
            'cost_proxy': total_cost,
            'energy_proxy': total_energy,
            'resource_waste': total_waste,
        }
    
    def calculate_reliability_metrics(
        self,
        metrics_history: List[SimulationMetrics],
        completed_workloads: List[Workload],
        failed_workloads: Dict[str, Workload]
    ) -> Dict[str, float]:
        """Calculate reliability metrics."""
        total_workloads = len(completed_workloads) + len(failed_workloads)
        success_rate = len(completed_workloads) / total_workloads if total_workloads > 0 else 1.0
        
        # Host availability
        if metrics_history:
            total_host_time = sum(m.total_hosts for m in metrics_history)
            active_host_time = sum(m.active_hosts for m in metrics_history)
            host_availability = active_host_time / total_host_time if total_host_time > 0 else 1.0
        else:
            host_availability = 1.0
        
        return {
            'workload_success_rate': success_rate,
            'host_availability': host_availability,
            'mean_time_to_failure': 3600.0,  # Placeholder
            'mean_time_to_recovery': 300.0,   # Placeholder
        }


class SimulationAnalyzer:
    """Analyzer for simulation results."""
    
    def __init__(self):
        self.calculator = MetricsCalculator()
        self.logger = logger.bind(component="SimulationAnalyzer")
    
    def analyze_simulation(
        self,
        metrics_history: List[SimulationMetrics],
        completed_workloads: List[Workload],
        failed_workloads: Optional[Dict[str, Workload]] = None
    ) -> Dict[str, Any]:
        """Comprehensive analysis of simulation results."""
        
        if failed_workloads is None:
            failed_workloads = {}
        
        self.logger.info(f"Analyzing simulation with {len(metrics_history)} metric points, "
                        f"{len(completed_workloads)} completed workloads")
        
        # Calculate different metric categories
        sla_metrics = self.calculator.calculate_sla_metrics(completed_workloads)
        resource_metrics = self.calculator.calculate_resource_metrics(metrics_history)
        scaling_metrics = self.calculator.calculate_scaling_metrics(metrics_history)
        cost_metrics = self.calculator.calculate_cost_metrics(metrics_history)
        reliability_metrics = self.calculator.calculate_reliability_metrics(
            metrics_history, completed_workloads, failed_workloads
        )
        
        # Combine all metrics
        analysis = {
            'summary': {
                'total_workloads': len(completed_workloads) + len(failed_workloads),
                'completed_workloads': len(completed_workloads),
                'failed_workloads': len(failed_workloads),
                'simulation_duration': metrics_history[-1].timestamp if metrics_history else 0.0,
            },
            'sla_metrics': sla_metrics,
            'resource_metrics': resource_metrics,
            'scaling_metrics': scaling_metrics,
            'cost_metrics': cost_metrics,
            'reliability_metrics': reliability_metrics,
            'performance_score': self._calculate_performance_score(
                sla_metrics, resource_metrics, scaling_metrics, reliability_metrics
            ),
            'metrics_history': [self._metrics_to_dict(m) for m in metrics_history],
        }
        
        self.logger.info(f"Analysis completed. Performance score: {analysis['performance_score']:.3f}")
        return analysis
    
    def _calculate_performance_score(
        self,
        sla_metrics: Dict[str, float],
        resource_metrics: Dict[str, float],
        scaling_metrics: Dict[str, float],
        reliability_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall performance score (0-1, higher is better)."""
        
        # SLA score (lower violation rate is better)
        sla_score = 1.0 - sla_metrics['sla_violation_rate']
        
        # Resource efficiency score
        resource_score = resource_metrics['resource_efficiency']
        
        # Reliability score
        reliability_score = reliability_metrics['workload_success_rate']
        
        # Scaling efficiency score
        scaling_score = scaling_metrics['scaling_efficiency']
        
        # Weighted average
        weights = {'sla': 0.4, 'resource': 0.2, 'reliability': 0.3, 'scaling': 0.1}
        
        performance_score = (
            weights['sla'] * sla_score +
            weights['resource'] * resource_score +
            weights['reliability'] * reliability_score +
            weights['scaling'] * scaling_score
        )
        
        return max(0.0, min(1.0, performance_score))
    
    def _metrics_to_dict(self, metrics: SimulationMetrics) -> Dict[str, Any]:
        """Convert SimulationMetrics to dictionary."""
        return {
            'timestamp': metrics.timestamp,
            'total_workloads': metrics.total_workloads,
            'completed_workloads': metrics.completed_workloads,
            'failed_workloads': metrics.failed_workloads,
            'queued_workloads': metrics.queued_workloads,
            'running_workloads': metrics.running_workloads,
            'sla_violations': metrics.sla_violations,
            'avg_queue_time': metrics.avg_queue_time,
            'avg_execution_time': metrics.avg_execution_time,
            'avg_total_time': metrics.avg_total_time,
            'total_hosts': metrics.total_hosts,
            'active_hosts': metrics.active_hosts,
            'failed_hosts': metrics.failed_hosts,
            'avg_cpu_utilization': metrics.avg_cpu_utilization,
            'avg_memory_utilization': metrics.avg_memory_utilization,
            'scale_up_events': metrics.scale_up_events,
            'scale_down_events': metrics.scale_down_events,
            'vm_starts': metrics.vm_starts,
            'vm_stops': metrics.vm_stops,
        }
    
    def compare_methods(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Compare results from different methods."""
        
        comparison = {}
        
        for method_name, method_results in results.items():
            if not method_results:
                continue
            
            # Calculate statistics across runs
            performance_scores = [r['performance_score'] for r in method_results]
            sla_violation_rates = [r['sla_metrics']['sla_violation_rate'] for r in method_results]
            resource_efficiencies = [r['resource_metrics']['resource_efficiency'] for r in method_results]
            
            comparison[method_name] = {
                'performance_score': {
                    'mean': np.mean(performance_scores),
                    'std': np.std(performance_scores),
                    'min': np.min(performance_scores),
                    'max': np.max(performance_scores),
                },
                'sla_violation_rate': {
                    'mean': np.mean(sla_violation_rates),
                    'std': np.std(sla_violation_rates),
                    'min': np.min(sla_violation_rates),
                    'max': np.max(sla_violation_rates),
                },
                'resource_efficiency': {
                    'mean': np.mean(resource_efficiencies),
                    'std': np.std(resource_efficiencies),
                    'min': np.min(resource_efficiencies),
                    'max': np.max(resource_efficiencies),
                },
                'runs': len(method_results),
            }
        
        # Rank methods by performance score
        method_scores = [(name, stats['performance_score']['mean']) 
                        for name, stats in comparison.items()]
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        comparison['ranking'] = [{'method': name, 'score': score} 
                               for name, score in method_scores]
        
        self.logger.info(f"Method comparison completed. Best method: {method_scores[0][0]} "
                        f"(score: {method_scores[0][1]:.3f})")
        
        return comparison
