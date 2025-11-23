"""Production-grade cloud autoscaling simulator with strict validation."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class CloudSimulator:
    """
    Pure capacity-based discrete-time simulator for cloud autoscaling.
    
    Production requirement: All configuration must be explicit.
    No defaults, no silent failures.
    """
    
    def __init__(self, sim_config: Dict[str, Any]):
        """
        Initialize simulator with strict validation.
        
        Args:
            sim_config: Simulation configuration dictionary
                       Required keys: step_minutes, min_machines, max_machines,
                                    machine_capacity, cost_per_machine_per_hour
        
        Raises:
            ValueError: If required configuration keys are missing or invalid
        """
        # Validate presence of all required keys
        required_keys = [
            'step_minutes',
            'min_machines',
            'max_machines',
            'machine_capacity',
            'cost_per_machine_per_hour'
        ]
        
        missing_keys = [key for key in required_keys if key not in sim_config]
        if missing_keys:
            raise ValueError(
                f"Missing required simulation config keys: {missing_keys}"
            )
        
        # Validate types and values
        self.step_minutes = sim_config['step_minutes']
        if not isinstance(self.step_minutes, (int, float)) or self.step_minutes <= 0:
            raise ValueError("step_minutes must be a positive number")
        
        self.min_machines = sim_config['min_machines']
        if not isinstance(self.min_machines, int) or self.min_machines < 1:
            raise ValueError("min_machines must be an integer >= 1")
        
        self.max_machines = sim_config['max_machines']
        if not isinstance(self.max_machines, int) or self.max_machines < self.min_machines:
            raise ValueError("max_machines must be an integer >= min_machines")
        
        self.machine_capacity = sim_config['machine_capacity']
        if not isinstance(self.machine_capacity, (int, float)) or self.machine_capacity <= 0:
            raise ValueError("machine_capacity must be a positive number")
        
        self.cost_per_machine_per_hour = sim_config['cost_per_machine_per_hour']
        if not isinstance(self.cost_per_machine_per_hour, (int, float)) or self.cost_per_machine_per_hour < 0:
            raise ValueError("cost_per_machine_per_hour must be a non-negative number")
        
        # Initial state
        self.current_machines = self.min_machines
        self.current_capacity = self.current_machines * self.machine_capacity
        
        # Histories
        self.time_history = []
        self.step_history = []
        self.demand_history = []
        self.capacity_history = []
        self.utilization_history = []
        self.machine_history = []
        self.violation_history = []
        self.events_history = []
        
        # Metrics
        self.total_violations = 0
        self.total_scale_events = 0
        self.total_cost = 0.0
    
    def run(
        self,
        demand_df: pd.DataFrame,
        autoscaler,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run simulation over demand timeline with optional result saving.
        
        Args:
            demand_df: DataFrame with columns: step, time, cpu_demand, mem_demand, new_instances
                      (machines_reporting is optional and not used)
            autoscaler: Autoscaler policy instance with decide() method
            output_dir: Base output directory (if None, uses 'results')
            save_results: Whether to save results to disk
        
        Returns:
            Dictionary with simulation results:
            - timeline: DataFrame with step-by-step results
            - events: List of scaling events
            - metrics: Summary metrics
            - run_dir: Path to results directory (if save_results=True)
        
        Raises:
            ValueError: If demand_df is missing required columns
        """
        # Validate demand_df has required columns
        required_cols = ['step', 'time', 'cpu_demand', 'mem_demand']
        missing_cols = [col for col in required_cols if col not in demand_df.columns]
        if missing_cols:
            raise ValueError(
                f"demand_df missing required columns: {missing_cols}\n"
                f"Available columns: {list(demand_df.columns)}"
            )
        
        # Reset state
        self.current_machines = self.min_machines
        self.current_capacity = self.current_machines * self.machine_capacity
        
        # Clear histories
        self.time_history = []
        self.step_history = []
        self.demand_history = []
        self.capacity_history = []
        self.utilization_history = []
        self.machine_history = []
        self.violation_history = []
        self.events_history = []
        self.total_violations = 0
        self.total_scale_events = 0
        self.total_cost = 0.0
        
        # Run simulation step by step
        for idx, row in demand_df.iterrows():
            step = int(row['step'])
            time = float(row['time'])
            
            # Calculate total demand (combine CPU and memory, take max)
            cpu_demand = float(row['cpu_demand'])
            mem_demand = float(row['mem_demand'])
            total_demand = max(cpu_demand, mem_demand)
            
            # Calculate utilization
            if self.current_capacity > 0:
                utilization = total_demand / self.current_capacity
            else:
                utilization = float('inf')
            
            # Check for SLA violation
            violation = 1 if utilization > 1.0 else 0
            self.total_violations += violation
            
            # Make scaling decision
            scaling_action = autoscaler.decide(
                current_capacity=self.current_capacity,
                current_machines=self.current_machines,
                demand=total_demand,
                utilization=utilization,
                time=time
            )
            
            # Apply scaling action
            if scaling_action != 0:
                old_machines = self.current_machines
                self.current_machines += scaling_action
                
                # Enforce limits
                self.current_machines = max(
                    self.min_machines,
                    min(self.current_machines, self.max_machines)
                )
                
                # Update capacity
                self.current_capacity = self.current_machines * self.machine_capacity
                
                # Record event
                if self.current_machines != old_machines:
                    self.total_scale_events += 1
                    self.events_history.append({
                        'step': step,
                        'time': time,
                        'action': 'scale_up' if scaling_action > 0 else 'scale_down',
                        'old_machines': old_machines,
                        'new_machines': self.current_machines,
                        'delta': self.current_machines - old_machines
                    })
            
            # Calculate cost for this time step
            step_cost = self.current_machines * self.cost_per_machine_per_hour * (self.step_minutes / 60.0)
            self.total_cost += step_cost
            
            # Record history
            self.step_history.append(step)
            self.time_history.append(time)
            self.demand_history.append(total_demand)
            self.capacity_history.append(self.current_capacity)
            self.utilization_history.append(utilization)
            self.machine_history.append(self.current_machines)
            self.violation_history.append(violation)
        
        # Compile results
        timeline_df = pd.DataFrame({
            'step': self.step_history,
            'time': self.time_history,
            'demand': self.demand_history,
            'capacity': self.capacity_history,
            'utilization': self.utilization_history,
            'machines': self.machine_history,
            'violation': self.violation_history
        })
        
        metrics = {
            'total_violations': self.total_violations,
            'violation_rate': self.total_violations / len(demand_df) if len(demand_df) > 0 else 0.0,
            'total_scale_events': self.total_scale_events,
            'scale_up_events': sum(1 for e in self.events_history if e['action'] == 'scale_up'),
            'scale_down_events': sum(1 for e in self.events_history if e['action'] == 'scale_down'),
            'avg_utilization': float(np.mean(self.utilization_history)) if self.utilization_history else 0.0,
            'avg_machines': float(np.mean(self.machine_history)) if self.machine_history else 0.0,
            'min_machines': int(np.min(self.machine_history)) if self.machine_history else self.min_machines,
            'max_machines': int(np.max(self.machine_history)) if self.machine_history else self.min_machines,
            'total_cost': self.total_cost,
            'cost_per_hour': self.total_cost / (len(demand_df) * self.step_minutes / 60.0) if len(demand_df) > 0 else 0.0,
            'cost_per_step': self.total_cost / len(demand_df) if len(demand_df) > 0 else 0.0
        }
        
        results = {
            'timeline': timeline_df,
            'events': self.events_history,
            'metrics': metrics
        }
        
        # Save results if requested
        if save_results:
            run_dir = self._save_results(timeline_df, self.events_history, metrics, output_dir)
            results['run_dir'] = run_dir
        
        return results
    
    def _save_results(
        self,
        timeline: pd.DataFrame,
        events: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        output_dir: Optional[str]
    ) -> Path:
        """
        Save simulation results to disk.
        
        Args:
            timeline: Timeline DataFrame
            events: List of scaling events
            metrics: Metrics dictionary
            output_dir: Base output directory
        
        Returns:
            Path to run directory
        """
        # Create run directory
        if output_dir is None:
            output_dir = "results"
        
        base_dir = Path(output_dir)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save timeline
        timeline_path = run_dir / "timeline.csv"
        timeline.to_csv(timeline_path, index=False)
        
        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save scaling events
        if events:
            events_df = pd.DataFrame(events)
            events_path = run_dir / "scale_events.csv"
            events_df.to_csv(events_path, index=False)
        
        # Create plots directory
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        return run_dir

