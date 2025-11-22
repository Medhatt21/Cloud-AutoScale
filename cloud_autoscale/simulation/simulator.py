"""Simplified cloud autoscaling simulator."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class CloudSimulator:
    """Simple discrete-time simulator for cloud autoscaling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize simulator.
        
        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config
        self.step_minutes = config.get('step_minutes', 5)
        
        # State
        self.current_capacity = 0
        self.current_machines = 0
        
        # History
        self.capacity_history = []
        self.utilization_history = []
        self.demand_history = []
        self.machine_history = []
        self.violations_history = []
        self.events_history = []
        
        # Metrics
        self.total_violations = 0
        self.total_scale_events = 0
        self.total_cost = 0.0
    
    def run(self, demand_df: pd.DataFrame, autoscaler) -> Dict[str, Any]:
        """
        Run simulation over demand timeline.
        
        Args:
            demand_df: DataFrame with columns: time, cpu_demand, mem_demand, new_instances
            autoscaler: Autoscaler policy instance
        
        Returns:
            Dictionary with simulation results
        """
        # Initialize with minimum capacity
        min_machines = self.config.get('min_machines', 1)
        machine_capacity = self.config.get('machine_capacity', 10)  # units per machine
        
        self.current_machines = min_machines
        self.current_capacity = self.current_machines * machine_capacity
        
        # Run simulation step by step
        for idx, row in demand_df.iterrows():
            time = row['time']
            
            # Calculate total demand (combine CPU and memory, take max)
            cpu_demand = row['cpu_demand']
            mem_demand = row['mem_demand']
            total_demand = max(cpu_demand, mem_demand)
            
            # Calculate utilization
            if self.current_capacity > 0:
                utilization = total_demand / self.current_capacity
            else:
                utilization = 1.0
            
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
                max_machines = self.config.get('max_machines', 100)
                self.current_machines = max(min_machines, min(self.current_machines, max_machines))
                
                # Update capacity
                self.current_capacity = self.current_machines * machine_capacity
                
                # Record event
                if self.current_machines != old_machines:
                    self.total_scale_events += 1
                    self.events_history.append({
                        'time': time,
                        'action': 'scale_up' if scaling_action > 0 else 'scale_down',
                        'old_machines': old_machines,
                        'new_machines': self.current_machines
                    })
            
            # Calculate cost (simple: cost per machine per time step)
            cost_per_machine_per_hour = self.config.get('cost_per_machine_per_hour', 0.1)
            step_cost = self.current_machines * cost_per_machine_per_hour * (self.step_minutes / 60)
            self.total_cost += step_cost
            
            # Record history
            self.capacity_history.append(self.current_capacity)
            self.utilization_history.append(utilization)
            self.demand_history.append(total_demand)
            self.machine_history.append(self.current_machines)
            self.violations_history.append(violation)
        
        # Compile results
        results = {
            'timeline': pd.DataFrame({
                'time': demand_df['time'],
                'demand': self.demand_history,
                'capacity': self.capacity_history,
                'utilization': self.utilization_history,
                'machines': self.machine_history,
                'violation': self.violations_history
            }),
            'events': self.events_history,
            'metrics': {
                'total_violations': self.total_violations,
                'violation_rate': self.total_violations / len(demand_df) if len(demand_df) > 0 else 0,
                'total_scale_events': self.total_scale_events,
                'avg_utilization': np.mean(self.utilization_history) if self.utilization_history else 0,
                'avg_machines': np.mean(self.machine_history) if self.machine_history else 0,
                'total_cost': self.total_cost,
                'stability': self.total_scale_events  # Lower is better
            }
        }
        
        return results

