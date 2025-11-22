"""Baseline threshold-based autoscaler."""

from typing import Dict, Any


class BaselineAutoscaler:
    """Simple threshold-based autoscaling policy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize baseline autoscaler.
        
        Args:
            config: Configuration with thresholds and cooldown periods
        """
        self.upper_threshold = config.get('upper_threshold', 0.7)
        self.lower_threshold = config.get('lower_threshold', 0.3)
        self.max_scale_per_step = config.get('max_scale_per_step', 1)
        self.cooldown_steps = config.get('cooldown_steps', 2)
        
        # State
        self.last_scale_up_time = -999
        self.last_scale_down_time = -999
        self.step_minutes = config.get('step_minutes', 5)
    
    def decide(self, current_capacity: float, current_machines: int, 
               demand: float, utilization: float, time: float) -> int:
        """
        Make scaling decision based on current state.
        
        Args:
            current_capacity: Current total capacity
            current_machines: Current number of machines
            demand: Current demand
            utilization: Current utilization (0-1+)
            time: Current simulation time
        
        Returns:
            Number of machines to add (positive) or remove (negative), or 0 for no action
        """
        current_step = int(time / self.step_minutes)
        
        # Check cooldown
        steps_since_scale_up = current_step - self.last_scale_up_time
        steps_since_scale_down = current_step - self.last_scale_down_time
        
        # Scale up if utilization exceeds upper threshold
        if utilization > self.upper_threshold:
            if steps_since_scale_up >= self.cooldown_steps:
                self.last_scale_up_time = current_step
                return self.max_scale_per_step
        
        # Scale down if utilization below lower threshold
        elif utilization < self.lower_threshold:
            if steps_since_scale_down >= self.cooldown_steps:
                self.last_scale_down_time = current_step
                return -self.max_scale_per_step
        
        # No action
        return 0

