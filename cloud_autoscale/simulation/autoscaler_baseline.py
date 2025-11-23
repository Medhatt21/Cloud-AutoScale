"""Baseline threshold-based autoscaler."""

from typing import Dict, Any


class BaselineAutoscaler:
    """Simple threshold-based autoscaling policy."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize baseline autoscaler.
        
        Production requirement: All configuration must be explicit.
        No default values are provided. Missing fields will raise KeyError.
        
        Args:
            config: Configuration with thresholds and cooldown periods
                    Required keys: upper_threshold, lower_threshold,
                                 max_scale_per_step, cooldown_steps, step_minutes
        
        Raises:
            KeyError: If required configuration keys are missing
        """
        # Strict access - will raise KeyError if missing
        self.upper_threshold = config['upper_threshold']
        self.lower_threshold = config['lower_threshold']
        self.max_scale_per_step = config['max_scale_per_step']
        self.cooldown_steps = config['cooldown_steps']
        self.step_minutes = config['step_minutes']
        
        # Defensive validation (config should already be validated, but double-check)
        assert 0 < self.lower_threshold < self.upper_threshold <= 1, \
            f"Invalid thresholds: lower={self.lower_threshold}, upper={self.upper_threshold}"
        
        # State (internal, not config)
        self.last_scale_up_time = -999
        self.last_scale_down_time = -999
    
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

