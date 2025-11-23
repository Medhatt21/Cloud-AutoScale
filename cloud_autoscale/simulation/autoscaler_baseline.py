"""Production-grade baseline threshold-based autoscaler with strict validation."""

from typing import Dict, Any


class BaselineAutoscaler:
    """
    Simple threshold-based autoscaling policy.
    
    Production requirement: All configuration must be explicit.
    No defaults, no silent failures.
    """
    
    def __init__(self, autoscaler_config: Dict[str, Any], step_minutes: float):
        """
        Initialize baseline autoscaler with strict validation.
        
        Args:
            autoscaler_config: Autoscaler configuration dictionary
                              Required keys: upper_threshold, lower_threshold,
                                           max_scale_per_step, cooldown_steps
            step_minutes: Time step size in minutes (from simulation config)
        
        Raises:
            ValueError: If required configuration keys are missing or invalid
        """
        # Validate presence of all required keys
        required_keys = [
            'upper_threshold',
            'lower_threshold',
            'max_scale_per_step',
            'cooldown_steps'
        ]
        
        missing_keys = [key for key in required_keys if key not in autoscaler_config]
        if missing_keys:
            raise ValueError(
                f"Missing required autoscaler config keys: {missing_keys}"
            )
        
        # Validate types and values
        self.upper_threshold = autoscaler_config['upper_threshold']
        if not isinstance(self.upper_threshold, (int, float)) or not (0 < self.upper_threshold <= 1):
            raise ValueError("upper_threshold must be a number between 0 and 1")
        
        self.lower_threshold = autoscaler_config['lower_threshold']
        if not isinstance(self.lower_threshold, (int, float)) or not (0 < self.lower_threshold < self.upper_threshold):
            raise ValueError("lower_threshold must be a number between 0 and upper_threshold")
        
        self.max_scale_per_step = autoscaler_config['max_scale_per_step']
        if not isinstance(self.max_scale_per_step, int) or self.max_scale_per_step < 1:
            raise ValueError("max_scale_per_step must be an integer >= 1")
        
        self.cooldown_steps = autoscaler_config['cooldown_steps']
        if not isinstance(self.cooldown_steps, int) or self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be a non-negative integer")
        
        self.step_minutes = step_minutes
        if not isinstance(self.step_minutes, (int, float)) or self.step_minutes <= 0:
            raise ValueError("step_minutes must be a positive number")
        
        # State (internal, not config)
        self.last_scale_up_step = -999999
        self.last_scale_down_step = -999999
    
    def decide(
        self,
        current_capacity: float,
        current_machines: int,
        demand: float,
        utilization: float,
        time: float
    ) -> int:
        """
        Make scaling decision based on current state.
        
        Args:
            current_capacity: Current total capacity
            current_machines: Current number of machines
            demand: Current demand
            utilization: Current utilization (0-1+, can be inf)
            time: Current simulation time in minutes
        
        Returns:
            Number of machines to add (positive) or remove (negative), or 0 for no action
        """
        current_step = int(time / self.step_minutes)
        
        # Check cooldown
        steps_since_scale_up = current_step - self.last_scale_up_step
        steps_since_scale_down = current_step - self.last_scale_down_step
        
        # Scale up if utilization exceeds upper threshold
        if utilization > self.upper_threshold:
            if steps_since_scale_up >= self.cooldown_steps:
                self.last_scale_up_step = current_step
                return self.max_scale_per_step
        
        # Scale down if utilization below lower threshold
        elif utilization < self.lower_threshold:
            if steps_since_scale_down >= self.cooldown_steps:
                self.last_scale_down_step = current_step
                return -self.max_scale_per_step
        
        # No action
        return 0

