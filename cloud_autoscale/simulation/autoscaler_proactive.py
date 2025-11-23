"""Production-grade ML-powered proactive autoscaler with strict validation."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from cloud_autoscale.forecasting import ForecastingModel


@dataclass
class ProactiveAutoscalerConfig:
    """Configuration for proactive autoscaler."""
    upper_threshold: float = 0.70
    lower_threshold: float = 0.30
    safety_margin: float = 1.10
    cooldown_steps: int = 3
    max_scale_per_step: int = 3
    history_window: int = 30  # number of recent steps to use for forecasting


class ProactiveAutoscaler:
    """
    ML-powered autoscaler that uses ForecastingModel to look ahead
    and decide scaling actions *before* violations occur.
    
    This autoscaler matches the interface of BaselineAutoscaler but adds
    ML forecasting capabilities for proactive scaling decisions.
    """
    
    def __init__(
        self,
        forecast_model: ForecastingModel,
        autoscaler_config: Dict[str, Any],
        step_minutes: float,
        min_machines: int = 1,
        max_machines: int = 1000,
    ):
        """
        Initialize proactive autoscaler with strict validation.
        
        Args:
            forecast_model: Trained ForecastingModel instance
            autoscaler_config: Autoscaler configuration dictionary
                    Required keys: upper_threshold, lower_threshold,
                                   max_scale_per_step, cooldown_steps
                    Optional keys: safety_margin, history_window
            step_minutes: Time step size in minutes (from simulation config)
            min_machines: Minimum number of machines allowed
            max_machines: Maximum number of machines allowed
        
        Raises:
            ValueError: If required configuration keys are missing or invalid
        """
        self.forecast_model = forecast_model
        self.step_minutes = step_minutes
        self.min_machines = min_machines
        self.max_machines = max_machines
        
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
        
        # Optional parameters with defaults
        self.safety_margin = autoscaler_config.get('safety_margin', 1.10)
        if not isinstance(self.safety_margin, (int, float)) or self.safety_margin < 1.0:
            raise ValueError("safety_margin must be a number >= 1.0")
        
        self.history_window = autoscaler_config.get('history_window', 30)
        if not isinstance(self.history_window, int) or self.history_window < 20:
            raise ValueError("history_window must be an integer >= 20")
        
        # State (internal, not config)
        self.last_scale_up_step = -999999
        self.last_scale_down_step = -999999
        self._cooldown_remaining = 0
    
    def decide(
        self,
        current_capacity: float,
        current_machines: int,
        demand: float,
        utilization: float,
        time: float,
        history_df: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Make scaling decision based on current state and ML forecast.
        
        This method signature matches BaselineAutoscaler.decide() but adds
        an optional history_df parameter for ML forecasting.
        
        Args:
            current_capacity: Current total capacity
            current_machines: Current number of machines
            demand: Current demand
            utilization: Current utilization (0-1+, can be inf)
            time: Current simulation time in minutes
            history_df: Optional DataFrame with historical data for forecasting
        
        Returns:
            Number of machines to add (positive) or remove (negative), or 0 for no action
        """
        current_step = int(time / self.step_minutes)
        
        # Check cooldown
        steps_since_scale_up = current_step - self.last_scale_up_step
        steps_since_scale_down = current_step - self.last_scale_down_step
        
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return 0
        
        # If no history provided or not enough history, fall back to reactive behavior
        if history_df is None or len(history_df) < self.history_window:
            # Reactive fallback (same as baseline)
            if utilization > self.upper_threshold:
                if steps_since_scale_up >= self.cooldown_steps:
                    self.last_scale_up_step = current_step
                    self._cooldown_remaining = self.cooldown_steps
                    return self.max_scale_per_step
            elif utilization < self.lower_threshold:
                if steps_since_scale_down >= self.cooldown_steps:
                    self.last_scale_down_step = current_step
                    self._cooldown_remaining = self.cooldown_steps
                    return -self.max_scale_per_step
            return 0
        
        # Clip history window
        history_window = history_df.tail(self.history_window)
        
        # Forecast future CPU demand using forward horizon
        try:
            forecast = self.forecast_model.multi_horizon(history_window)
            seq = forecast["full"]
            
            # Use horizon-max for better anticipation (first 3 steps only)
            horizon_cpu = max(seq[0:3]) * self.safety_margin
        except Exception as e:
            # If forecasting fails, fall back to reactive behavior
            print(f"Warning: Forecasting failed ({e}), using reactive policy")
            if utilization > self.upper_threshold:
                if steps_since_scale_up >= self.cooldown_steps:
                    self.last_scale_up_step = current_step
                    self._cooldown_remaining = self.cooldown_steps
                    return self.max_scale_per_step
            elif utilization < self.lower_threshold:
                if steps_since_scale_down >= self.cooldown_steps:
                    self.last_scale_down_step = current_step
                    self._cooldown_remaining = self.cooldown_steps
                    return -self.max_scale_per_step
            return 0
        
        # Calculate machine capacity (demand is max of CPU and memory)
        machine_capacity = current_capacity / max(current_machines, 1)
        
        # Calculate projected utilization using horizon forecast
        projected_util = horizon_cpu / max(current_capacity, 1e-6)
        
        scale_delta = 0
        
        if projected_util >= self.upper_threshold:
            # Scale up proactively based on horizon forecast
            desired_machines = int(np.ceil(horizon_cpu / machine_capacity))
            
            # Ensure at least +1 and obey max per step
            delta = desired_machines - current_machines
            if delta > 0:
                delta = min(delta, self.max_scale_per_step)
            scale_delta = delta
            
        elif projected_util <= self.lower_threshold:
            # Scale down less aggressively
            desired_machines = max(
                self.min_machines,
                int(np.ceil(horizon_cpu / machine_capacity))
            )
            
            delta = desired_machines - current_machines
            
            if delta < 0:
                delta = max(delta, -self.max_scale_per_step)
            
            scale_delta = delta
        
        # Clamp to bounds
        new_machines = current_machines + scale_delta
        if new_machines < self.min_machines:
            scale_delta = self.min_machines - current_machines
        elif new_machines > self.max_machines:
            scale_delta = self.max_machines - current_machines
        
        # Apply cooldown if we actually scaled
        if scale_delta != 0:
            if scale_delta > 0:
                self.last_scale_up_step = current_step
            else:
                self.last_scale_down_step = current_step
            self._cooldown_remaining = self.cooldown_steps
        
        return scale_delta

