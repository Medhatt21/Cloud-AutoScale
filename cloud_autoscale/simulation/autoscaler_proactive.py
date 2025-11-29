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
    down_cooldown_multiplier: int = 3  # multiplier for downscale cooldown
    downscale_confirmation: bool = True  # require double-confirmation for downscale


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
            'cooldown_steps',
            'down_cooldown_multiplier',
            'downscale_confirmation'
        ]
        
        missing_keys = [key for key in required_keys if key not in autoscaler_config]
        if missing_keys:
            raise KeyError(
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
        
        # New required parameters
        self.down_cooldown_multiplier = autoscaler_config['down_cooldown_multiplier']
        if not isinstance(self.down_cooldown_multiplier, int) or self.down_cooldown_multiplier < 1:
            raise ValueError("down_cooldown_multiplier must be an integer >= 1")
        
        self.downscale_confirmation = autoscaler_config['downscale_confirmation']
        if not isinstance(self.downscale_confirmation, bool):
            raise ValueError("downscale_confirmation must be a boolean")
        
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
        
        # Reactive fallback only allowed when history is insufficient
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
                    self._cooldown_remaining = self.cooldown_steps * self.down_cooldown_multiplier
                    return -1  # Hard cap downscale at -1
            return 0
        
        # Clip history window
        history_window = history_df.tail(self.history_window)
        
        # Forecast future CPU demand using direct multi-horizon models
        # No try-except: fail-fast if forecasting fails
        forecast = self.forecast_model.multi_horizon(history_window)
        
        # Asymmetric horizon logic
        # Scale-up: use aggressive horizon (max of t+1 and 80% of t+3) with safety margin
        horizon_cpu_up = max(
            forecast["t+1"],
            0.8 * forecast["t+3"]
        ) * self.safety_margin
        
        # Scale-down: use conservative horizon (90% of t+1 only)
        horizon_cpu_down = forecast["t+1"] * 0.9
        
        # Calculate machine capacity (demand is max of CPU and memory)
        machine_capacity = current_capacity / max(current_machines, 1)
        
        # Calculate projected utilization for scale-up using aggressive horizon
        projected_util_up = horizon_cpu_up / max(current_capacity, 1e-6)
        
        # Calculate projected utilization for scale-down using conservative horizon
        projected_util_down = horizon_cpu_down / max(current_capacity, 1e-6)
        
        scale_delta = 0
        
        if projected_util_up >= self.upper_threshold:
            # Scale up proactively based on aggressive horizon forecast
            desired_machines = int(np.ceil(horizon_cpu_up / machine_capacity))
            
            # Ensure at least +1 and obey max per step
            delta = desired_machines - current_machines
            if delta > 0:
                delta = min(delta, self.max_scale_per_step)
            scale_delta = delta
            
        elif projected_util_down <= self.lower_threshold:
            # Scale down conservatively with double-confirmation rule
            if self.downscale_confirmation:
                # Both current and projected must be below threshold
                if utilization <= self.lower_threshold and projected_util_down <= self.lower_threshold:
                    desired_machines = max(
                        self.min_machines,
                        int(np.ceil(horizon_cpu_down / machine_capacity))
                    )
                    
                    delta = desired_machines - current_machines
                    
                    if delta < 0:
                        # Hard cap: max downscale per step = 1
                        delta = -1
                    
                    scale_delta = delta
                # else: skip downscale entirely
            else:
                # No double-confirmation required
                desired_machines = max(
                    self.min_machines,
                    int(np.ceil(horizon_cpu_down / machine_capacity))
                )
                
                delta = desired_machines - current_machines
                
                if delta < 0:
                    # Hard cap: max downscale per step = 1
                    delta = -1
                
                scale_delta = delta
        
        # Clamp to bounds
        new_machines = current_machines + scale_delta
        if new_machines < self.min_machines:
            scale_delta = self.min_machines - current_machines
        elif new_machines > self.max_machines:
            scale_delta = self.max_machines - current_machines
        
        # Apply asymmetric cooldown if we actually scaled
        if scale_delta != 0:
            if scale_delta > 0:
                self.last_scale_up_step = current_step
                # Upscale: normal cooldown
                self._cooldown_remaining = self.cooldown_steps
            else:
                self.last_scale_down_step = current_step
                # Downscale: extended cooldown
                self._cooldown_remaining = self.cooldown_steps * self.down_cooldown_multiplier
        
        return scale_delta

