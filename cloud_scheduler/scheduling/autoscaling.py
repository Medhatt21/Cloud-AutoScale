"""Autoscaling policies for cloud infrastructure."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta
import numpy as np
from loguru import logger

from ..core.resources import Host, VirtualMachine, ResourceSpecs, ResourceType
from ..core.workload import Workload
from ..core.simulator import SimulationMetrics


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingDecision:
    """A scaling decision with action and parameters."""
    action: ScalingAction
    resource_type: str  # "host", "vm", "container"
    count: int  # number of resources to add/remove
    reason: str
    confidence: float = 1.0
    target_specs: Optional[ResourceSpecs] = None


@dataclass
class AutoscalingConfig:
    """Configuration for autoscaling policies."""
    # Threshold-based scaling
    cpu_scale_up_threshold: float = 0.8
    cpu_scale_down_threshold: float = 0.3
    memory_scale_up_threshold: float = 0.8
    memory_scale_down_threshold: float = 0.3
    
    # Queue-based scaling
    queue_scale_up_threshold: int = 10
    queue_scale_down_threshold: int = 0
    
    # Timing constraints
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    evaluation_window: float = 180.0  # 3 minutes
    
    # Scaling limits
    min_hosts: int = 1
    max_hosts: int = 100
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    
    # Advanced settings
    enable_predictive_scaling: bool = False
    enable_scheduled_scaling: bool = False


class BaseAutoscaler(ABC):
    """Abstract base class for autoscaling policies."""
    
    def __init__(self, config: AutoscalingConfig):
        self.config = config
        self.last_scale_up_time: float = 0.0
        self.last_scale_down_time: float = 0.0
        self.metrics_history: List[SimulationMetrics] = []
        self.scaling_history: List[Tuple[float, ScalingDecision]] = []
        
        logger.info(f"Autoscaler initialized: {self.__class__.__name__}")
    
    @abstractmethod
    def make_scaling_decision(
        self,
        current_metrics: SimulationMetrics,
        hosts: Dict[str, Host],
        workload_queue: List[Workload]
    ) -> Optional[ScalingDecision]:
        """Make a scaling decision based on current state."""
        pass
    
    def _can_scale_up(self, current_time: float) -> bool:
        """Check if scaling up is allowed (cooldown period)."""
        return (current_time - self.last_scale_up_time) >= self.config.scale_up_cooldown
    
    def _can_scale_down(self, current_time: float) -> bool:
        """Check if scaling down is allowed (cooldown period)."""
        return (current_time - self.last_scale_down_time) >= self.config.scale_down_cooldown
    
    def _update_scaling_history(self, timestamp: float, decision: ScalingDecision) -> None:
        """Update scaling history."""
        self.scaling_history.append((timestamp, decision))
        
        if decision.action == ScalingAction.SCALE_UP:
            self.last_scale_up_time = timestamp
        elif decision.action == ScalingAction.SCALE_DOWN:
            self.last_scale_down_time = timestamp
    
    def _get_recent_metrics(self, window_seconds: float) -> List[SimulationMetrics]:
        """Get metrics within the specified time window."""
        if not self.metrics_history:
            return []
        
        current_time = self.metrics_history[-1].timestamp
        cutoff_time = current_time - window_seconds
        
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class ThresholdAutoscaler(BaseAutoscaler):
    """Threshold-based autoscaler similar to AWS Auto Scaling."""
    
    def __init__(self, config: AutoscalingConfig):
        super().__init__(config)
    
    def make_scaling_decision(
        self,
        current_metrics: SimulationMetrics,
        hosts: Dict[str, Host],
        workload_queue: List[Workload]
    ) -> Optional[ScalingDecision]:
        """Make scaling decision based on resource utilization thresholds."""
        self.metrics_history.append(current_metrics)
        current_time = current_metrics.timestamp
        
        # Get recent metrics for decision
        recent_metrics = self._get_recent_metrics(self.config.evaluation_window)
        if len(recent_metrics) < 2:
            return None
        
        # Calculate average utilization
        avg_cpu = sum(m.avg_cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.avg_memory_utilization for m in recent_metrics) / len(recent_metrics)
        avg_queue_length = sum(m.queued_workloads for m in recent_metrics) / len(recent_metrics)
        
        # Check for scale-up conditions
        scale_up_reasons = []
        if avg_cpu > self.config.cpu_scale_up_threshold:
            scale_up_reasons.append(f"CPU utilization {avg_cpu:.2f} > {self.config.cpu_scale_up_threshold}")
        
        if avg_memory > self.config.memory_scale_up_threshold:
            scale_up_reasons.append(f"Memory utilization {avg_memory:.2f} > {self.config.memory_scale_up_threshold}")
        
        if avg_queue_length > self.config.queue_scale_up_threshold:
            scale_up_reasons.append(f"Queue length {avg_queue_length:.1f} > {self.config.queue_scale_up_threshold}")
        
        # Check for scale-down conditions
        scale_down_reasons = []
        if (avg_cpu < self.config.cpu_scale_down_threshold and 
            avg_memory < self.config.memory_scale_down_threshold and
            avg_queue_length <= self.config.queue_scale_down_threshold):
            scale_down_reasons.append(f"Low utilization: CPU {avg_cpu:.2f}, Memory {avg_memory:.2f}")
        
        # Make scaling decision
        current_host_count = len([h for h in hosts.values() if h.state.value == "available"])
        
        if scale_up_reasons and self._can_scale_up(current_time):
            if current_host_count < self.config.max_hosts:
                decision = ScalingDecision(
                    action=ScalingAction.SCALE_UP,
                    resource_type="host",
                    count=self.config.scale_up_increment,
                    reason="; ".join(scale_up_reasons),
                    confidence=min(1.0, max(avg_cpu, avg_memory) - 0.5)
                )
                
                self._update_scaling_history(current_time, decision)
                logger.info(f"Scaling up: {decision.reason}")
                return decision
        
        elif scale_down_reasons and self._can_scale_down(current_time):
            if current_host_count > self.config.min_hosts:
                decision = ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    resource_type="host",
                    count=self.config.scale_down_increment,
                    reason="; ".join(scale_down_reasons),
                    confidence=1.0 - max(avg_cpu, avg_memory)
                )
                
                self._update_scaling_history(current_time, decision)
                logger.info(f"Scaling down: {decision.reason}")
                return decision
        
        return None


class ScheduledAutoscaler(BaseAutoscaler):
    """Scheduled autoscaler for predictable workload patterns."""
    
    def __init__(self, config: AutoscalingConfig):
        super().__init__(config)
        self.scaling_schedule: List[Tuple[str, int, int]] = []  # (time_pattern, min_hosts, max_hosts)
        self._setup_default_schedule()
    
    def _setup_default_schedule(self) -> None:
        """Setup default scaling schedule (business hours)."""
        # Business hours: scale up
        self.scaling_schedule.extend([
            ("08:00", 5, 20),  # Morning scale-up
            ("12:00", 3, 15),  # Lunch time scale-down
            ("13:00", 5, 20),  # Afternoon scale-up
            ("18:00", 2, 10),  # Evening scale-down
            ("22:00", 1, 5),   # Night scale-down
        ])
    
    def add_scheduled_scaling(self, time_pattern: str, min_hosts: int, max_hosts: int) -> None:
        """Add a scheduled scaling rule."""
        self.scaling_schedule.append((time_pattern, min_hosts, max_hosts))
        self.scaling_schedule.sort(key=lambda x: x[0])
        logger.info(f"Added scheduled scaling: {time_pattern} -> {min_hosts}-{max_hosts} hosts")
    
    def make_scaling_decision(
        self,
        current_metrics: SimulationMetrics,
        hosts: Dict[str, Host],
        workload_queue: List[Workload]
    ) -> Optional[ScalingDecision]:
        """Make scaling decision based on schedule and thresholds."""
        self.metrics_history.append(current_metrics)
        current_time = current_metrics.timestamp
        
        # Get current time of day (simulate with modulo for demo)
        simulated_hour = int((current_time / 3600) % 24)
        simulated_minute = int((current_time / 60) % 60)
        current_time_str = f"{simulated_hour:02d}:{simulated_minute:02d}"
        
        # Find applicable schedule
        target_min_hosts = self.config.min_hosts
        target_max_hosts = self.config.max_hosts
        
        for time_pattern, min_hosts, max_hosts in self.scaling_schedule:
            if current_time_str >= time_pattern:
                target_min_hosts = min_hosts
                target_max_hosts = max_hosts
        
        current_host_count = len([h for h in hosts.values() if h.state.value == "available"])
        
        # Check if we need to scale based on schedule
        if current_host_count < target_min_hosts and self._can_scale_up(current_time):
            decision = ScalingDecision(
                action=ScalingAction.SCALE_UP,
                resource_type="host",
                count=target_min_hosts - current_host_count,
                reason=f"Scheduled scaling: need {target_min_hosts} hosts at {current_time_str}",
                confidence=1.0
            )
            
            self._update_scaling_history(current_time, decision)
            logger.info(f"Scheduled scale-up: {decision.reason}")
            return decision
        
        elif current_host_count > target_max_hosts and self._can_scale_down(current_time):
            decision = ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                resource_type="host",
                count=current_host_count - target_max_hosts,
                reason=f"Scheduled scaling: limit to {target_max_hosts} hosts at {current_time_str}",
                confidence=1.0
            )
            
            self._update_scaling_history(current_time, decision)
            logger.info(f"Scheduled scale-down: {decision.reason}")
            return decision
        
        # Fall back to threshold-based scaling within schedule limits
        threshold_autoscaler = ThresholdAutoscaler(self.config)
        threshold_decision = threshold_autoscaler.make_scaling_decision(
            current_metrics, hosts, workload_queue
        )
        
        if threshold_decision:
            # Adjust decision to respect schedule limits
            if threshold_decision.action == ScalingAction.SCALE_UP:
                max_scale_up = target_max_hosts - current_host_count
                if max_scale_up > 0:
                    threshold_decision.count = min(threshold_decision.count, max_scale_up)
                    threshold_decision.reason += f" (limited by schedule: max {target_max_hosts})"
                    return threshold_decision
            
            elif threshold_decision.action == ScalingAction.SCALE_DOWN:
                max_scale_down = current_host_count - target_min_hosts
                if max_scale_down > 0:
                    threshold_decision.count = min(threshold_decision.count, max_scale_down)
                    threshold_decision.reason += f" (limited by schedule: min {target_min_hosts})"
                    return threshold_decision
        
        return None


class PredictiveAutoscaler(BaseAutoscaler):
    """Predictive autoscaler using simple forecasting."""
    
    def __init__(self, config: AutoscalingConfig):
        super().__init__(config)
        self.prediction_horizon = 600.0  # 10 minutes
        self.min_history_points = 10
    
    def make_scaling_decision(
        self,
        current_metrics: SimulationMetrics,
        hosts: Dict[str, Host],
        workload_queue: List[Workload]
    ) -> Optional[ScalingDecision]:
        """Make scaling decision based on workload prediction."""
        self.metrics_history.append(current_metrics)
        current_time = current_metrics.timestamp
        
        # Need sufficient history for prediction
        if len(self.metrics_history) < self.min_history_points:
            return None
        
        # Simple trend-based prediction
        recent_metrics = self._get_recent_metrics(self.config.evaluation_window * 2)
        if len(recent_metrics) < 5:
            return None
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.avg_cpu_utilization for m in recent_metrics])
        memory_trend = self._calculate_trend([m.avg_memory_utilization for m in recent_metrics])
        queue_trend = self._calculate_trend([float(m.queued_workloads) for m in recent_metrics])
        
        # Predict future values
        current_cpu = current_metrics.avg_cpu_utilization
        current_memory = current_metrics.avg_memory_utilization
        current_queue = float(current_metrics.queued_workloads)
        
        predicted_cpu = current_cpu + cpu_trend * (self.prediction_horizon / 60.0)  # per minute
        predicted_memory = current_memory + memory_trend * (self.prediction_horizon / 60.0)
        predicted_queue = max(0, current_queue + queue_trend * (self.prediction_horizon / 60.0))
        
        logger.debug(f"Predictions: CPU {predicted_cpu:.3f}, Memory {predicted_memory:.3f}, "
                    f"Queue {predicted_queue:.1f}")
        
        # Make proactive scaling decisions
        current_host_count = len([h for h in hosts.values() if h.state.value == "available"])
        
        # Proactive scale-up
        if ((predicted_cpu > self.config.cpu_scale_up_threshold or 
             predicted_memory > self.config.memory_scale_up_threshold or
             predicted_queue > self.config.queue_scale_up_threshold) and
            self._can_scale_up(current_time) and
            current_host_count < self.config.max_hosts):
            
            confidence = max(
                max(0, predicted_cpu - self.config.cpu_scale_up_threshold),
                max(0, predicted_memory - self.config.memory_scale_up_threshold),
                max(0, (predicted_queue - self.config.queue_scale_up_threshold) / 10.0)
            )
            
            decision = ScalingDecision(
                action=ScalingAction.SCALE_UP,
                resource_type="host",
                count=self.config.scale_up_increment,
                reason=f"Predictive scaling: CPU→{predicted_cpu:.2f}, Mem→{predicted_memory:.2f}, Queue→{predicted_queue:.1f}",
                confidence=min(1.0, confidence)
            )
            
            self._update_scaling_history(current_time, decision)
            logger.info(f"Predictive scale-up: {decision.reason}")
            return decision
        
        # Proactive scale-down (more conservative)
        elif ((predicted_cpu < self.config.cpu_scale_down_threshold and 
               predicted_memory < self.config.memory_scale_down_threshold and
               predicted_queue <= self.config.queue_scale_down_threshold) and
              self._can_scale_down(current_time) and
              current_host_count > self.config.min_hosts):
            
            # Only scale down if trend is consistently downward
            if cpu_trend < 0 and memory_trend < 0:
                confidence = min(
                    self.config.cpu_scale_down_threshold - predicted_cpu,
                    self.config.memory_scale_down_threshold - predicted_memory
                )
                
                decision = ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    resource_type="host",
                    count=self.config.scale_down_increment,
                    reason=f"Predictive scaling: CPU→{predicted_cpu:.2f}, Mem→{predicted_memory:.2f}",
                    confidence=max(0.0, confidence)
                )
                
                self._update_scaling_history(current_time, decision)
                logger.info(f"Predictive scale-down: {decision.reason}")
                return decision
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Linear regression slope
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        return slope


class HybridAutoscaler(BaseAutoscaler):
    """Hybrid autoscaler combining multiple strategies."""
    
    def __init__(self, config: AutoscalingConfig):
        super().__init__(config)
        self.threshold_autoscaler = ThresholdAutoscaler(config)
        self.scheduled_autoscaler = ScheduledAutoscaler(config)
        self.predictive_autoscaler = PredictiveAutoscaler(config)
        
        # Weights for different strategies
        self.strategy_weights = {
            'threshold': 0.4,
            'scheduled': 0.3,
            'predictive': 0.3
        }
    
    def make_scaling_decision(
        self,
        current_metrics: SimulationMetrics,
        hosts: Dict[str, Host],
        workload_queue: List[Workload]
    ) -> Optional[ScalingDecision]:
        """Make scaling decision using hybrid approach."""
        self.metrics_history.append(current_metrics)
        
        # Get decisions from all strategies
        threshold_decision = self.threshold_autoscaler.make_scaling_decision(
            current_metrics, hosts, workload_queue
        )
        scheduled_decision = self.scheduled_autoscaler.make_scaling_decision(
            current_metrics, hosts, workload_queue
        )
        predictive_decision = self.predictive_autoscaler.make_scaling_decision(
            current_metrics, hosts, workload_queue
        )
        
        decisions = [
            ('threshold', threshold_decision),
            ('scheduled', scheduled_decision),
            ('predictive', predictive_decision)
        ]
        
        # Filter out None decisions
        valid_decisions = [(name, decision) for name, decision in decisions if decision is not None]
        
        if not valid_decisions:
            return None
        
        # If all agree on action type, combine them
        actions = [decision.action for _, decision in valid_decisions]
        if len(set(actions)) == 1:
            # All agree on action type
            action = actions[0]
            
            # Weighted average of counts and confidence
            total_weight = sum(self.strategy_weights[name] for name, _ in valid_decisions)
            weighted_count = sum(
                self.strategy_weights[name] * decision.count 
                for name, decision in valid_decisions
            ) / total_weight
            
            weighted_confidence = sum(
                self.strategy_weights[name] * decision.confidence 
                for name, decision in valid_decisions
            ) / total_weight
            
            # Combine reasons
            reasons = [f"{name}: {decision.reason}" for name, decision in valid_decisions]
            
            hybrid_decision = ScalingDecision(
                action=action,
                resource_type="host",
                count=int(round(weighted_count)),
                reason=f"Hybrid decision - {'; '.join(reasons)}",
                confidence=weighted_confidence
            )
            
            self._update_scaling_history(current_metrics.timestamp, hybrid_decision)
            logger.info(f"Hybrid scaling decision: {hybrid_decision.reason}")
            return hybrid_decision
        
        else:
            # Conflicting decisions - use highest confidence
            best_decision = max(valid_decisions, key=lambda x: x[1].confidence)
            decision = best_decision[1]
            decision.reason = f"Hybrid (conflict resolved by confidence): {decision.reason}"
            
            self._update_scaling_history(current_metrics.timestamp, decision)
            logger.info(f"Hybrid scaling decision (conflict): {decision.reason}")
            return decision
