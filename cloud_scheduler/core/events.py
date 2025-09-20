"""Simulation events and event types."""

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass
from loguru import logger


class EventType(Enum):
    """Types of simulation events."""
    
    # Workload events
    WORKLOAD_ARRIVAL = "workload_arrival"
    WORKLOAD_COMPLETION = "workload_completion"
    WORKLOAD_FAILURE = "workload_failure"
    
    # Resource events
    HOST_FAILURE = "host_failure"
    HOST_RECOVERY = "host_recovery"
    VM_START = "vm_start"
    VM_STOP = "vm_stop"
    CONTAINER_START = "container_start"
    CONTAINER_STOP = "container_stop"
    
    # Scaling events
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    AUTOSCALE_CHECK = "autoscale_check"
    
    # Monitoring events
    METRICS_COLLECTION = "metrics_collection"
    SLA_CHECK = "sla_check"


@dataclass
class SimulationEvent:
    """A simulation event with timestamp and associated data."""
    
    timestamp: float
    event_type: EventType
    resource_id: str
    data: Dict[str, Any]
    priority: int = 0
    
    def __post_init__(self) -> None:
        """Log event creation."""
        logger.debug(
            f"Event created: {self.event_type.value} at {self.timestamp:.2f}s "
            f"for resource {self.resource_id}"
        )
    
    def __lt__(self, other: "SimulationEvent") -> bool:
        """Compare events for priority queue ordering."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority


class EventBus:
    """Event bus for managing simulation events."""
    
    def __init__(self) -> None:
        self.events: list[SimulationEvent] = []
        self.event_handlers: Dict[EventType, list] = {}
        
    def subscribe(self, event_type: EventType, handler) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.value}")
    
    def publish(self, event: SimulationEvent) -> None:
        """Publish an event to all subscribers."""
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error handling event {event.event_type.value}: {e}")
    
    def schedule_event(self, event: SimulationEvent) -> None:
        """Schedule an event for future execution."""
        self.events.append(event)
        self.events.sort()  # Keep events sorted by timestamp
        logger.debug(f"Event scheduled: {event.event_type.value} at {event.timestamp}")
    
    def get_next_event(self) -> Optional[SimulationEvent]:
        """Get the next event to process."""
        if self.events:
            return self.events.pop(0)
        return None
    
    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
        logger.debug("Event bus cleared")
