"""Cloud Scheduling and Autoscaling Simulator."""

__version__ = "0.1.0"
__author__ = "Cloud Project Team"

from loguru import logger

# Configure loguru for the entire package
logger.add(
    "logs/cloud_scheduler_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)

logger.info("Cloud Scheduler Simulator initialized")
