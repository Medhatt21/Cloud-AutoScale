"""Data loader utilities (alias for backward compatibility)."""

# Re-export from data module
from ..data.loaders import CloudTraceLoader, GoogleTraceLoader, AzureTraceLoader, SyntheticTraceLoader
from ..data.preprocessors import DataPreprocessor

# Aliases for backward compatibility
CloudTraceLoader = CloudTraceLoader
DataPreprocessor = DataPreprocessor

__all__ = [
    "CloudTraceLoader",
    "GoogleTraceLoader", 
    "AzureTraceLoader",
    "SyntheticTraceLoader",
    "DataPreprocessor",
]
