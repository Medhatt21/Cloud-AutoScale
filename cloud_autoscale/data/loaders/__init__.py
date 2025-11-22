"""Data loaders for Cloud AutoScale."""

from .synthetic_loader import SyntheticLoader
from .gcp2019_loader import GCP2019Loader

__all__ = ['SyntheticLoader', 'GCP2019Loader']

