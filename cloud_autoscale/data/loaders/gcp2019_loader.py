"""GCP 2019 trace data loader - Production version.

This loader reads pre-processed Parquet files from data/processed/.
It does NOT read raw JSONL files or generate synthetic fallbacks.
All data must be prepared beforehand using the data processing pipeline.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class GCP2019Loader:
    """Load pre-processed GCP 2019 trace data from Parquet files."""
    
    def __init__(
        self,
        processed_dir: str | Path,
        step_minutes: int,
        duration_minutes: Optional[int] = None
    ):
        """
        Initialize GCP 2019 data loader.
        
        This loader expects processed Parquet files in the specified directory:
        - cluster_level.parquet (required)
        - machine_level.parquet (optional, not used by this loader)
        
        Args:
            processed_dir: Path to directory containing processed Parquet files
            step_minutes: Time bucket size in minutes
            duration_minutes: Optional duration limit in minutes (None = use all data)
        
        Raises:
            ValueError: If processed_dir doesn't exist or required files are missing
        """
        self.processed_dir = Path(processed_dir)
        self.step_minutes = step_minutes
        self.duration_minutes = duration_minutes
        
        # Validate directory exists
        if not self.processed_dir.exists():
            raise ValueError(
                f"Processed data directory does not exist: {self.processed_dir}\n"
                f"Please run the data processing pipeline first."
            )
        
        # Validate required file exists
        self.cluster_file = self.processed_dir / 'cluster_level.parquet'
        if not self.cluster_file.exists():
            raise ValueError(
                f"Required file not found: {self.cluster_file}\n"
                f"Please ensure the data processing pipeline has been run."
            )
    
    def load(self) -> pd.DataFrame:
        """
        Load and prepare cluster-level demand data.
        
        Returns:
            DataFrame with columns: time, cpu_demand, mem_demand, new_instances
            - time: Time in minutes (float)
            - cpu_demand: CPU demand in cores (float)
            - mem_demand: Memory demand in GB (float)
            - new_instances: Number of new instances created (int)
        
        Raises:
            ValueError: If data cannot be loaded or is malformed
        """
        # Load using Polars if available (faster), otherwise Pandas
        if HAS_POLARS:
            df = self._load_with_polars()
        else:
            df = self._load_with_pandas()
        
        return df
    
    def _load_with_polars(self) -> pd.DataFrame:
        """Load data using Polars (faster for large files)."""
        # Read parquet file
        df = pl.read_parquet(self.cluster_file)
        
        # Validate required columns exist
        required_cols = ['bucket_s', 'cpu_demand', 'mem_demand', 'new_instances_cluster']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {self.cluster_file}: {missing_cols}\n"
                f"Available columns: {df.columns}"
            )
        
        # Convert bucket_s (seconds) to time (minutes)
        df = df.with_columns([
            (pl.col('bucket_s') / 60.0).alias('time')
        ])
        
        # Select and rename columns
        df = df.select([
            'time',
            'cpu_demand',
            'mem_demand',
            pl.col('new_instances_cluster').alias('new_instances')
        ])
        
        # Sort by time
        df = df.sort('time')
        
        # Truncate to duration if specified
        if self.duration_minutes is not None:
            df = df.filter(pl.col('time') < self.duration_minutes)
        
        # Convert to pandas for compatibility with simulator
        return df.to_pandas()
    
    def _load_with_pandas(self) -> pd.DataFrame:
        """Load data using Pandas (fallback if Polars not available)."""
        # Read parquet file
        df = pd.read_parquet(self.cluster_file)
        
        # Validate required columns exist
        required_cols = ['bucket_s', 'cpu_demand', 'mem_demand', 'new_instances_cluster']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {self.cluster_file}: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Convert bucket_s (seconds) to time (minutes)
        df['time'] = df['bucket_s'] / 60.0
        
        # Select and rename columns
        df = df[['time', 'cpu_demand', 'mem_demand', 'new_instances_cluster']].copy()
        df.rename(columns={'new_instances_cluster': 'new_instances'}, inplace=True)
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        # Truncate to duration if specified
        if self.duration_minutes is not None:
            df = df[df['time'] < self.duration_minutes].reset_index(drop=True)
        
        return df
