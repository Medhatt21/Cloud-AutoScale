"""GCP 2019 trace data loader - Production version.

This loader reads pre-processed Parquet files from data/processed/.
It does NOT read raw JSONL files or generate synthetic fallbacks.
All data must be prepared beforehand using the data processing pipeline.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

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
        Load and prepare cluster-level demand data with time continuity validation.
        
        Returns simulation-ready DataFrame with minimal normalization only.
        NO ML feature engineering (lags, rolling, etc.) - those belong in modeling notebooks.
        
        Returns:
            DataFrame with columns:
            - step: int, sequential with no gaps (0..N-1)
            - time: float, minutes since start (step * step_minutes)
            - cpu_demand: float, CPU demand
            - mem_demand: float, memory demand
            - new_instances: float, number of new instances (raw)
            - new_instances_norm: float, log1p(new_instances) for stable signal
            - machines_reporting: float, number of machines reporting (optional context)
        
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
        required_cols = ['bucket_index', 'cpu_demand', 'mem_demand']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {self.cluster_file}: {missing_cols}\n"
                f"Available columns: {df.columns}"
            )
        
        # Sort by bucket_index
        df = df.sort('bucket_index')
        
        # Create step and time columns
        df = df.with_columns([
            pl.col('bucket_index').cast(pl.Int64).alias('step'),
            (pl.col('bucket_index') * self.step_minutes).alias('time')
        ])
        
        # Map columns
        df = df.with_columns([
            pl.col('cpu_demand').cast(pl.Float64),
            pl.col('mem_demand').cast(pl.Float64),
            pl.col('new_instances_cluster').fill_null(0).cast(pl.Float64).alias('new_instances') if 'new_instances_cluster' in df.columns else pl.lit(0.0).alias('new_instances'),
            pl.col('machines').fill_null(0).cast(pl.Float64).alias('machines_reporting') if 'machines' in df.columns else pl.lit(float('nan')).alias('machines_reporting')
        ])
        
        # Convert to pandas for gap filling
        df_pd = df.to_pandas()
        
        # Detect and fill gaps (ensure time continuity)
        df_pd = self._fill_gaps(df_pd)
        
        # Minimal normalization for simulation stability only
        df_pd['new_instances_norm'] = np.log1p(df_pd['new_instances'])
        
        # Select final columns (NO ML features)
        final_cols = [
            'step', 'time', 'cpu_demand', 'mem_demand',
            'new_instances', 'new_instances_norm', 'machines_reporting'
        ]
        df_pd = df_pd[final_cols].copy()
        
        # Truncate to duration if specified
        if self.duration_minutes is not None:
            max_steps = self.duration_minutes // self.step_minutes
            df_pd = df_pd[df_pd['step'] < max_steps].reset_index(drop=True)
        
        return df_pd
    
    def _load_with_pandas(self) -> pd.DataFrame:
        """Load data using Pandas (fallback if Polars not available)."""
        # Read parquet file
        df = pd.read_parquet(self.cluster_file)
        
        # Validate required columns exist
        required_cols = ['bucket_index', 'cpu_demand', 'mem_demand']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {self.cluster_file}: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Sort by bucket_index
        df = df.sort_values('bucket_index').reset_index(drop=True)
        
        # Create step and time columns
        df['step'] = df['bucket_index'].astype(int)
        df['time'] = df['step'] * self.step_minutes
        
        # Map columns
        df['cpu_demand'] = df['cpu_demand'].astype(float)
        df['mem_demand'] = df['mem_demand'].astype(float)
        df['new_instances'] = df.get('new_instances_cluster', 0).fillna(0).astype(float)
        df['machines_reporting'] = df.get('machines', np.nan).astype(float)
        
        # Detect and fill gaps (ensure time continuity)
        df = self._fill_gaps(df)
        
        # Minimal normalization for simulation stability only
        df['new_instances_norm'] = np.log1p(df['new_instances'])
        
        # Select final columns (NO ML features like lags, rolling, etc.)
        final_cols = [
            'step', 'time', 'cpu_demand', 'mem_demand', 
            'new_instances', 'new_instances_norm', 'machines_reporting'
        ]
        df = df[final_cols].copy()
        
        # Truncate to duration if specified
        if self.duration_minutes is not None:
            max_steps = self.duration_minutes // self.step_minutes
            df = df[df['step'] < max_steps].reset_index(drop=True)
        
        return df
    
    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and fill gaps in step sequence.
        
        Missing steps are filled with zeros for demand and events.
        
        Args:
            df: DataFrame with 'step' column
        
        Returns:
            DataFrame with no gaps in step sequence
        """
        if len(df) == 0:
            return df
        
        min_step = int(df['step'].min())
        max_step = int(df['step'].max())
        expected_steps = pd.DataFrame({'step': range(min_step, max_step + 1)})
        
        # Merge to identify missing steps
        df_filled = expected_steps.merge(df, on='step', how='left')
        
        # Fill missing values
        df_filled['time'] = df_filled['step'] * self.step_minutes
        df_filled['cpu_demand'] = df_filled['cpu_demand'].fillna(0.0)
        df_filled['mem_demand'] = df_filled['mem_demand'].fillna(0.0)
        df_filled['new_instances'] = df_filled['new_instances'].fillna(0.0)
        df_filled['machines_reporting'] = df_filled['machines_reporting'].ffill().fillna(0.0)
        
        return df_filled
