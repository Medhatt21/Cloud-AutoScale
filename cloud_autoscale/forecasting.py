"""Forecasting helper module for ML-driven autoscaling.

This module provides a lightweight wrapper around trained ML models
for use inside the simulator by proactive autoscalers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
import pandas as pd
import joblib


class ForecastingModel:
    """
    Lightweight wrapper around the trained ML model + scaler + feature metadata.
    
    This is used *inside* the simulator by proactive autoscalers.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize forecasting model from saved artifacts.
        
        Args:
            run_dir: Path to a specific simulation run directory, e.g. results/run_20251123_175247
                     The modeling notebook should have saved:
                       - run_dir / "modeling" / "model_t1.pkl"
                       - run_dir / "modeling" / "model_t3.pkl"
                       - run_dir / "modeling" / "model_t6.pkl"
                       - run_dir / "modeling" / "scaler.pkl"
                       - run_dir / "modeling" / "feature_cols.json"
        
        Raises:
            RuntimeError: If required model artifacts are missing
        """
        self.run_dir = Path(run_dir)
        model_dir = self.run_dir / "modeling"
        
        # Validate model directory exists
        if not model_dir.exists():
            raise RuntimeError(
                f"Modeling directory not found: {model_dir}\n"
                f"Please run the modeling notebook first to train and save models."
            )
        
        # Load three separate direct multi-horizon models (mandatory)
        model_t1_path = model_dir / "model_t1.pkl"
        if not model_t1_path.exists():
            raise RuntimeError(
                f"Model file not found: {model_t1_path}\n"
                f"Direct multi-horizon model for t+1 is required."
            )
        self.model_t1 = joblib.load(model_t1_path)
        
        model_t3_path = model_dir / "model_t3.pkl"
        if not model_t3_path.exists():
            raise RuntimeError(
                f"Model file not found: {model_t3_path}\n"
                f"Direct multi-horizon model for t+3 is required."
            )
        self.model_t3 = joblib.load(model_t3_path)
        
        model_t6_path = model_dir / "model_t6.pkl"
        if not model_t6_path.exists():
            raise RuntimeError(
                f"Model file not found: {model_t6_path}\n"
                f"Direct multi-horizon model for t+6 is required."
            )
        self.model_t6 = joblib.load(model_t6_path)
        
        # Load scaler
        scaler_path = model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise RuntimeError(
                f"Scaler file not found: {scaler_path}\n"
                f"Please run the modeling notebook to save the scaler."
            )
        self.scaler = joblib.load(scaler_path)
        
        # Load feature columns
        feature_cols_path = model_dir / "feature_cols.json"
        if not feature_cols_path.exists():
            raise RuntimeError(
                f"Feature columns file not found: {feature_cols_path}\n"
                f"Please run the modeling notebook to save feature metadata."
            )
        with open(feature_cols_path, "r") as f:
            self.feature_cols: List[str] = json.load(f)
    
    # ---------- feature engineering ----------
    
    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rebuild the same engineered features as in modeling_pipeline.ipynb.
        
        Features:
        - lags: 1, 2, 3, 6, 12
        - rolling means: 3, 6, 12 (using shift(1) to avoid leakage)
        - diff: cpu_diff1, mem_diff1
        - cyclical: sin_day, cos_day (step is the canonical bucket index)
        
        Args:
            df: DataFrame with columns: step, time, cpu_demand, mem_demand, new_instances_norm
        
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Lags
        for lag in [1, 2, 3, 6, 12]:
            df[f"cpu_lag{lag}"] = df["cpu_demand"].shift(lag)
            df[f"mem_lag{lag}"] = df["mem_demand"].shift(lag)
            df[f"evt_lag{lag}"] = df["new_instances_norm"].shift(lag)
        
        # Rolling means with leakage-safe shift
        for w in [3, 6, 12]:
            df[f"cpu_ma{w}"] = df["cpu_demand"].shift(1).rolling(window=w, min_periods=1).mean()
            df[f"mem_ma{w}"] = df["mem_demand"].shift(1).rolling(window=w, min_periods=1).mean()
            df[f"evt_ma{w}"] = df["new_instances_norm"].shift(1).rolling(window=w, min_periods=1).mean()
        
        # Differencing
        df["cpu_diff1"] = df["cpu_demand"].diff()
        df["mem_diff1"] = df["mem_demand"].diff()
        
        # Cyclical (288 steps per day for 5-minute intervals)
        # Ensure step is numeric to avoid type issues
        step_values = pd.to_numeric(df["step"], errors='coerce').fillna(0).astype(float)
        df["sin_day"] = np.sin(2 * np.pi * step_values / 288.0)
        df["cos_day"] = np.cos(2 * np.pi * step_values / 288.0)
        
        return df
    
    def _prepare_single_row(self, history: pd.DataFrame) -> np.ndarray:
        """
        Take a history DataFrame (at least ~12 rows), compute features, and return the
        last row as a scaled feature vector ready for inference.
        
        Args:
            history: DataFrame with recent time steps (needs at least 12 rows for lag features)
        
        Returns:
            Scaled feature vector (1D numpy array)
        """
        fe = self._add_features(history)
        fe = fe.dropna().reset_index(drop=True)
        
        if len(fe) == 0:
            raise ValueError(
                "Not enough history to compute features. "
                "Need at least 12 time steps with valid data."
            )
        
        latest = fe.iloc[-1]
        X = latest[self.feature_cols].to_frame().T
        X_scaled = self.scaler.transform(X)
        return X_scaled[0]  # Return 1D array
    
    # ---------- forecasting APIs ----------
    
    def predict_next(self, history: pd.DataFrame) -> float:
        """
        One-step-ahead forecast of cpu_demand using direct t+1 model.
        
        Args:
            history: DataFrame with recent time steps (columns: step, time, cpu_demand, 
                     mem_demand, new_instances_norm, etc.)
        
        Returns:
            Predicted CPU demand for next time step
        """
        X = self._prepare_single_row(history)
        y_hat = float(self.model_t1.predict(X.reshape(1, -1))[0])
        
        # Fail-fast validation
        if np.isnan(y_hat):
            raise RuntimeError("Model t+1 prediction returned NaN")
        
        return y_hat
    
    
    def multi_horizon(self, history: pd.DataFrame) -> Dict[str, Any]:
        """
        Direct multi-horizon forecast using separate models for t+1, t+3, t+6.
        
        Args:
            history: DataFrame with recent time steps
        
        Returns:
            Dictionary with keys: 't+1', 't+3', 't+6', 'full'
            'full' contains [t+1, None, t+3, None, None, t+6] for compatibility
        
        Raises:
            RuntimeError: If any model prediction fails or returns NaN
        """
        X = self._prepare_single_row(history)
        
        # Predict using three separate direct models
        t1 = float(self.model_t1.predict(X.reshape(1, -1))[0])
        t3 = float(self.model_t3.predict(X.reshape(1, -1))[0])
        t6 = float(self.model_t6.predict(X.reshape(1, -1))[0])
        
        # Fail-fast validation
        if np.isnan(t1):
            raise RuntimeError("Model t+1 prediction returned NaN")
        if np.isnan(t3):
            raise RuntimeError("Model t+3 prediction returned NaN")
        if np.isnan(t6):
            raise RuntimeError("Model t+6 prediction returned NaN")
        
        out = {
            "t+1": t1,
            "t+3": t3,
            "t+6": t6,
            "full": [t1, None, t3, None, None, t6],
        }
        return out

