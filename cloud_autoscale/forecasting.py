"""Forecasting helper module for ML-driven autoscaling.

This module provides a lightweight wrapper around trained ML models
for use inside the simulator by proactive autoscalers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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
                       - run_dir / "modeling" / "model.pkl"
                       - run_dir / "modeling" / "scaler.pkl"
                       - run_dir / "modeling" / "feature_cols.json"
        
        Raises:
            FileNotFoundError: If required model artifacts are missing
        """
        self.run_dir = Path(run_dir)
        model_dir = self.run_dir / "modeling"
        
        # Validate model directory exists
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Modeling directory not found: {model_dir}\n"
                f"Please run the modeling notebook first to train and save models."
            )
        
        # Load model
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please run the modeling notebook to save the trained model."
            )
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {scaler_path}\n"
                f"Please run the modeling notebook to save the scaler."
            )
        self.scaler = joblib.load(scaler_path)
        
        # Load feature columns
        feature_cols_path = model_dir / "feature_cols.json"
        if not feature_cols_path.exists():
            raise FileNotFoundError(
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
        df["sin_day"] = np.sin(2 * np.pi * df["step"] / 288.0)
        df["cos_day"] = np.cos(2 * np.pi * df["step"] / 288.0)
        
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
        One-step-ahead forecast of cpu_demand.
        
        Args:
            history: DataFrame with recent time steps (columns: step, time, cpu_demand, 
                     mem_demand, new_instances_norm, etc.)
        
        Returns:
            Predicted CPU demand for next time step
        """
        X = self._prepare_single_row(history)
        y_hat = float(self.model.predict(X.reshape(1, -1))[0])
        return y_hat
    
    def recursive_forecast(self, history: pd.DataFrame, steps: int = 6) -> List[float]:
        """
        Simplified recursive multi-step forecast (t+1 ... t+steps).
        
        For each step, we recompute features by appending the last prediction
        to the history as a new cpu_demand row, keeping other columns constant.
        This keeps implementation simple and avoids heavy changes to the simulator.
        
        Args:
            history: DataFrame with recent time steps
            steps: Number of steps to forecast ahead (default: 6)
        
        Returns:
            List of predicted CPU demands
        """
        preds: List[float] = []
        hist = history.copy()
        
        for _ in range(steps):
            y_hat = self.predict_next(hist)
            preds.append(y_hat)
            
            # Append a "fake" next row using predicted cpu_demand,
            # and carrying forward other series (mem_demand, events) from last row
            last = hist.iloc[-1].copy()
            new_row = last.copy()
            new_row["step"] = last["step"] + 1
            
            # Calculate time delta from last two rows
            if len(hist) >= 2:
                time_delta = hist["time"].iloc[-1] - hist["time"].iloc[-2]
            else:
                time_delta = 5.0  # Default to 5 minutes
            
            new_row["time"] = last["time"] + time_delta
            new_row["cpu_demand"] = y_hat
            # Leave mem_demand/new_instances_norm as last known values for simplicity
            
            hist = pd.concat([hist, new_row.to_frame().T], ignore_index=True)
        
        return preds
    
    def multi_horizon(self, history: pd.DataFrame) -> Dict[str, float]:
        """
        Convenience helper for autoscaler: returns t+1, t+3, t+6
        plus the full sequence.
        
        Args:
            history: DataFrame with recent time steps
        
        Returns:
            Dictionary with keys: 't+1', 't+3', 't+6', 'full'
        """
        seq = self.recursive_forecast(history, steps=6)
        out = {
            "t+1": seq[0],
            "t+3": seq[2],
            "t+6": seq[5],
            "full": seq,
        }
        return out

