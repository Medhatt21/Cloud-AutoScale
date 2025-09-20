"""Workload forecasting models using ML/DL techniques."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loguru import logger

# Optional imports for specific models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from ..core.simulator import SimulationMetrics


class WorkloadForecaster(ABC):
    """Abstract base class for workload forecasting models."""
    
    def __init__(self, forecast_horizon: int = 10):
        self.forecast_horizon = forecast_horizon
        self.is_trained = False
        self.logger = logger.bind(component=self.__class__.__name__)
    
    @abstractmethod
    def fit(self, metrics_history: List[SimulationMetrics]) -> None:
        """Train the forecasting model."""
        pass
    
    @abstractmethod
    def predict(self, steps: int = None) -> np.ndarray:
        """Make predictions for future time steps."""
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate forecasting performance."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
        }


class LSTMForecaster(WorkloadForecaster):
    """LSTM-based workload forecasting model."""
    
    def __init__(
        self,
        forecast_horizon: int = 10,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        super().__init__(forecast_horizon)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"LSTM Forecaster initialized with horizon {forecast_horizon}, "
                        f"sequence length {sequence_length}")
    
    def fit(self, metrics_history: List[SimulationMetrics]) -> None:
        """Train the LSTM model."""
        if len(metrics_history) < self.sequence_length + self.forecast_horizon:
            self.logger.warning("Insufficient data for training LSTM model")
            return
        
        # Prepare data
        features = self._extract_features(metrics_history)
        X, y = self._create_sequences(features)
        
        if len(X) == 0:
            self.logger.warning("No sequences created for training")
            return
        
        # Create model
        input_size = X.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout
        ).to(self.device)
        
        # Train model
        self._train_model(X, y)
        self.is_trained = True
        
        self.logger.info("LSTM model training completed")
    
    def predict(self, steps: int = None) -> np.ndarray:
        """Make predictions using the trained LSTM model."""
        if not self.is_trained or self.model is None:
            self.logger.warning("Model not trained, returning zeros")
            return np.zeros(steps or self.forecast_horizon)
        
        steps = steps or self.forecast_horizon
        
        # Use the last sequence for prediction
        if not hasattr(self, 'last_sequence'):
            self.logger.warning("No last sequence available for prediction")
            return np.zeros(steps)
        
        predictions = []
        current_sequence = self.last_sequence.clone()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_sequence.unsqueeze(0))
                predictions.append(pred.item())
                
                # Update sequence (shift and append prediction)
                current_sequence = torch.cat([
                    current_sequence[1:],
                    pred.unsqueeze(0)
                ])
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def _extract_features(self, metrics_history: List[SimulationMetrics]) -> np.ndarray:
        """Extract features from metrics history."""
        features = []
        
        for metrics in metrics_history:
            feature_vector = [
                metrics.queued_workloads,
                metrics.running_workloads,
                metrics.avg_cpu_utilization,
                metrics.avg_memory_utilization,
                metrics.active_hosts,
                metrics.sla_violations,
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        return features
    
    def _create_sequences(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences for training."""
        X, y = [], []
        
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            y.append(features[i + self.sequence_length, 0])  # Predict queued workloads
        
        X = np.array(X)
        y = np.array(y)
        
        # Store last sequence for prediction
        if len(X) > 0:
            self.last_sequence = torch.FloatTensor(X[-1]).to(self.device)
        
        return X, y
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """Train the LSTM model."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


class LSTMModel(nn.Module):
    """LSTM neural network model."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class ARIMAForecaster(WorkloadForecaster):
    """ARIMA-based workload forecasting model."""
    
    def __init__(self, forecast_horizon: int = 10, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__(forecast_horizon)
        self.order = order
        self.model = None
        self.fitted_model = None
        
        if not STATSMODELS_AVAILABLE:
            self.logger.error("statsmodels not available, ARIMA forecasting disabled")
    
    def fit(self, metrics_history: List[SimulationMetrics]) -> None:
        """Train the ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            self.logger.error("Cannot fit ARIMA model: statsmodels not available")
            return
        
        if len(metrics_history) < 20:
            self.logger.warning("Insufficient data for ARIMA model")
            return
        
        # Extract time series data
        ts_data = [metrics.queued_workloads for metrics in metrics_history]
        
        try:
            # Fit ARIMA model
            self.model = ARIMA(ts_data, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_trained = True
            
            self.logger.info(f"ARIMA{self.order} model fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to fit ARIMA model: {e}")
    
    def predict(self, steps: int = None) -> np.ndarray:
        """Make predictions using the trained ARIMA model."""
        if not self.is_trained or self.fitted_model is None:
            self.logger.warning("ARIMA model not trained, returning zeros")
            return np.zeros(steps or self.forecast_horizon)
        
        steps = steps or self.forecast_horizon
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return np.maximum(forecast, 0)  # Ensure non-negative predictions
            
        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {e}")
            return np.zeros(steps)


class ProphetForecaster(WorkloadForecaster):
    """Prophet-based workload forecasting model."""
    
    def __init__(self, forecast_horizon: int = 10):
        super().__init__(forecast_horizon)
        self.model = None
        
        if not PROPHET_AVAILABLE:
            self.logger.error("Prophet not available, forecasting disabled")
    
    def fit(self, metrics_history: List[SimulationMetrics]) -> None:
        """Train the Prophet model."""
        if not PROPHET_AVAILABLE:
            self.logger.error("Cannot fit Prophet model: Prophet not available")
            return
        
        if len(metrics_history) < 10:
            self.logger.warning("Insufficient data for Prophet model")
            return
        
        # Prepare data for Prophet
        data = []
        for i, metrics in enumerate(metrics_history):
            data.append({
                'ds': pd.Timestamp.now() + pd.Timedelta(seconds=i * 60),  # Assume 1-minute intervals
                'y': metrics.queued_workloads
            })
        
        df = pd.DataFrame(data)
        
        try:
            # Create and fit Prophet model
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            self.model.fit(df)
            self.is_trained = True
            
            self.logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to fit Prophet model: {e}")
    
    def predict(self, steps: int = None) -> np.ndarray:
        """Make predictions using the trained Prophet model."""
        if not self.is_trained or self.model is None:
            self.logger.warning("Prophet model not trained, returning zeros")
            return np.zeros(steps or self.forecast_horizon)
        
        steps = steps or self.forecast_horizon
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps, freq='T')  # 1-minute frequency
            
            # Make predictions
            forecast = self.model.predict(future)
            predictions = forecast['yhat'].tail(steps).values
            
            return np.maximum(predictions, 0)  # Ensure non-negative predictions
            
        except Exception as e:
            self.logger.error(f"Prophet prediction failed: {e}")
            return np.zeros(steps)


class EnsembleForecaster(WorkloadForecaster):
    """Ensemble forecaster combining multiple models."""
    
    def __init__(
        self,
        forecast_horizon: int = 10,
        models: Optional[List[WorkloadForecaster]] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__(forecast_horizon)
        
        if models is None:
            models = [
                LSTMForecaster(forecast_horizon),
                ARIMAForecaster(forecast_horizon),
                ProphetForecaster(forecast_horizon),
            ]
        
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, metrics_history: List[SimulationMetrics]) -> None:
        """Train all ensemble models."""
        self.logger.info(f"Training ensemble of {len(self.models)} models")
        
        for i, model in enumerate(self.models):
            try:
                model.fit(metrics_history)
                if model.is_trained:
                    self.logger.info(f"Model {i+1} ({model.__class__.__name__}) trained successfully")
                else:
                    self.logger.warning(f"Model {i+1} ({model.__class__.__name__}) failed to train")
            except Exception as e:
                self.logger.error(f"Error training model {i+1}: {e}")
        
        # Check if at least one model is trained
        self.is_trained = any(model.is_trained for model in self.models)
        
        if self.is_trained:
            self.logger.info("Ensemble model training completed")
        else:
            self.logger.error("No models in ensemble were successfully trained")
    
    def predict(self, steps: int = None) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            self.logger.warning("Ensemble not trained, returning zeros")
            return np.zeros(steps or self.forecast_horizon)
        
        steps = steps or self.forecast_horizon
        predictions = []
        total_weight = 0
        
        for model, weight in zip(self.models, self.weights):
            if model.is_trained:
                try:
                    pred = model.predict(steps)
                    predictions.append(pred * weight)
                    total_weight += weight
                except Exception as e:
                    self.logger.error(f"Error getting prediction from {model.__class__.__name__}: {e}")
        
        if not predictions:
            self.logger.warning("No valid predictions from ensemble models")
            return np.zeros(steps)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        
        return np.maximum(ensemble_pred, 0)  # Ensure non-negative predictions
