from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from cloud_autoscale.rl.agent import DQNAgent

class RLAutoscaler:
    """
    RL-based autoscaler using trained DQN agent.
    """
    
    def __init__(
        self, 
        autoscaler_config: Dict[str, Any],
        step_minutes: int,
        min_machines: int,
        max_machines: int
    ):
        self.config = autoscaler_config
        self.step_minutes = step_minutes
        self.min_machines = min_machines
        self.max_machines = max_machines
        
        # Action mapping
        self.action_mapping = autoscaler_config.get('action_mapping', [-10, -5, 0, 5, 10])
        self.action_dim = len(self.action_mapping)
        
        # State dim = 9 (as defined in Env)
        self.state_dim = 9
        
        # Load agent
        model_rl_dir = autoscaler_config.get('model_rl_dir', 'latest')
        self.agent = self._load_agent(model_rl_dir)
        
        # History buffer for lag features
        self.util_history = [0.0, 0.0, 0.0]

    def _load_agent(self, model_dir_str: str) -> DQNAgent:
        """Load the trained agent."""
        # Resolve path
        if model_dir_str == 'latest':
            # Try to find latest in results
            base_dir = Path("results")
            run_dirs = sorted(base_dir.glob('run_rl_*')) # distinct RL runs? or just run_*?
            # User example: output results/run_rl_2025xxxx
            # Logic: look for any run with model_rl/agent_policy.pth
            candidates = []
            if base_dir.exists():
                for d in base_dir.iterdir():
                    if (d / "model_rl" / "agent_policy.pth").exists():
                        candidates.append(d)
            if not candidates:
                 raise FileNotFoundError("No trained RL models found in results/")
            # Sort by name (timestamp)
            candidates.sort(key=lambda p: p.name)
            model_path = candidates[-1] / "model_rl" / "agent_policy.pth"
            print(f"   Auto-detected RL model: {model_path}")
        else:
            model_path = Path(model_dir_str)
            if model_path.is_dir():
                model_path = model_path / "model_rl" / "agent_policy.pth"
            
            if not model_path.exists():
                # Try direct path
                if Path(model_dir_str).exists() and Path(model_dir_str).is_file():
                    model_path = Path(model_dir_str)
                else:
                    raise FileNotFoundError(f"RL model not found at: {model_path}")
        
        agent = DQNAgent(self.state_dim, self.action_dim, device="cpu") # Inference on CPU usually
        agent.load(str(model_path))
        agent.policy_net.eval()
        return agent

    def decide(
        self,
        current_capacity: float,
        current_machines: int,
        demand: float,
        utilization: float,
        time: float,
        history_df: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Make scaling decision.
        """
        # Update history
        # We need to maintain a running history of utilization
        # The simulator calls decide() every step.
        # Ideally we should use history_df if provided, but if not, we maintain local state.
        # But local state might be lost if simulator recreates autoscaler (it doesn't).
        
        # Update util history
        self.util_history.pop(0)
        self.util_history.append(utilization if utilization != float('inf') else 1.0)
        
        # Construct State
        # [util, demand, machines, lag1, lag2, lag3, f1, f3, f6]
        # Normalize
        
        # Naive forecast: f1=f3=f6=demand
        # (Consistent with Env implementation if we don't have better forecasts)
        
        state = np.array([
            utilization if utilization != float('inf') else 0.0,
            demand / 1000.0,
            current_machines / 1000.0,
            self.util_history[2],
            self.util_history[1],
            self.util_history[0],
            demand / 1000.0, # Naive
            demand / 1000.0, # Naive
            demand / 1000.0  # Naive
        ], dtype=np.float32)
        
        # Get action
        action_idx = self.agent.act(state, training=False)
        scale_delta = self.action_mapping[action_idx]
        
        return scale_delta

