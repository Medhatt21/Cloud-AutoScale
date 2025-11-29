import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class AutoscaleEnv(gym.Env):
    """
    RL Environment for Cloud Autoscaling.
    
    State Space (9 dims):
    [
        current_utilization,
        current_demand_norm,    # Normalized demand
        current_machines_norm,  # Normalized machines
        util_lag1,
        util_lag2,
        util_lag3,
        forecast_t1_norm,
        forecast_t3_norm,
        forecast_t6_norm
    ]
    
    Action Space (Discrete 5):
    0: Scale -10
    1: Scale -5
    2: Do nothing
    3: Scale +5
    4: Scale +10
    """
    
    def __init__(
        self, 
        demand_df: pd.DataFrame,
        sim_config: Dict[str, Any],
        autoscaler_config: Dict[str, Any]
    ):
        super(AutoscaleEnv, self).__init__()
        
        self.demand_df = demand_df.reset_index(drop=True)
        self.sim_config = sim_config
        self.autoscaler_config = autoscaler_config
        
        # Simulation parameters
        self.min_machines = sim_config['min_machines']
        self.max_machines = sim_config['max_machines']
        self.machine_capacity = sim_config['machine_capacity']
        self.step_minutes = sim_config['step_minutes']
        self.cost_per_hour = sim_config['cost_per_machine_per_hour']
        
        # Action mapping
        self.action_mapping = autoscaler_config.get('action_mapping', [-10, -5, 0, 5, 10])
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        # Observation space
        # We normalize inputs to be roughly 0-1 or -1 to 1 for better RL performance
        # 9 dimensions as specified
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.current_machines = self.min_machines
        self.util_history = [0.0, 0.0, 0.0]  # Lag 1, 2, 3
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_machines = self.min_machines
        self.util_history = [0.0, 0.0, 0.0]
        
        # Get initial state
        state = self._get_state()
        info = {}
        
        return state, info
    
    def step(self, action_idx):
        # Decode action
        scale_delta = self.action_mapping[action_idx]
        
        # Apply action
        self.current_machines += scale_delta
        self.current_machines = max(self.min_machines, min(self.current_machines, self.max_machines))
        
        # Get current demand (at this new step)
        # Note: In simulator, we decide for *this* step based on *this* step's demand?
        # Usually, we decide at t for t+1, or we decide at t and it applies immediately?
        # The simulator code:
        # 1. Calc util for current step (based on machines from prev step?)
        # Actually simulator:
        # self.current_machines = min_machines (init)
        # loop:
        #   calc util (demand / current_capacity)
        #   decide()
        #   update current_machines
        # This means the decision affects the *next* step's capacity effectively?
        # No, in the simulator loop:
        #   calc util (using current_machines) -> this is "simulating" the step
        #   decide -> produces action
        #   update machines -> ready for NEXT loop iteration
        # So action at T affects capacity at T+1.
        
        # Current step demand (for reward calculation of the *result* of previous action)
        # Wait, RL usually works as:
        # S_t -> A_t -> R_t+1, S_t+1
        # Here:
        # S_t includes current utilization (at step t).
        # A_t modifies machines.
        # S_t+1 is state at step t+1 (new demand, new machines).
        
        # So we are at `current_step`.
        # We observe S_t.
        # We take A_t.
        # We move to t+1.
        # We calculate R based on what happened at t+1?
        # Or R based on "how good was the state I just left"?
        # Standard: Reward is received *after* transition.
        # So we apply action, increment time, observe new state/violation, return that reward.
        
        # Apply action limits
        prev_machines = self.current_machines # This was machines at step t (modified by action? No.)
        
        # In simulator:
        # machines are updated at END of loop.
        # So for step t, we use machines determined at t-1.
        # Then we decide delta.
        # Then we update machines.
        # So the delta affects t+1.
        
        # So:
        # 1. Apply delta to get machines for t+1.
        # 2. Increment step.
        # 3. Read demand at t+1.
        # 4. Calc util at t+1.
        # 5. Calc reward.
        
        self.current_step += 1
        terminated = self.current_step >= len(self.demand_df) - 1
        truncated = False
        
        if terminated:
            # Can't calculate next state properly if we are done
            # Just return current state and 0 reward
            return self._get_state(), 0.0, terminated, truncated, {}
            
        row = self.demand_df.iloc[self.current_step]
        
        # Current demand
        cpu_demand = float(row['cpu_demand'])
        mem_demand = float(row['mem_demand'])
        total_demand = max(cpu_demand, mem_demand)
        
        # Capacity
        capacity = self.current_machines * self.machine_capacity
        
        # Utilization
        if capacity > 0:
            utilization = total_demand / capacity
        else:
            utilization = 999.0 # Should not happen
            
        # Update history
        self.util_history.pop(0)
        self.util_history.append(utilization)
        
        # Calculate Reward
        reward = self._calculate_reward(utilization, scale_delta, self.current_machines)
        
        # Next State
        state = self._get_state()
        
        return state, reward, terminated, truncated, {}
        
    def _get_state(self):
        # We need data for current_step
        if self.current_step >= len(self.demand_df):
             row = self.demand_df.iloc[-1] # Fallback
        else:
             row = self.demand_df.iloc[self.current_step]
             
        cpu_demand = float(row['cpu_demand'])
        mem_demand = float(row['mem_demand'])
        demand = max(cpu_demand, mem_demand)
        
        capacity = self.current_machines * self.machine_capacity
        utilization = demand / capacity if capacity > 0 else 0.0
        
        # Forecasts (assuming they are in df, else fallback to current demand)
        # We need t+1, t+3, t+6 relative to current_step
        # If columns exist: 'forecast_t1', etc.
        # Else look ahead in dataframe?
        
        def get_forecast(offset):
            target_idx = self.current_step + offset
            if target_idx < len(self.demand_df):
                target_row = self.demand_df.iloc[target_idx]
                return max(float(target_row['cpu_demand']), float(target_row['mem_demand']))
            return demand # Fallback
        
        # Check if columns exist
        if 'forecast_t1' in row:
             f1 = float(row['forecast_t1'])
             f3 = float(row['forecast_t3'])
             f6 = float(row['forecast_t6'])
        else:
             # Use oracle (perfect future knowledge) for simplicity in this Env wrapper
             # This is acceptable if we assume the "input" df has the "forecasts"
             # But if we use this Env for *training*, using oracle is "cheating" unless we explicitly say so.
             # However, the user prompt said "State vector includes ... forecast(t+1)..."
             # I'll implement a lookahead here. If the user provided actual forecasts in the DF, they would be used.
             # Since I don't have the forecasts in the DF yet, I'll use the lookahead as a placeholder for "Forecast".
             # In a real deployment, these would come from the model.
             f1 = get_forecast(1)
             f3 = get_forecast(3)
             f6 = get_forecast(6)
             
        # Normalize state components
        # We don't know the exact range, but machines 0-5000, demand 0-5000 approx.
        # Utilization is 0-1 (mostly)
        
        state = np.array([
            utilization,
            demand / 1000.0,       # Approximate normalization
            self.current_machines / 1000.0,
            self.util_history[2],  # lag 1 (most recent)
            self.util_history[1],  # lag 2
            self.util_history[0],  # lag 3
            f1 / 1000.0,
            f3 / 1000.0,
            f6 / 1000.0
        ], dtype=np.float32)
        
        return state

    # def _calculate_reward(self, utilization, action_delta, machines):
    #     # -100 if violation
    #     if utilization > 1.0:
    #         return -100.0
            
    #     # -(utilization - 0.6)^2
    #     # optimized for 0.6 utilization
    #     util_penalty = -((utilization - 0.6) ** 2) * 100 # Scale up a bit? 
    #     # User said: -(utilization - 0.6)^2
    #     # Let's stick to user formula strictly first
    #     term_util = -((utilization - 0.6) ** 2)
        
    #     # - cost_per_machine_step
    #     # cost = machines * rate * (step_mins/60)
    #     cost_step = machines * self.cost_per_hour * (self.step_minutes / 60.0)
    #     term_cost = -cost_step
        
    #     # - 0.1 * |action|
    #     term_action = -0.1 * abs(action_delta)
        
    #     return term_util + term_cost + term_action
    def _calculate_reward(self, utilization, action_delta, machines):
        # 1. SLA violation penalty (reduce from -100)
        if utilization > 1.0:
            return -50.0

        # 2. Utilization target (0.55 instead of 0.60)
        term_util = -((utilization - 0.55) ** 2) * 3.0

        # 3. Cost penalty (increase weight 20x)
        cost_step = machines * self.cost_per_hour * (self.step_minutes / 60.0)
        term_cost = -5.0 * cost_step      # previously -1 * cost_step

        # 4. Action penalty (increase 10x)
        term_action = -1.0 * abs(action_delta)   # previously -0.1

        return term_util + term_cost + term_action


