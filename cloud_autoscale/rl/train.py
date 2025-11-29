import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm
from typing import Dict, Any

from cloud_autoscale.data import SyntheticLoader, GCP2019Loader
from cloud_autoscale.rl.agent import DQNAgent
from cloud_autoscale.rl.env import AutoscaleEnv

def train_rl(config: Dict[str, Any], output_dir: Path):
    """
    Train RL agent.
    """
    print("=" * 70)
    print("RL TRAINING START")
    print("=" * 70)
    
    # 1. Load Data
    print("📊 Loading training data...")
    if config['mode'] == 'synthetic':
        loader = SyntheticLoader(
            pattern=config['data'].get('synthetic_pattern', 'periodic'),
            duration_minutes=config['data'].get('duration_minutes', 1440),
            step_minutes=config['simulation']['step_minutes'],
            seed=42
        )
    elif config['mode'] == 'gcp_2019':
        loader = GCP2019Loader(
            processed_dir=config['data']['processed_dir'],
            step_minutes=config['simulation']['step_minutes'],
            duration_minutes=config['data'].get('duration_minutes')
        )
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")
        
    demand_df = loader.load()
    print(f"   ✓ Loaded {len(demand_df)} steps")
    
    # 2. Initialize Environment
    print("🌍 Initializing environment...")
    env = AutoscaleEnv(
        demand_df=demand_df,
        sim_config=config['simulation'],
        autoscaler_config=config['autoscaler']
    )
    
    # 3. Initialize Agent
    print("🤖 Initializing DQN agent...")
    rl_config = config['training']
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=float(rl_config['lr']),
        gamma=float(rl_config['gamma']),
        epsilon_start=float(rl_config['epsilon_start']),
        epsilon_end=float(rl_config['epsilon_end']),
        epsilon_decay=float(rl_config['epsilon_decay']),
        replay_size=int(rl_config['replay_size']),
        batch_size=int(rl_config['batch_size']),
        device=device
    )
    
    # 4. Training Loop
    num_episodes = int(rl_config['num_episodes'])
    max_steps = int(rl_config.get('max_steps', len(demand_df)))
    
    print(f"▶️  Starting training for {num_episodes} episodes...")
    
    best_reward = -float('inf')
    model_dir = output_dir / "model_rl"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        pbar = tqdm(range(max_steps), desc=f"Ep {episode+1}/{num_episodes}", leave=False)
        for step in pbar:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
                
        metrics.append({
            'episode': episode,
            'reward': episode_reward,
            'epsilon': agent.epsilon
        })
        
        print(f"   Episode {episode+1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(str(model_dir / "agent_policy_best.pth"))
            
        # Always save latest
        agent.save(str(model_dir / "agent_policy.pth"))
        
    # Save training metrics
    pd.DataFrame(metrics).to_csv(output_dir / "training_metrics.csv", index=False)
    print(f"✅ Training completed. Model saved to {model_dir}")

