import os
import torch
import numpy as np
import supersuit as ss
from stable_baselines3 import DQN
from env import RLSwarm
from UAV import Leader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def make_env():
    leader = Leader(position=np.array([300, 300]),
                    orientation=np.float32(0))
    env = RLSwarm(leader_uav=leader, num_agents=16, render_mode=None)
    # env = ss.black_death_v3(env)
    env = ss.pad_observations_v0(env)
    return env

def train():
    num_envs = 32
    # Correct way to vectorize: pass a lambda
    env = ss.pettingzoo_env_to_vec_env_v1(make_env())
    env = ss.concat_vec_envs_v1(
        env, 
        num_envs, 
        num_cpus=4, 
        base_class="stable_baselines3"
    )
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4, 
        buffer_size=1000,
        batch_size=128,
        gamma=0.75,
        exploration_final_eps=0.01,
        exploration_fraction=0.3,
        target_update_interval=1000,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        model.learn(total_timesteps=50_000_000, progress_bar=True)
    finally:
        model.save("swarm_model")
        env.close()

if __name__ == "__main__":
    train()