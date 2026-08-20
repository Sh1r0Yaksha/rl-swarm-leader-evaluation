import os
from dotenv import load_dotenv
import torch
import numpy as np
import supersuit as ss
from stable_baselines3 import DQN
from env import RLSwarm
from UAV import Leader

load_dotenv()

GRID_SIZE = int(os.getenv("GRID_SIZE", "150"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def make_env():
    leader = Leader(position=np.array([GRID_SIZE/2, GRID_SIZE/2]),
                    orientation=np.float32(0))
    env = RLSwarm(leader_uav=leader, num_agents=int(os.getenv("NUM_AGENTS", "16")), render_mode=None)
    # env = ss.black_death_v3(env)
    env = ss.pad_observations_v0(env)
    return env

def train():
    num_envs = int(os.getenv("TRAIN_NUM_ENV", "16"))
    # Correct way to vectorize: pass a lambda
    env = ss.pettingzoo_env_to_vec_env_v1(make_env())
    env = ss.concat_vec_envs_v1(
        env, 
        num_envs, 
        num_cpus=12, 
        base_class="stable_baselines3"
    )
    
    model = DQN.load(os.getenv("SAVE_NAME", "swarm_model"), env=env)

    model.exploration_initial_eps = 0.1
    model.exploration_final_eps = 0

    try:
        model.learn(
            total_timesteps=int(os.getenv("TRAIN_TIMESTEPS", "50_000")),
            progress_bar=True,
            tb_log_name="dqn_swarm_run",
            log_interval=1
        )
    finally:
        model.save(os.getenv("REFINE_SAVE_NAME", "swarm_model"))
        env.close()

if __name__ == "__main__":
    train()