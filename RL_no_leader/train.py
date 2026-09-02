import os
from dotenv import load_dotenv
import torch
import numpy as np
import supersuit as ss
from stable_baselines3 import DQN
from RL_no_leader.env import RLNoLeader
from UAV import Leader

load_dotenv()

GRID_SIZE = int(os.getenv("GRID_SIZE", "150"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def make_env():
    pos = np.random.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    hdg = np.float32(np.random.uniform(-np.pi, np.pi))
    leader = Leader(position=pos,
                    orientation=hdg)
    env = RLNoLeader(leader_uav=leader, num_agents=int(os.getenv("NUM_AGENTS", "16")), render_mode=None)
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
    
    model = DQN(
        "MlpPolicy",
        env,
        gamma=0.75,
        verbose=1,
        tensorboard_log="./tb_logs/",
        exploration_fraction=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        model.learn(
            total_timesteps=int(os.getenv("TRAIN_TIMESTEPS", "50_000")),
            progress_bar=True,
            tb_log_name="dqn_swarm_run",
            log_interval=1
        )
    finally:
        file_name = os.getenv("SAVE_NAME", "swarm_model")
        file_path = f"models/{file_name}" 
        model.save(file_path)
        env.close()

if __name__ == "__main__":
    train()