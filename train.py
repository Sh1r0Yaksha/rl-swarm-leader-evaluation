import os
import torch
import supersuit as ss
from stable_baselines3 import DQN
from env import RLSwarm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def train():
    num_envs = 1000
    num_uavs_per_env = 5

    # Build instance first, then pass to concat_vec_envs_v1
    env = RLSwarm(num_agents=num_uavs_per_env, render_mode=None)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,                          # ← instance, not callable
        num_envs,
        num_cpus=12,
        base_class="stable_baselines3",
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        target_update_interval=1000,
        verbose=1,
        device=device
    )

    try:
        model.learn(total_timesteps=10000000, progress_bar=True)
        print("\nTraining reached target timesteps.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    finally:
        # This block runs regardless of whether training finished or was interrupted
        model.save("shared_swarm_model")
        env.close()
        print("Model saved as 'shared_swarm_model' and environment closed.")

if __name__ == "__main__":
    train()