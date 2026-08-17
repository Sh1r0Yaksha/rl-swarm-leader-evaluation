import os
from dotenv import load_dotenv
import supersuit as ss
from env import RLSwarm
from stable_baselines3 import DQN
import numpy as np
from UAV import Leader

load_dotenv()

def test():
    num_uavs = int(os.getenv("NUM_AGENTS", "16"))
    leader = Leader(position=np.array([300, 300]),
                    orientation=np.float32(0))
    env = RLSwarm(leader_uav=leader, num_agents=num_uavs, render_mode="human")
    # env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    try:
        model = DQN.load("swarm_model", env=env)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Check the filename.")
        env.close()
        return

    obs = env.reset()
    print("Starting visualization. Press Ctrl+C to stop.")

    episode = 0
    total_reward = np.zeros(num_uavs)

    try:
        while True:
            actions, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)

            total_reward += rewards

            # SB3 vectorized envs auto-reset on done, but we track episodes manually
            if dones.any():
                episode += 1
                print(f"Episode {episode} finished | Rewards: {total_reward}")
                total_reward = np.zeros(num_uavs)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        env.close()

if __name__ == "__main__":
    test()