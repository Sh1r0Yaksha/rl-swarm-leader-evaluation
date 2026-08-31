import os
import numpy as np
from dotenv import load_dotenv
from env import RLSwarm
from UAV import Leader
from boid_model import ExtendedBoidModel

load_dotenv()

GRID_SIZE = int(os.getenv("GRID_SIZE", "150"))

def test_boid():
    num_uavs = int(os.getenv("NUM_AGENTS", "16"))
    pos = np.random.uniform(100, GRID_SIZE - 100, size=2).astype(np.float32)
    hdg = np.float32(np.random.uniform(-np.pi, np.pi))
    leader = Leader(position=pos, orientation=hdg)
    
    # Initialize env (Set log_csv=True to automatically write metrics to CSV)
    env = RLSwarm(leader_uav=leader, num_agents=num_uavs, render_mode="human", log_csv=True)
    boid = ExtendedBoidModel(followers=env.followers, leader=env.leader)

    obs, info = env.reset()
    
    try:
        while True:
            # Generate rule-based boid actions directly
            actions = boid.predict_actions()
            obs, rewards, terminations, truncations, infos = env.step(actions)

            if any(terminations.values()) or any(truncations.values()):
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("Test stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    test_boid()