from env import RLSwarm
from stable_baselines3 import DQN

env = RLSwarm(render_mode="human")
obs, info = env.reset()
model = DQN.load("dqn_gym_model")

steps = 0
while True:
    # predict() returns the best action based on the learned policy
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if (steps > 200):
        truncated = True
        steps = 0
        
    if terminated or truncated:
        obs, info = env.reset()

env.close()