from env import RLSwarm
from stable_baselines3 import DQN

env = RLSwarm(render_mode=None)

model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    target_update_interval=1000,
    gamma=0.99,
    exploration_fraction=0.2
)

# 3. Train the Agent
print("Training started...")
model.learn(total_timesteps=100000)
model.save("dqn_gym_model")
print("Training complete!")