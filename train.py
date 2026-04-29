from env import RLSwarm
from stable_baselines3 import DQN

env = RLSwarm(render_mode=None)

model = DQN(
    "MlpPolicy", 
    env, 
    learning_rate=5e-4,      
    buffer_size=100000, 
    batch_size=64,           
    gamma=0.99, 
    exploration_final_eps=0.05, 
    exploration_fraction=0.3,
    verbose=1
)

# 3. Train the Agent
print("Training started...")
model.learn(total_timesteps=500000)
model.save("dqn_gym_model")
print("Training complete!")