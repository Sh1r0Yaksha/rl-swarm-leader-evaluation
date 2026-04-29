import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from UAV import UAV

class RLSwarm(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    GRID_SIZE = 200
    TARGET = np.array([100, 100], dtype=np.float32)
    def __init__(self, render_mode=None):
        super(RLSwarm, self).__init__()

        # UAV setup
        self.uav = UAV()
        self.action_space = spaces.Discrete(3)
        self.dt = np.float32(1.0 / self.metadata["render_fps"])

        # Steps setup
        self.max_steps = 1000
        self.current_step = 0

        # Observation Space Setup
        # Obs: [rel_x, rel_y, sin(hdg), cos(hdg)]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Pygame Setup
        self.window_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        # Calculate relative position and normalize by GRID_SIZE
        relative_pos = (self.TARGET - self.uav.position) / self.GRID_SIZE
        
        # Get heading components
        heading_sin = np.sin(self.uav.orientation)
        heading_cos = np.cos(self.uav.orientation)
        
        return np.array([
            relative_pos[0], 
            relative_pos[1], 
            heading_sin, 
            heading_cos
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position to center
        
        self.uav.position = self.np_random.uniform(low=20, high=self.GRID_SIZE-20, size=(2,)).astype(np.float32)
        self.uav.orientation = self.np_random.uniform(low=-np.pi, high=np.pi)
        self.current_step = 0

        info = {}
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), info
    
    def step(self, action):
        step_size = 5.0
        if action == 0: self.uav.angular_speed = 3.0  # Left
        elif action == 1: self.uav.angular_speed = -3.0 # Right
        else: self.uav.angular_speed = 0.0             # Straight

        # 4. Define Reward Logic
        self.uav.step(self.dt)

        observation = self._get_obs()
        
        dist_from_target = np.linalg.norm(self.uav.position - self.TARGET)

        # Progressive reward: getting closer is good
        reward = -dist_from_target / self.GRID_SIZE

        # Survival Penalty
        reward -= 0.1
        # print(f'distance from target: {dist_from_target} reward: {reward}')
        
        # Termination Logic
        terminated = False
        if np.any(self.uav.position < 0) or np.any(self.uav.position > self.GRID_SIZE):
            terminated = True
            reward = -100.0

        if dist_from_target < 50:
            terminated = True 
            reward = 1000.0

        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        # if terminated:
        #     if dist_from_target < 15:
        #         print("DEBUG: Episode ended - SUCCESS")
        #     else:
        #         print(f"DEBUG: Episode ended - CRASH at {self.uav.position}")
        # if truncated:
        #     print("DEBUG: Episode ended - TIMEOUT")

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # White background

        # Draw the Agent (a blue circle)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.uav.position.astype(int),
            20,
        )

        # Draw the target (red dot)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self.TARGET.astype(int),
            5,
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()