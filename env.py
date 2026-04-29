import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class RLSwarm(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    GRID_SIZE = 500
    TARGET = np.array([100, 100], dtype=np.float32)
    def __init__(self, render_mode=None):
        super(RLSwarm, self).__init__()

        self.action_space = spaces.Discrete(4)

        # 2. Define Observation Space
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)

        # Pygame Setup
        self.window_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position to center
        
        self.state = self.np_random.uniform(low=0, high=self.GRID_SIZE, size=(2,)).astype(np.float32)
        
        info = {}
        if self.render_mode == "human":
            self._render_frame()

        return self.state / self.GRID_SIZE, info
    
    def step(self, action):
        # 3. Apply Logic
        # 0: Left, 1: Right, 2: Up, 3: Down
        step_size = 5.0
        if action == 0: self.state[0] -= step_size
        if action == 1: self.state[0] += step_size
        if action == 2: self.state[1] -= step_size
        if action == 3: self.state[1] += step_size

        # 4. Define Reward Logic
        
        dist_from_target = np.linalg.norm(self.state - self.TARGET)

        reward = (1 / (1 + dist_from_target)) - 0.1
        # print(f'distance from target: {dist_from_target} reward: {reward}')
        
        # 5. Define Termination Logic
        terminated = False
        if np.any(self.state < 0) or np.any(self.state > 500):
            terminated = True
            reward = -25.0

        if dist_from_target < 5:
            terminated = True 
            reward = 1000.0

        truncated = False
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return self.state / self.GRID_SIZE, reward, terminated, truncated, info
    
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
            self.state.astype(int),
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