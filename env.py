import gymnasium as gym
import functools
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import numpy as np
from UAV import UAV

class RLSwarm(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    GRID_SIZE = 200
    def __init__(self,num_agents:int=5, render_mode=None):
        super(RLSwarm, self).__init__()

        self.TARGET = np.array([100, 100], dtype=np.float32)

        self.n_agents = num_agents
        self.possible_agents = [f"uav_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # UAV setup
        # self.uav = UAV()
        # self.action_space = spaces.Discrete(3)
        self.dt = np.float32(1.0 / self.metadata["render_fps"])

        # Steps setup
        self.max_steps = 1000
        self.current_step = 0

        # Observation Space Setup
        # Obs: [rel_x, rel_y, sin(hdg), cos(hdg)]
        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
            for agent in self.possible_agents
        }
        # self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }

        self.uavs = {agent: UAV() for agent in self.possible_agents}

        # Pygame Setup
        self.window_size = self.GRID_SIZE
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self.action_spaces[agent]

    def _get_obs(self, agent_id):
        uav = self.uavs[agent_id]
        # Calculate relative position and normalize by GRID_SIZE
        rel_pos = (self.TARGET - uav.position) / self.GRID_SIZE
        
        # Feature: normalized agent index to help parameter sharing
        idx = self.possible_agents.index(agent_id) / self.n_agents
        return np.array([
            rel_pos[0], rel_pos[1], 
            np.sin(uav.orientation), np.cos(uav.orientation),
            idx
        ], dtype=np.float32)

        # # Get heading components
        # heading_sin = np.sin(self.uav.orientation)
        # heading_cos = np.cos(self.uav.orientation)
        
        # return np.array([
        #     relative_pos[0], 
        #     relative_pos[1], 
        #     heading_sin, 
        #     heading_cos
        # ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        observations = {}
        self.current_step = 0
        for agent in self.agents:
            self.uavs[agent].position = np.random.uniform(20, 180, size=(2,)).astype(np.float32)
            self.uavs[agent].orientation = np.random.uniform(-np.pi, np.pi)
            observations[agent] = self._get_obs(agent)
        
        info = {}
        if self.render_mode == "human":
            self._render_frame()

        return observations, info
        
        
        # # Reset agent position to center
        
        # self.uav.position = self.np_random.uniform(low=20, high=self.GRID_SIZE-20, size=(2,)).astype(np.float32)
        # self.TARGET = self.np_random.uniform(low=0, high=self.GRID_SIZE, size=(2,)).astype(np.float32)
        # self.uav.orientation = self.np_random.uniform(low=-np.pi, high=np.pi)
        # self.current_step = 0

        # info = {}
        # if self.render_mode == "human":
        #     self._render_frame()

        # return self._get_obs(), info
    
    def step(self, actions):
        self.current_step += 1  # ← move to top, outside any loop

        for agent_id, action in actions.items():
            uav = self.uavs[agent_id]
            if action == 0: uav.angular_direction = 1
            elif action == 1: uav.angular_direction = -1
            else: uav.angular_direction = 0
            uav.step(self.dt)

        all_pos = np.array([u.position for u in self.uavs.values()])
        centroid = np.mean(all_pos, axis=0)
        dist_centroid = np.linalg.norm(centroid - self.TARGET)

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent_id in self.agents:
            uav = self.uavs[agent_id]
            dist_target = np.linalg.norm(uav.position - self.TARGET)

            r = 0.6 * (-dist_target / self.GRID_SIZE) + 0.4 * (-dist_centroid / self.GRID_SIZE)

            for other_id, other_uav in self.uavs.items():
                if agent_id != other_id:
                    if np.linalg.norm(uav.position - other_uav.position) < 10:
                        r -= 2.0

            # Success condition
            if dist_target < 25:
                r += 100.0
                terminations[agent_id] = True
            else:
                out_of_bounds = np.any(uav.position < 0) or np.any(uav.position > self.GRID_SIZE)
                terminations[agent_id] = out_of_bounds

            truncations[agent_id] = self.current_step >= self.max_steps  # ← uses self, not agent_id
            rewards[agent_id] = r
            observations[agent_id] = self._get_obs(agent_id)
            infos[agent_id] = {}

        # Remove done agents
        self.agents = [
            agent for agent in self.agents
            if not terminations[agent] and not truncations[agent]
        ]

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos
        
        # # 4. Define Reward Logic
        # self.uav.step(self.dt)

        # observation = self._get_obs()
        
        # dist_from_target = np.linalg.norm(self.uav.position - self.TARGET)

        # # Progressive reward: getting closer is good
        # reward = -dist_from_target / self.GRID_SIZE

        # # Survival Penalty
        # reward -= 0.1
        # # print(f'distance from target: {dist_from_target} reward: {reward}')
        
        # # Termination Logic
        # terminated = False
        # if np.any(self.uav.position < 0) or np.any(self.uav.position > self.GRID_SIZE):
        #     terminated = True
        #     reward = -100.0

        # if dist_from_target < 25:
        #     terminated = True 
        #     reward = 1000.0

        # truncated = False
        # self.current_step += 1
        # if self.current_step >= self.max_steps:
        #     truncated = True
        
        # info = {}

        # if self.render_mode == "human":
        #     self._render_frame()

        # if terminated:
        #     if dist_from_target < 15:
        #         print("DEBUG: Episode ended - SUCCESS")
        #     else:
        #         print(f"DEBUG: Episode ended - CRASH at {self.uav.position}")
        # if truncated:
        #     print("DEBUG: Episode ended - TIMEOUT")

        # return observation, reward, terminated, truncated, info
    
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

        # 1. Draw the target (red dot) first so UAVs are on top
        pygame.draw.circle(canvas, (255, 0, 0), self.TARGET.astype(int), 25)

        # 2. Iterate through all UAVs and draw them as arrows
        color = (0, 0, 255) # Blue
        arrow_length = 15
        head_length = 7
        head_angle = 0.4

        for agent_id, uav in self.uavs.items():
            # Get position and orientation for this specific UAV
            cx, cy = float(uav.position[0]), float(uav.position[1])
            angle = float(uav.orientation)

            # Calculate shaft
            fwd_x = float(arrow_length * np.cos(angle))
            fwd_y = float(arrow_length * np.sin(angle))
            start_point = (cx, cy)
            end_point = (cx + fwd_x, cy + fwd_y)

            # Draw shaft
            pygame.draw.line(canvas, color, start_point, end_point, 3)

            # Calculate and draw wings
            alpha = angle + np.pi - head_angle
            beta = angle + np.pi + head_angle
            
            w1 = (float(end_point[0] + head_length * np.cos(alpha)), 
                  float(end_point[1] + head_length * np.sin(alpha)))
            w2 = (float(end_point[0] + head_length * np.cos(beta)), 
                  float(end_point[1] + head_length * np.sin(beta)))

            pygame.draw.line(canvas, color, end_point, w1, 3)
            pygame.draw.line(canvas, color, end_point, w2, 3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()