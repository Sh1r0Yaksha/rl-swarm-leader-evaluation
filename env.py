import gymnasium as gym
import functools
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import numpy as np
from UAV import UAV

class RLSwarm(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    GRID_SIZE = 500
    d1 = 20
    d2 = 40
    d3 = 80
    discount_factor = 0.75
    w_CoA = -20
    w_CoM = 1
    w_Coh = 2
    n_UAVs = 5
    def __init__(self,num_agents:int=5, render_mode=None):
        super(RLSwarm, self).__init__()

        # self.TARGET = np.array([100, 100], dtype=np.float32)

        self.n_agents = num_agents
        self.possible_agents = [f"uav_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        # UAV setup
        self.dt = np.float32(1.0 / self.metadata["render_fps"])

        # Steps setup
        self.max_steps = 1000
        self.current_step = 0

        # Observation Space Setup
        # Obs: [rel_pos_with_neighbours, relo_orient_with_neighbours]
        # self.observation_spaces = {
        #     agent: spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        #     for agent in self.possible_agents
        # }

        # Shape: 2 (Self orientation) + (n-1)*4 (Relative neighbor data)
        self.obs_shape = 2 + (self.n_agents - 1) * 4

        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_shape,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
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
        # rel_pos = (self.TARGET - uav.position) / self.GRID_SIZE

        # 1. Start with self orientation
        obs = [np.sin(uav.orientation), np.cos(uav.orientation)]

        # 2. Add all OTHER agents relative to this one
        for other_id in self.possible_agents:
            if other_id == agent_id:
                continue

            other_uav = self.uavs[other_id]

            # Position of 'other' relative to 'self'
            # Normalized by a "Sensing Range" or GRID_SIZE
            rel_pos = (other_uav.position - uav.position) / self.GRID_SIZE

            # Heading of 'other' relative to 'self'
            diff_angle = other_uav.orientation - uav.orientation
            rel_hdg_sin = np.sin(diff_angle)
            rel_hdg_cos = np.cos(diff_angle)

            obs.extend([rel_pos[0], rel_pos[1], rel_hdg_sin, rel_hdg_cos])
        
        return np.array(obs, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        observations = {}
        self.current_step = 0

        initial_positions = []
        for agent in self.agents:
            pos = np.random.uniform(100, 400, size=(2,)).astype(np.float32)
            self.uavs[agent].position = pos
            self.uavs[agent].orientation = np.random.uniform(-np.pi, np.pi)
            initial_positions.append(pos)

        init_centroid = np.mean(initial_positions, axis=0)

        for agent in self.agents:
            uav = self.uavs[agent]
            uav.prev_dc = np.linalg.norm(uav.position - init_centroid)
            observations[agent] = self._get_obs(agent)
        
        info = {}
        if self.render_mode == "human":
            self._render_frame()

        return observations, info
    
    def step(self, actions):
        self.current_step += 1
        
        # 1. Update UAV Physics
        for agent_id, action in actions.items():
            uav = self.uavs[agent_id]
            # Action space mapped to angular changes
            if action == 0: uav.angular_direction = 1
            elif action == 1: uav.angular_direction = -1
            else: uav.angular_direction = 0
            uav.step(self.dt)

        # 2. Pre-calculate group metrics for rewards
        all_pos = np.array([u.position for u in self.uavs.values()])
        centroid = np.mean(all_pos, axis=0) #
        
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent_id, uav in self.uavs.items():
            # Initial reward components
            r_CoA = 0  # Collision Avoidance
            r_Coh = 0  # Cohesion
            r_CoM = 0  # Connectivity Maintenance
            r_Ali = 0  # Alignment (Leader-Following)

            # a) Collision Avoidance: Penalty if distance < d1
            for other_id, other_uav in self.uavs.items():
                if agent_id != other_id:
                    dist = np.linalg.norm(uav.position - other_uav.position)
                    if dist < self.d1:
                        r_CoA += self.w_CoA

            # b) Cohesion: Reward for moving closer to the centroid
            # Note: Requires tracking previous distance to centroid
            curr_dc = np.linalg.norm(uav.position - centroid)
            prev_dc = getattr(uav, 'prev_dc', curr_dc)
            if curr_dc < prev_dc:
                r_Coh = self.w_Coh
            uav.prev_dc = curr_dc

            # c) Connectivity Maintenance: Reward for staying in 'Flight Zone'
            # Flight zone is defined between d1 and d3
            for other_id, other_uav in self.uavs.items():
                if agent_id != other_id:
                    dist = np.linalg.norm(uav.position - other_uav.position)
                    if self.d1 <= dist <= self.d3:
                        r_CoM += self.w_CoM

            # Final reward summation
            rewards[agent_id] = r_CoA + r_Coh + r_CoM + r_Ali
            
            # Boundary & Step Logic
            out_of_bounds = np.any(uav.position < 0) or np.any(uav.position > self.GRID_SIZE)
            terminations[agent_id] = out_of_bounds
            truncations[agent_id] = self.current_step >= self.max_steps
            observations[agent_id] = self._get_obs(agent_id)
            infos[agent_id] = {}

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos
    
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

        # # 1. Draw the target (red dot) first so UAVs are on top
        # pygame.draw.circle(canvas, (255, 0, 0), self.TARGET.astype(int), 25)

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

        if self.render_mode == "human" :
            if self.clock is not None and self.window is not None:
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