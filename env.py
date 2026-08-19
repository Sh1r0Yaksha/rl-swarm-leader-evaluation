import gymnasium as gym
import functools
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import numpy as np
from UAV import UAV, Follower, Leader
import os
from dotenv import load_dotenv

load_dotenv()

RENDER_FPS = int(os.getenv("RENDER_FPS", "150"))
RENDER_MULTIPLIER = int(os.getenv("RENDER_MULTIPLIER", "150"))

class RLSwarm(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}
    GRID_SIZE = int(os.getenv("GRID_SIZE", "150"))
    d1 = int(os.getenv("D1", "5"))
    d2 = int(os.getenv("D2", "10"))
    d3 = int(os.getenv("D3", "20"))
    discount_factor = 0.75
    w_CoA = -20
    w_CoM = 1
    w_Coh = 2
    w_Ali1 = 5
    w_Ali2 = -1
    def __init__(self, leader_uav: Leader, num_agents:int=5, render_mode=None):
        super(RLSwarm, self).__init__()

        self.n_agents = num_agents
        self.possible_agents: list[str] = [f"uav_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.leader = leader_uav
        self.init_leader_pos = leader_uav.position.copy()
        self.init_leader_orient = leader_uav.orientation

        # UAV setup
        self.dt = np.float32(1.0)

        # Steps setup
        self.max_steps = int(os.getenv("EPISODE_TIMESTEPS", "500"))
        self.current_step = 0

        # Shape: 4 (Self pos and orientation) + (n-1 + leader) * 4 (Relative neighbor data)
        self.obs_shape = 4 + (self.n_agents - 1 + 1) * 4

        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_shape,), dtype=np.float32)
            for agent in self.agents
        }
        
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.agents
        }

        self.followers: dict[str, Follower] = {agent: Follower() for agent in self.agents}

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
        uav = self.followers[agent_id]

        # 1. Start with self pos and orientation
        
        obs = [uav.position[0] / self.GRID_SIZE,
               uav.position[1] / self.GRID_SIZE,
               np.sin(uav.orientation),
               np.cos(uav.orientation)]

        # 2. Add the leaders observation
        obs.extend(self._rel_obs(uav, self.leader))

        # 3. Add OTHER Learning agents relative to this one
        for other_id, other_uav in self.followers.items():
            if other_id != agent_id:
                obs.extend(self._rel_obs(uav, other_uav))        

        return np.array(obs, dtype=np.float32)
    
    def _rel_obs(self, main_uav, target_uav):
        """Helper to calculate relative position and heading"""
        rel_pos = (target_uav.position - main_uav.position) / self.GRID_SIZE
        diff_angle = target_uav.orientation - main_uav.orientation
        return [rel_pos[0], rel_pos[1], np.sin(diff_angle), np.cos(diff_angle)]
    
    def set_init_leader_state(self,
                              pos: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
                              orient: np.float32 = np.float32(0.0)):
        """Method to update the starting position from outside"""
        self.init_leader_pos = pos
        self.init_leader_orient = orient

    def reset(self, seed=None, options=None):
        observations = {}
        self.current_step = 0
        leader_x = np.float32(np.random.uniform(self.GRID_SIZE/10, self.GRID_SIZE - self.GRID_SIZE/10))
        leader_y = np.float32(np.random.uniform(self.GRID_SIZE/10, self.GRID_SIZE - self.GRID_SIZE/10))
        leader_pos = np.array([leader_x, leader_y])
        leader_hdg = np.float32(np.random.uniform(-np.pi, np.pi))
        self.leader.reset(leader_pos, leader_hdg)

        initial_positions = [self.leader.position]
        for id, uav in self.followers.items():
            pos = np.random.uniform(100, self.GRID_SIZE - 100, size=(2,)).astype(np.float32)
            orient = np.float32(np.random.uniform(-np.pi, np.pi))
            uav.reset(pos, orient)
            initial_positions.append(pos)

        init_centroid = np.mean(initial_positions, axis=0)

        for agent, uav in self.followers.items():
            uav.prev_dc = np.linalg.norm(uav.position - init_centroid)
            uav.prev_dl = np.linalg.norm(uav.position - self.leader.position)
            observations[agent] = self._get_obs(agent)
        
        info = {}
        if self.render_mode == "human":
            self._render_frame()

        return observations, info
    
    def step(self, actions):
        self.current_step += 1
        
        self.leader.step(self.dt)

        # 1. Update UAV Physics
        for agent_id, action in actions.items():
            uav = self.followers[agent_id]
            # Action space mapped to angular changes
            if action == 0: uav.angular_direction = 1
            elif action == 1: uav.angular_direction = -1
            else: uav.angular_direction = 0
            uav.step(self.dt)

        # 2. Pre-calculate group metrics for rewards

        all_pos = np.array([u.position for u in self.followers.values()] + [self.leader.position])
        centroid = np.mean(all_pos, axis=0) #
        
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent_id, uav in self.followers.items():
            # Initial reward components
            r_CoA = 0  # Collision Avoidance
            r_Coh = 0  # Cohesion
            r_CoM = 0  # Connectivity Maintenance
            r_Ali = 0  # Alignment (Leader-Following)


            current_dc = np.linalg.norm(uav.position - centroid)
            current_dl = np.linalg.norm(uav.position - self.leader.position)

            # a) Collision Avoidance: Penalty if distance < d1
            for other_id, other_uav in self.followers.items():
                if agent_id != other_id:
                    dist = np.linalg.norm(uav.position - other_uav.position)
                    if dist < self.d1:
                        r_CoA += self.w_CoA

            if current_dl < self.d1:
                r_CoA += self.w_CoA

            # b) Cohesion: Reward for moving closer to the centroid
            # Note: Requires tracking previous distance to centroid

            prev_dc = uav.prev_dc
            if current_dc < prev_dc:
                r_Coh = self.w_Coh
            uav.prev_dc = current_dc

            # c) Alignment with Leader: Reward for moving closer to the Leader
            prev_dl = uav.prev_dl
            if current_dl < prev_dl:
                r_Ali = self.w_Ali1
            elif current_dl > prev_dl:
                r_Ali = self.w_Ali2

            # d) Connectivity Maintenance: Reward for staying in 'Flight Zone'
            # Flight zone is defined between d1 and d3
            for other_id, other_uav in self.followers.items():
                if agent_id != other_id:
                    dist = np.linalg.norm(uav.position - other_uav.position)
                    if self.d1 <= dist <= self.d3:
                        r_CoM += self.w_CoM

            if self.d1 <= current_dl <= self.d3:
                r_CoM += self.w_CoM

            # Final reward summation
            rewards[agent_id] = r_CoA + r_Coh + r_CoM + r_Ali
            
            # Boundary & Step Logic
            out_of_bounds = np.any(uav.position < 0) or np.any(uav.position > self.GRID_SIZE)
            terminations[agent_id] = out_of_bounds
            truncations[agent_id] = self.current_step >= self.max_steps
            observations[agent_id] = self._get_obs(agent_id)
            infos[agent_id] = {}
        
        leader_out_of_bounds = np.any(self.leader.position < 0) or np.any(self.leader.position > self.GRID_SIZE)
        if any(terminations.values()):
            for agent_id in self.followers:
                terminations[agent_id] = True
                rewards[agent_id] -= 100

        if leader_out_of_bounds:
            for agent_id in self.followers:
                terminations[agent_id] = True

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        arrow_multiplier = min(2, RENDER_MULTIPLIER)
        # Scale canvas and window size
        scaled_size = int(self.window_size * RENDER_MULTIPLIER)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((scaled_size, scaled_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((scaled_size, scaled_size))
        canvas.fill((255, 255, 255)) # White background

        # Scale arrow dimensions and line thickness proportionally
        arrow_length = 15 * arrow_multiplier
        head_length = 7 * arrow_multiplier
        line_width = max(1, int(3 * arrow_multiplier))
        head_angle = 0.4
        
        # 1. Followers (Blue)
        color = (0, 0, 255)
        for agent_id, uav in self.followers.items():
            start_point, end_point, w1, w2 = self.make_arrow(uav, arrow_length, head_length, head_angle)
            pygame.draw.line(canvas, color, start_point, end_point, line_width)
            pygame.draw.line(canvas, color, end_point, w1, line_width)
            pygame.draw.line(canvas, color, end_point, w2, line_width)
        
        # 2. Leader (Red)
        color = (255, 0, 0)
        start_point, end_point, w1, w2 = self.make_arrow(self.leader, arrow_length, head_length, head_angle)
        pygame.draw.line(canvas, color, start_point, end_point, line_width)
        pygame.draw.line(canvas, color, end_point, w1, line_width)
        pygame.draw.line(canvas, color, end_point, w2, line_width)

        if self.render_mode == "human":
            if self.clock is not None and self.window is not None:
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(RENDER_FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


    def make_arrow(self, uav, arrow_length, head_length, head_angle):
        # Scale UAV coordinates by RENDER_MULTIPLIER
        cx = float(uav.position[0]) * RENDER_MULTIPLIER
        cy = float(uav.position[1]) * RENDER_MULTIPLIER
        angle = float(uav.orientation)

        # Calculate shaft
        fwd_x = float(arrow_length * np.cos(angle))
        fwd_y = float(arrow_length * np.sin(angle))
        start_point = (cx, cy)
        end_point = (cx + fwd_x, cy + fwd_y)

        # Calculate wings
        alpha = angle + np.pi - head_angle
        beta = angle + np.pi + head_angle
        
        w1 = (
            float(end_point[0] + head_length * np.cos(alpha)), 
            float(end_point[1] + head_length * np.sin(alpha))
        )
        w2 = (
            float(end_point[0] + head_length * np.cos(beta)), 
            float(end_point[1] + head_length * np.sin(beta))
        )
        
        return start_point, end_point, w1, w2
        


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()