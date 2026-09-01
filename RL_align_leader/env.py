import gymnasium as gym
import functools
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import numpy as np
from UAV import UAV, Follower, Leader
import os
import csv
from dotenv import load_dotenv

load_dotenv()

RENDER_FPS = int(os.getenv("RENDER_FPS", "150"))
RENDER_MULTIPLIER = int(os.getenv("RENDER_MULTIPLIER", "150"))

class RLAlignLeader(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}
    GRID_SIZE = int(os.getenv("GRID_SIZE", "150"))
    spawn_margin = GRID_SIZE/10
    d1 = int(os.getenv("D1", "5"))
    d2 = int(os.getenv("D2", "10"))
    d3 = int(os.getenv("D3", "20"))
    discount_factor = 0.75
    w_CoA = -20
    w_CoM = 0
    w_Coh = 0
    w_Ali1 = 20
    w_Ali2 = -4
    w_CwB = 0
    w_Hdg_align = 20.0   # weight with range [-20.0, +20.0] for leader heading
    def __init__(self, leader_uav: Leader, num_agents:int=5, render_mode=None, log_csv=False):
        super(RLAlignLeader, self).__init__()

        self.n_agents = num_agents
        self.possible_agents: list[str] = [f"uav_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.leader = leader_uav
        self.init_leader_pos = leader_uav.position.copy()
        self.init_leader_orient = leader_uav.orientation

        # Metric tracking variables
        self.proximity_advances = 0
        self.ideal_range_tracking = 0
        self.near_collision_events = 0
        self.heading_deviations = []
        self.prev_dl = {}
        self.run_count = 0
        self.step_counter = 0

        # CSV Logging Setup
        self.log_csv = log_csv
        if self.log_csv:
            self.csv_filepath = self._init_csv_file()

        # UAV setup
        self.dt = np.float32(1.0)

        # Steps setup
        self.max_steps = int(os.getenv("EPISODE_TIMESTEPS", "500"))
        self.current_step = 0

        # Shape: 2 (Self orientation) + (n-1 + leader) * 4 (Relative neighbor data)
        self.obs_shape = 2 + (self.n_agents - 1 + 1) * 4

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

    def _init_csv_file(self, folder="metrics", prefix="metrics_", ext=".csv") -> str:
        """Creates the 'metrics' directory and returns the next auto-incremented filepath."""
        os.makedirs(folder, exist_ok=True)
        counter = 1
        while True:
            filepath = os.path.join(folder, f"{prefix}{counter}{ext}")
            if not os.path.exists(filepath):
                break
            counter += 1

        fieldnames = [
            "Run",
            "Steps",
            "Proximity Advances",
            "Ideal Range Tracking",
            "Near-Collision Events",
            "Heading Consistency",
        ]
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        return filepath

    def _log_run_to_csv(self):
        """Appends the accumulated metric totals for the current run as a row in the CSV."""
        if not self.log_csv or self.step_counter == 0:
            return

        self.run_count += 1
        avg_heading = float(np.mean(self.heading_deviations)) if self.heading_deviations else 0.0

        row = {
            "Run": self.run_count,
            "Steps": self.step_counter,
            "Proximity Advances": self.proximity_advances,
            "Ideal Range Tracking": self.ideal_range_tracking,
            "Near-Collision Events": self.near_collision_events,
            "Heading Consistency": round(avg_heading, 2),
        }

        fieldnames = [
            "Run",
            "Steps",
            "Proximity Advances",
            "Ideal Range Tracking",
            "Near-Collision Events",
            "Heading Consistency",
        ]
        with open(self.csv_filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _compute_heading_diff_deg(self, uav_orient: float, leader_orient: float) -> float:
        """Calculates directional heading difference in degrees bounded to [0, 180]."""
        diff_rad = np.abs(uav_orient - leader_orient)
        diff_rad = np.arctan2(np.sin(diff_rad), np.cos(diff_rad))
        return float(np.degrees(diff_rad))

    def _compute_orientation_error(self, uav_orient: np.float32, leader_orient: np.float32) -> np.float32:
        """Calculates absolute heading difference wrapped to [0, pi] radians."""
        diff = np.arctan2(
            np.sin(uav_orient - leader_orient),
            np.cos(uav_orient - leader_orient)
        )
        return np.float32(np.abs(diff))

    def _get_obs(self, agent_id):
        uav = self.followers[agent_id]

        # 1. Start with self orientation
        
        obs = [np.sin(uav.orientation),
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
        leader_x = np.float32(np.random.uniform((self.GRID_SIZE/2) - self.spawn_margin, (self.GRID_SIZE/2) + self.spawn_margin))
        leader_y = np.float32(np.random.uniform((self.GRID_SIZE/2) - self.spawn_margin, (self.GRID_SIZE/2) + self.spawn_margin))
        leader_pos = np.array([leader_x, leader_y])
        leader_hdg = np.float32(np.random.uniform(-np.pi, np.pi))
        self.leader.reset(leader_pos, leader_hdg)

        initial_positions = [self.leader.position]
        for id, uav in self.followers.items():
            pos_x = np.float32(np.random.uniform((self.GRID_SIZE/2) - self.spawn_margin, (self.GRID_SIZE/2) + self.spawn_margin))
            pos_y = np.float32(np.random.uniform((self.GRID_SIZE/2) - self.spawn_margin, (self.GRID_SIZE/2) + self.spawn_margin))
            pos = np.array([pos_x, pos_y])
            orient = np.float32(np.random.uniform(-np.pi, np.pi))
            uav.reset(pos, orient)
            initial_positions.append(pos)

        init_centroid = np.mean(initial_positions, axis=0)

        for agent, uav in self.followers.items():
            uav.prev_dc = np.linalg.norm(uav.position - init_centroid)
            uav.prev_dl = np.linalg.norm(uav.position - self.leader.position)
            observations[agent] = self._get_obs(agent)

        # Reset metric counters on episode reset
        self.proximity_advances = 0
        self.ideal_range_tracking = 0
        self.near_collision_events = 0
        self.heading_deviations = []
        self.step_counter = 0

        # Store initial distance to leader for each follower
        self.prev_dl = {
            agent_id: np.linalg.norm(uav.position - self.leader.position)
            for agent_id, uav in self.followers.items()
        }
        
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

        self.step_counter += 1
        step_heading_diffs = []

        # 2. Pre-calculate group metrics for rewards
        all_pos = np.array([u.position for u in self.followers.values()] + [self.leader.position])
        centroid = np.mean(all_pos, axis=0) #
        
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        for agent_id, uav in self.followers.items():
            # Initial reward components
            r_CoA = 0  # Collision Avoidance
            r_Coh = 0  # Cohesion
            r_CoM = 0  # Connectivity Maintenance
            r_Ali = 0  # Distance Alignment (Leader)
            r_Hdg = 0  # Angle Alignment (Leader)

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
            # Distance
            prev_dl = uav.prev_dl
            if current_dl < prev_dl:
                r_Ali = self.w_Ali1
            elif current_dl > prev_dl:
                r_Ali = self.w_Ali2
            uav.prev_dl = current_dl

            # Angle
            # Cosine similarity between follower heading and leader heading
            # Range: [-1.0 (opposite) to +1.0 (perfectly aligned)]
            cos_sim = np.cos(uav.orientation - self.leader.orientation)
            r_Hdg = cos_sim * self.w_Hdg_align

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
            rewards[agent_id] = r_CoA + r_Coh + r_CoM + r_Ali + r_Hdg
            
            # Boundary & Step Logic
            out_of_bounds = np.any(uav.position < 0) or np.any(uav.position > self.GRID_SIZE)
            terminations[agent_id] = out_of_bounds
            if (out_of_bounds):
                rewards[agent_id] += self.w_CwB
            truncations[agent_id] = self.current_step >= self.max_steps
            observations[agent_id] = self._get_obs(agent_id)
            infos[agent_id] = {}

            # Metrics Calculation
            # 1. Proximity Advances (distance to leader decreased)
            if current_dl < self.prev_dl.get(agent_id, current_dl):
                self.proximity_advances += 1
            self.prev_dl[agent_id] = current_dl

            # 2. Ideal Range Tracking (d1 <= distance <= d2)
            if self.d1 <= current_dl <= self.d2:
                self.ideal_range_tracking += 1

            # 3. Near-Collision Events (distance < d1)
            if current_dl < self.d1:
                self.near_collision_events += 1

            # 4. Heading Consistency (angular error in degrees)
            h_diff = self._compute_heading_diff_deg(float(uav.orientation), float(self.leader.orientation))
            step_heading_diffs.append(h_diff)
            self.heading_deviations.append(h_diff)

        step_avg_heading = np.mean(step_heading_diffs) if step_heading_diffs else 0.0
        cum_avg_heading = np.mean(self.heading_deviations) if self.heading_deviations else 0.0
        
        if any(terminations.values()):
            for agent_id in self.followers:
                terminations[agent_id] = True

        if self.render_mode == "human":
            self._render_frame()

        # # Live metric logging directly from the step method
        # print(
        #     f"Step {self.step_counter:4d} | "
        #     f"Proximity Adv: {self.proximity_advances:5d} | "
        #     f"Ideal Range: {self.ideal_range_tracking:5d} | "
        #     f"Near Collisions: {self.near_collision_events:4d} | "
        #     f"Step Hdg Dev: {step_avg_heading:5.1f}° | "
        #     f"Cum Hdg Dev: {cum_avg_heading:5.1f}°"
        # )

        # Log metrics to CSV if all agents terminate/truncate
        all_done = all(terminations.values()) or all(truncations.values())
        if all_done:
            self._log_run_to_csv()

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
        if self.step_counter > 0:
            self._log_run_to_csv()
            self.step_counter = 0
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()