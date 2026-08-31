import numpy as np
import os
from dotenv import load_dotenv
from UAV import Follower, Leader

load_dotenv()

class ExtendedBoidModel:
    """
    Rule-based Extended Boid Model based on Reynolds' flocking principles:
    - Separation (avoid collision / maintain d1)
    - Cohesion (move towards flock centroid)
    - Alignment & Following (align and steer towards Leader)
    """
    def __init__(self, followers: dict[str, Follower], leader: Leader):
        self.followers = followers
        self.leader = leader
        
        # Distance thresholds matching the environment setup
        self.d1 = float(os.getenv("D1", "5"))
        self.d2 = float(os.getenv("D2", "10"))
        self.d3 = float(os.getenv("D3", "20"))
        
        # Rule weighting factors
        self.w_sep = 3.0
        self.w_coh = 1.0
        self.w_ali = 2.0
        self.w_fol = 2.5

    def get_action(self, agent_id: str) -> int:
        uav = self.followers[agent_id]
        
        v_sep = np.zeros(2, dtype=np.float32)
        v_coh = np.zeros(2, dtype=np.float32)
        v_ali = np.zeros(2, dtype=np.float32)
        v_fol = np.zeros(2, dtype=np.float32)

        # 1. Separation & Cohesion against other followers
        neighbor_count = 0
        positions = []
        for other_id, other_uav in self.followers.items():
            if other_id == agent_id:
                continue
            
            diff = uav.position - other_uav.position
            dist = np.linalg.norm(diff)
            positions.append(other_uav.position)

            # Separation rule (too close < d1)
            if 0 < dist < self.d1:
                v_sep += (diff / (dist + 1e-5))

            if dist < self.d3:
                neighbor_count += 1

        # Cohesion towards neighbor centroid
        if len(positions) > 0:
            centroid = np.mean(positions, axis=0)
            diff_coh = centroid - uav.position
            dist_coh = np.linalg.norm(diff_coh)
            if dist_coh > 0:
                v_coh = diff_coh / dist_coh

        # 2. Alignment & Following towards the Leader
        leader_diff = self.leader.position - uav.position
        leader_dist = np.linalg.norm(leader_diff)
        
        if leader_dist < self.d1:
            v_sep += (uav.position - self.leader.position) / (leader_dist + 1e-5)
        elif leader_dist > 0:
            v_fol = leader_diff / leader_dist

        # Leader Alignment
        v_ali = np.array([np.cos(self.leader.orientation), np.sin(self.leader.orientation)], dtype=np.float32)

        # 3. Combine desired steering force vectors
        desired_vector = (
            self.w_sep * v_sep +
            self.w_coh * v_coh -
            self.w_ali * v_ali -
            self.w_fol * v_fol
        )

        if np.linalg.norm(desired_vector) == 0:
            return 2  # Action 2: Maintain heading

        # 4. Map target angle to discrete angular actions (0: Turn Right, 1: Turn Left, 2: Straight)
        target_angle = np.arctan2(desired_vector[1], desired_vector[0])
        angle_diff = (target_angle - uav.orientation + np.pi) % (2 * np.pi) - np.pi

        if angle_diff > 0.1:
            return 1  # Turn Left
        elif angle_diff < -0.1:
            return 0  # Turn Right
        else:
            return 2  # Go Straight

    def predict_actions(self) -> dict[str, int]:
        """Returns action dictionary for all active followers."""
        return {agent_id: self.get_action(agent_id) for agent_id in self.followers.keys()}