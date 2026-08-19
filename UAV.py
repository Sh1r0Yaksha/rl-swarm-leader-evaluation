import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

GRID_SIZE = int(os.getenv("GRID_SIZE", "150"))

class UAV:
    def __init__(
        self,
        speed: np.float32 = np.float32(1.0),# 1 units/frame
        angular_speed: np.float32 = np.float32(np.radians(4)),
        angular_direction: int = 0,
        position: np.ndarray = np.array([0.0, 0.0], dtype=np.float32), 
        orientation: np.float32 = np.float32(0.0),
    ):
        """
        Initializes a 2D UAV object.
        
        :param speed: Scalar velocity of the UAV.
        :param angular_speed: Scalar Angular velocity of the UAV.
        :param angular_direction: Direction of heading of UAV (right -1, left 1 or 0)
        :param position: 1D array of shape (2,) representing [x, y].
        :param orientation: Heading angle in radians (0 is East).
        """
        
        self.speed = np.float32(speed)
        self.position = np.array(position, dtype=np.float32)
        self._orientation = np.float32(0.0)
        self.orientation = orientation
        self.angular_speed = np.float32(angular_speed)
        self.angular_direction = angular_direction

    @property
    def orientation(self) -> np.float32:
        return self._orientation
    
    @orientation.setter
    def orientation(self, value):
        # Normalizes any angle to the range [-pi, pi]
        self._orientation = np.float32(((value + np.pi) % (2 * np.pi)) - np.pi)

    def step(self, dt: float):
        vx = self.speed * np.cos(self.orientation)
        vy = self.speed * np.sin(self.orientation)
        velocity = np.array([vx, vy], dtype=np.float32)

        self.position += velocity * dt
        self.orientation += self.angular_direction * self.angular_speed * dt

    def reset(
        self,
        position: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
        orientation: np.float32 = np.float32(0.0)
    ):
        self.position = np.array(position, dtype=np.float32)
        self.orientation = orientation
        self.angular_direction = 0

class Follower(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_dc: np.float32 = np.float32(0.0)
        self.prev_dl: np.float32 = np.float32(0.0)

    def reset_tracker(self, initial_dc: float, initial_dl: float):
        """Call this on environment reset with actual starting distances."""
        self.prev_dc = np.float32(initial_dc)
        self.prev_dl = np.float32(initial_dl)

class Leader(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_timer = 0  # Frame counter for smooth steering

    def step(self, dt: float, grid_w: float = GRID_SIZE, grid_h: float = GRID_SIZE):
        margin = 25.0  # Safe distance from boundary
        
        # 1. Wall avoidance check
        if (self.position[0] < margin or self.position[0] > grid_w - margin or
            self.position[1] < margin or self.position[1] > grid_h - margin):
            
            # Calculate angle pointing toward grid center
            target_angle = np.arctan2((grid_h / 2.0) - self.position[1], (grid_w / 2.0) - self.position[0])
            angle_diff = (target_angle - self.orientation + np.pi) % (2 * np.pi) - np.pi
            
            # Turn toward center
            if abs(angle_diff) < 0.1:
                self.angular_direction = 0
            elif abs(abs(angle_diff) - np.pi) < 0.1:
                # BREAK THE DEADLOCK: Force consistent left turn when facing directly away from center
                self.angular_direction = 1
            else:
                self.angular_direction = 1 if angle_diff > 0 else -1
                
            self.turn_timer = 0

        # 2. Random wander when safe
        elif self.turn_timer <= 0:
            rng = np.random.random()
            if rng < 0.40:
                self.angular_direction = -1   # Turn right
            elif rng > 0.60:
                self.angular_direction = 1    # Turn left
            else:
                self.angular_direction = 0    # Straight
            self.turn_timer = np.random.randint(15, 35)
        else:
            self.turn_timer -= 1

        # 3. Apply orientation and position physics
        super().step(dt)