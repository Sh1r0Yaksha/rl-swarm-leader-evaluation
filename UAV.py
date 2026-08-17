import numpy as np

class UAV:
    def __init__(
        self,
        speed: np.float32 = np.float32(240.0),  # 240 units/sec (4 units/frame at 60 FPS)
        angular_speed: np.float32 = np.float32(np.radians(200)),  # ~3.49 rad/s
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

    def step(self, dt: float, grid_w: float = 800.0, grid_h: float = 600.0):
        # Steer back toward center if nearing borders
        margin = 100.0
        if self.position[0] < margin or self.position[0] > grid_w - margin or \
           self.position[1] < margin or self.position[1] > grid_h - margin:
            target_angle = np.arctan2((grid_h/2) - self.position[1], (grid_w/2) - self.position[0])
            angle_diff = (target_angle - self.orientation + np.pi) % (2 * np.pi) - np.pi
            self.angular_direction = 1 if angle_diff > 0 else -1
            self.turn_timer = 30
        elif self.turn_timer <= 0:
            rng = np.random.random()
            if rng < 0.20: self.angular_direction = -1
            elif rng > 0.80: self.angular_direction = 1
            else: self.angular_direction = 0
            self.turn_timer = np.random.randint(60, 120)
        else:
            self.turn_timer -= 1

        super().step(dt)