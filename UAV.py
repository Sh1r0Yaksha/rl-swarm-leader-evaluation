import numpy as np
from numpy._core.multiarray import array

class UAV:
    def __init__(
        self,
        speed: np.float32 = np.float32(50.0),
        angular_speed: np.float32 = np.float32(3),
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

    def step(self, dt):
        vx = self.speed * np.cos(self.orientation)
        vy = self.speed * np.sin(self.orientation)
        velocity = np.array([vx, vy], dtype=np.float32)

        self.position += velocity * dt

        self.orientation += self.angular_direction * self.angular_speed * dt

    def reset(self,
              position: np.ndarray = np.array([0.0, 0.0], dtype=np.float32),
              orientation: np.float32 = np.float32(0.0)):
        self.position = np.array(position, dtype=np.float32)
        self.orientation = orientation
        self.angular_direction = 0

    def __repr__(self):
        return f"UAV(pos={self.position}, speed={self.speed}, hdg={self.orientation})"

class Follower(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_dc: np.float32 = np.float32(0.0)
        self.prev_dl: np.float32 = np.float32(0.0)

class Leader(UAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, dt):
        rng = np.random.random()
        if rng < 0.25: self.angular_direction = -1
        elif rng > 0.85: self.angular_direction = 1
        else: self.angular_direction = 0
        return super().step(dt)