import numpy as np

class UAV:
    def __init__(
        self,
        speed: np.float32 = 50,
        angular_speed: np.float32 = np.float32(0.0),
        position: np.ndarray = np.array([0.0, 0.0], dtype=np.float32), 
        orientation: np.float32 = np.float32(0.0)
    ):
        """
        Initializes a 2D UAV object.
        
        :param speed: Scalar velocity of the UAV.
        :param angular_speed: Scalar Angular velocity of the UAV.
        :param position: 1D array of shape (2,) representing [x, y].
        :param orientation: Heading angle in radians (0 is East).
        """
        
        self.speed = np.float32(speed)
        self.position = np.array(position, dtype=np.float32)
        self._orientation = np.float32(0.0)
        self.orientation = orientation
        self.angular_speed = np.float32(angular_speed)

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

        self.orientation += self.angular_speed * dt

    def __repr__(self):
        return f"UAV(pos={self.position}, speed={self.speed}, hdg={self.orientation})"