class LSTMPlayer:
    def __init__(self):
        print("lstm player created")

        # Current player velocities
        self._x_vel = 1
        self._y_vel = 2

    def step_velocity(self, frame):
        return self._x_vel, self._y_vel

    def reset_velocities(self):
        self.reset_x_velocity()
        self.reset_y_velocity()

    def reset_x_velocity(self):
        self._x_vel *= -1
        
    def reset_y_velocity(self):
        self._y_vel *= -1

    def handle_event(self, event):
        pass
