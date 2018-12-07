import contextlib
with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *

# Player controlled by user to manually move the camera in game
class HumanPlayer():
    def __init__(self, max_vel, accel, decel):
        # Maximum speed the player can move
        self._max_vel = max_vel
        # Acceleration when moving forward
        self._accel = accel
        # Deceleration when stopping
        self._decel = decel

        # Current player velocities
        self._x_vel = 0
        self._y_vel = 0

        # States of arrow keys
        self._up_pressed = False
        self._down_pressed = False
        self._left_pressed = False
        self._right_pressed = False

    # Updates the x and y velocity for the next time step
    def step_velocity(self, frame):
        self._x_vel = self.calculate_velocity(
                self._x_vel, 
                self._right_pressed,
                self._left_pressed)
        self._y_vel = self.calculate_velocity(
                self._y_vel, 
                self._down_pressed,
                self._up_pressed)

        return self._x_vel, self._y_vel

    # Set x and y velocities to 0
    def reset_velocities(self):
        self.reset_x_velocity()
        self.reset_y_velocity()

    # Set x velocity to 0
    def reset_x_velocity(self):
        self._x_vel = 0

    # Set y velocity to 0
    def reset_y_velocity(self):
        self._y_vel = 0

    # Calculates the new velocity along a particular axis
    def calculate_velocity(self, vel, pos_pressed, neg_pressed):
        # If the player exclusively moving positively or negatively along an axis
        if pos_pressed != neg_pressed:
            if pos_pressed:
                # Positive pressed, add velocity in the positive direction
                accel = self._decel if vel < 0 else self._accel
                new_vel = vel + accel
                return min(new_vel, self._max_vel)
            elif neg_pressed:
                # Negative pressed, add velocity in the negative direction
                accel = self._decel if vel > 0 else self._accel
                new_vel = vel - accel
                return max(new_vel, self._max_vel * -1)
        else:
            # Stopping, decelerating in the opposite direction as current velocity
            if vel < 0:
                new_vel = vel + self._decel
                return min(0, new_vel)
            elif vel > 0:
                new_vel = vel - self._decel
                return max(0, new_vel)
        return vel

    # Handles keyboard events that move player with arrow keys
    def handle_event(self, event):
        if event.type == KEYDOWN:
            if event.key == K_UP:
                self.set_up(True)
            if event.key == K_DOWN:
                self.set_down(True)
            if event.key == K_LEFT:
                self.set_left(True)
            if event.key == K_RIGHT:
                self.set_right(True)
        if event.type == KEYUP:
            if event.key == K_UP:
                self.set_up(False)
            if event.key == K_DOWN:
                self.set_down(False)
            if event.key == K_LEFT:
                self.set_left(False)
            if event.key == K_RIGHT:
                self.set_right(False)

    # Update up state
    def set_up(self, state):
        self._up_pressed = state

    # Update down state
    def set_down(self, state):
        self._down_pressed = state

    # Update left state
    def set_left(self, state):
        self._left_pressed = state

    # Update right state
    def set_right(self, state):
        self._right_pressed = state

