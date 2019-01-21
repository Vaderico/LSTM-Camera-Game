import torch
from torchvision import transforms

import contextlib
import numpy as np
from PIL import Image
with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *

class Player:
    def __init__(self):
        # Current player velocities
        self._x_vel = 0
        self._y_vel = 0

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

    # Do nothing when handling event by default
    def handle_event(self, event):
        pass

    def reset(self):
        pass

# Player controlled by user to manually move the camera in game
class LSTMPlayer(Player):
    def __init__(self, model_path):
        Player.__init__(self)
        self._model = torch.load(model_path)

        self._model.init_hidden()

        self._frame_count = 1

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def reset(self):
        self._model.init_hidden()
        self.reset_velocities()
        self._frame_count = 1

    # Updates the x and y velocity for the next time step
    def step_velocity(self, frame):
        img_string = pygame.image.tostring(frame, "RGB", False)
        img_pil = Image.frombytes("RGB",(224,224), img_string)

        img = np.array(img_pil)
        X = torch.tensor(self._transform(img)).cuda()
        X = X.view((1,3,224,224))

        with torch.no_grad():
            vel = self._model(X)[0]

        self._x_vel = vel[0].item()
        self._y_vel = vel[1].item()

        print("%0.4f, %0.4f, frame: %d" % (self._x_vel, self._y_vel, self._frame_count))
        self._frame_count += 1

        return self._x_vel, self._y_vel

# Player controlled by user to manually move the camera in game
class HumanPlayer(Player):
    def __init__(self, max_vel, accel, decel):
        Player.__init__(self)
        # Maximum speed the player can move
        self._max_vel = max_vel
        # Acceleration when moving forward
        self._accel = accel
        # Deceleration when stopping
        self._decel = decel

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


