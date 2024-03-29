import matplotlib.pyplot as plt
plt.ion()
import pygame
from typing import Tuple

class Map():
    def __init__(self):
        self.x = []
        self.y = []
        self.MAP_DELAY = 20
        self.delay_counter = 0
        self.target = (None, None)
    def update_minimap(self, current_x : int, current_y : int) -> None:
        if self.delay_counter < self.MAP_DELAY:
            self.delay_counter += 1
            return
        plt.clf()
        self.delay_counter = 0
        plt.scatter(self.x, self.y, color='black')
        plt.scatter(current_x, current_y, color='blue')
        if self.target[0] is not None and self.target[1] is not None:
            plt.scatter(self.target[0], self.target[1], color='green')
        plt.axis('off')
        # print(f"Current position: ({current_x}, {current_y})")  
        plt.draw()
        plt.pause(0.01)

class Localizer():
    def __init__(self):
        self.current_x = 0
        self.current_y = 0
        self.heading = 0
        self.map = Map()
        self._was_holding_right = False
        self._was_holding_left = False
    def track(self, key_presses) -> None:
        # turning should be tracked once per key press while forward backwards need to be tracked while the key is held
        if key_presses[pygame.K_UP]:
            self._forward()
        if key_presses[pygame.K_DOWN]:
            self._backward()
        if key_presses[pygame.K_LEFT] and not self._was_holding_left:
            self._was_holding_left = True
            self.heading += 90 # turning is based off of intervals of 90 degrees which needs to be self verified by the player via visual inspection
            if self.heading >= 360:
                self.heading = self.heading % 360
        if key_presses[pygame.K_RIGHT] and not self._was_holding_right:
            self._was_holding_right = True
            self.heading -= 90
            if self.heading < 0:
                self.heading = 360 + self.heading
        self._was_holding_left = key_presses[pygame.K_LEFT]
        self._was_holding_right = key_presses[pygame.K_RIGHT]
    def _forward(self, navigation=False) -> None:
        if self.heading == 0:
            self.current_y += 1
        elif self.heading == 90:
            self.current_x -= 1
        elif self.heading == 180:
            self.current_y -= 1
        elif self.heading == 270:
            self.current_x += 1
        if not navigation:
            self.map.x.append(self.current_x)
            self.map.y.append(self.current_y)
    def _backward(self, navigation=False) -> None:
        if self.heading == 0:
            self.current_y -= 1
        elif self.heading == 90:
            self.current_x += 1
        elif self.heading == 180:
            self.current_y += 1
        elif self.heading == 270:
            self.current_x -= 1
        if not navigation:
            self.map.x.append(self.current_x)
            self.map.y.append(self.current_y)
    def get_pose(self) -> Tuple[int, int, int]:
        return (self.current_x, self.current_y, self.heading)