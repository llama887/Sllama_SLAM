import matplotlib.pyplot as plt
plt.ion()
import pygame
from typing import Tuple
import pickle

class Map():
    def __init__(self):
        self.x = []
        self.y = []
        self.target = (None, None)
        self.mapped_x = None
        self.mapped_y = None
        self.delay_counter = 0
        self.MAP_DELAY = 10
        self.store_counter = 0
        self.STORE_DELAY = 50
    def update_minimap(self, current_x : int, current_y : int, heading) -> None:
        if self.delay_counter < self.MAP_DELAY:
            self.delay_counter += 1
        else:
            self.delay_counter = 0
            if heading == 0:
                forward = (current_x, current_y + 5)
            elif heading == 90:
                forward = (current_x - 5, current_y)
            elif heading == 180:
                forward = (current_x, current_y - 5)
            elif heading == 270:
                forward = (current_x + 5, current_y)
            plt.clf()
            plt.scatter(self.x, self.y, color='black')
            if self.target[0] is not None and self.target[1] is not None:
                plt.scatter(self.target[0], self.target[1], color='green')
            if self.mapped_x != None and self.mapped_y != None:
                plt.scatter(self.mapped_x, self.mapped_y, color='red', marker="P")
            plt.scatter(current_x, current_y, color='blue')
            plt.scatter(forward[0], forward[1], color='orange', marker="x")
            plt.axis('off')
            plt.draw()
            plt.pause(0.01)
        if self.store_counter < self.STORE_DELAY:
            self.store_counter += 1
            return
        self.store_counter = 0
        self.save_poses()
        
    def save_poses(self):
        with open('x.pickle','wb') as f:
            pickle.dump(self.x, f)
        with open('y.pickle', 'wb') as f:
            pickle.dump(self.y, f)
    def load_poses(self):
        with open('x.pickle', 'rb') as f:
            self.x = pickle.load(f)
        with open('y.pickle', 'rb') as f:
            self.y = pickle.load(f)


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