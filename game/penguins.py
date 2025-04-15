# Robot class
import pygame
from config import TILE_SIZE, ROBOT_COLORS

class Robot:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y

    def draw(self, surface):
        center = (self.x * TILE_SIZE + TILE_SIZE // 2, self.y * TILE_SIZE + TILE_SIZE // 2)
        pygame.draw.circle(surface, ROBOT_COLORS[self.color], center, TILE_SIZE // 3)

    def move_until_blocked(self, direction, board, other_robots):
        dx, dy = 0, 0
        if direction == 'up': dy = -1
        elif direction == 'down': dy = 1
        elif direction == 'left': dx = -1
        elif direction == 'right': dx = 1

        while True:
            new_x = self.x + dx
            new_y = self.y + dy

            # Check walls
            if not board.can_move(self.y, self.x, direction):
                break

            # Check robot collision
            if any(r.x == new_x and r.y == new_y for r in other_robots if r != self):
                break

            # Move
            self.x, self.y = new_x, new_y
