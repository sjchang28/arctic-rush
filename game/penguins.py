# penguin class
import pygame
from config import TILE_SIZE, PENGUIN_COLORS

class Penguin:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y

    def draw(self, surface):
        # Calculate pixel position for center of the cell
        px = self.x * TILE_SIZE + TILE_SIZE // 2
        py = self.y * TILE_SIZE + TILE_SIZE // 2
        radius = TILE_SIZE // 2 - 4  # Padding for clean look

        # Draw the penguin
        pygame.draw.circle(surface, PENGUIN_COLORS.get(self.color, (200, 200, 200)), (px, py), radius)

    def move_until_blocked(self, direction, board, other_penguins):
        penguin_moved = False
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

            # Check penguin collision
            if any(r.x == new_x and r.y == new_y for r in other_penguins if r != self):
                break

            # Move
            self.x, self.y = new_x, new_y
            penguin_moved = True
        return penguin_moved
