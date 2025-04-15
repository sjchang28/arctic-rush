# Renders grid
# Stores wall data

import pygame
from config import TILE_SIZE, GRID_WIDTH, GRID_HEIGHT, GREY, BLACK

# Directions for readability
UP, RIGHT, DOWN, LEFT = 'up', 'right', 'down', 'left'

class Board:
    def __init__(self):
        # Initialize a 2D grid with empty wall data for each tile
        # Each cell is a dict with {direction: True/False}
        self.walls = [[{UP: False, RIGHT: False, DOWN: False, LEFT: False}
                       for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        self._add_border_walls()
        self._add_internal_walls()

    def _add_border_walls(self):
        # Outer border
        for x in range(GRID_WIDTH):
            self.walls[0][x][UP] = True
            self.walls[GRID_HEIGHT - 1][x][DOWN] = True
        for y in range(GRID_HEIGHT):
            self.walls[y][0][LEFT] = True
            self.walls[y][GRID_WIDTH - 1][RIGHT] = True

    def _add_internal_walls(self):
        # Add custom walls for testing (you can load this from a file later)
        self.add_wall(5, 5, RIGHT)
        self.add_wall(5, 6, LEFT)

        self.add_wall(7, 10, DOWN)
        self.add_wall(8, 10, UP)

    def add_wall(self, row, col, direction):
        self.walls[row][col][direction] = True

        # Mirror wall in adjacent cell
        if direction == UP and row > 0:
            self.walls[row - 1][col][DOWN] = True
        elif direction == DOWN and row < GRID_HEIGHT - 1:
            self.walls[row + 1][col][UP] = True
        elif direction == LEFT and col > 0:
            self.walls[row][col - 1][RIGHT] = True
        elif direction == RIGHT and col < GRID_WIDTH - 1:
            self.walls[row][col + 1][LEFT] = True

    def can_move(self, row, col, direction):
        """Returns True if no wall blocks the movement in that direction."""
        return not self.walls[row][col][direction]

    def draw(self, surface):
        # Draw grid lines
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, GREY, rect, 1)

        # Draw walls
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x, y = col * TILE_SIZE, row * TILE_SIZE

                if self.walls[row][col][UP]:
                    pygame.draw.line(surface, BLACK, (x, y), (x + TILE_SIZE, y), 3)
                if self.walls[row][col][RIGHT]:
                    pygame.draw.line(surface, BLACK, (x + TILE_SIZE, y), (x + TILE_SIZE, y + TILE_SIZE), 3)
                if self.walls[row][col][DOWN]:
                    pygame.draw.line(surface, BLACK, (x, y + TILE_SIZE), (x + TILE_SIZE, y + TILE_SIZE), 3)
                if self.walls[row][col][LEFT]:
                    pygame.draw.line(surface, BLACK, (x, y), (x, y + TILE_SIZE), 3)
