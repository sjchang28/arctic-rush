# Renders grid
# Stores wall data

import pygame
import json, os, numpy as np
from config import TILE_SIZE, BOARD_WIDTH, BOARD_HEIGHT, GREY, BLACK, UP, DOWN, LEFT, RIGHT

# Get absolute path to the level.json file
current_dir = os.path.dirname(__file__)
level_file = os.path.join(current_dir, "levels", "level_01.json")

class Board:
    
    def __init__(self, wall_file=level_file):
        
        # Initialize a 2D grid with empty wall data for each tile
        # Each cell is a dict with {direction: True/False}
        self.walls = [[{UP: False, RIGHT: False, DOWN: False, LEFT: False}
                       for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

        self._add_border_walls()
        self._load_custom_walls(wall_file=wall_file)
    
    
    def flatten_walls(self):
        
        flat_array = []

        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                cell = self.walls[row][col]
                flat_array.extend([
                    int(cell[UP]),
                    int(cell[RIGHT]),
                    int(cell[DOWN]),
                    int(cell[LEFT]),
                ])

        flat_np = np.array(flat_array)
        return flat_np


    def _add_border_walls(self):
        
        # Outer border
        for x in range(BOARD_WIDTH):
            self.walls[0][x][UP] = True
            self.walls[BOARD_HEIGHT - 1][x][DOWN] = True
        for y in range(BOARD_HEIGHT):
            self.walls[y][0][LEFT] = True
            self.walls[y][BOARD_WIDTH - 1][RIGHT] = True
    
    
    def _load_custom_walls(self, wall_file):
        
        if not os.path.exists(wall_file):
            print(f"❌ Wall file '{wall_file}' not found.")
            return

        with open(wall_file, "r") as f:
            data = json.load(f)

        for wall in data.get("walls", []):
            row, col, direction = wall["row"], wall["col"], wall["dir"]
            self.add_wall(row, col, direction)


    def add_wall(self, row, col, direction):
        
        self.walls[row][col][direction] = True

        # Mirror wall in adjacent cell
        # if direction == UP and row > 0:
        #     self.walls[row - 1][col][DOWN] = True
        # elif direction == DOWN and row < BOARD_HEIGHT - 1:
        #     self.walls[row + 1][col][UP] = True
        # elif direction == LEFT and col > 0:
        #     self.walls[row][col - 1][RIGHT] = True
        # elif direction == RIGHT and col < BOARD_WIDTH - 1:
        #     self.walls[row][col + 1][LEFT] = True


    def can_move(self, col, row, direction):
        
        """Returns True if no wall blocks the movement in that direction."""
        
        return not self.walls[row][col][direction]
    
    def is_wall(self, x, y, direction):
        
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return True  # Treat out-of-bounds as walls
        return self.walls[y][x].get(direction, False)
    
    
    def next_cell(self, x, y, direction):
        
        if direction == UP:
            return x, y - 1
        elif direction == DOWN:
            return x, y + 1
        elif direction == LEFT:
            return x - 1, y
        elif direction == RIGHT:
            return x + 1, y
        return x, y


    def draw_board(self, surface):
        
        # Draw grid lines
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(surface, GREY, rect, 1)

        # Draw walls
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                x, y = col * TILE_SIZE, row * TILE_SIZE

                if self.walls[row][col][UP]:
                    pygame.draw.line(surface, BLACK, (x, y), (x + TILE_SIZE, y), 3)
                if self.walls[row][col][RIGHT]:
                    pygame.draw.line(surface, BLACK, (x + TILE_SIZE, y), (x + TILE_SIZE, y + TILE_SIZE), 3)
                if self.walls[row][col][DOWN]:
                    pygame.draw.line(surface, BLACK, (x, y + TILE_SIZE), (x + TILE_SIZE, y + TILE_SIZE), 3)
                if self.walls[row][col][LEFT]:
                    pygame.draw.line(surface, BLACK, (x, y), (x, y + TILE_SIZE), 3)

