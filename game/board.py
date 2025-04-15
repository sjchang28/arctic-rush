# Renders grid
# Stores wall data

import pygame
import json, os
from config import TILE_SIZE, GRID_WIDTH, GRID_HEIGHT, GREY, BLACK, UP, DOWN, LEFT, RIGHT

# Get absolute path to the level.json file
current_dir = os.path.dirname(__file__)
level_file = os.path.join(current_dir, "levels", "level_01.json")

class Board:
    def __init__(self, wall_file=level_file):
        # Initialize a 2D grid with empty wall data for each tile
        # Each cell is a dict with {direction: True/False}
        self.walls = [[{UP: False, RIGHT: False, DOWN: False, LEFT: False}
                       for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        self._add_border_walls()
        self._load_custom_walls(wall_file=wall_file)

    def _add_border_walls(self):
        # Outer border
        for x in range(GRID_WIDTH):
            self.walls[0][x][UP] = True
            self.walls[GRID_HEIGHT - 1][x][DOWN] = True
        for y in range(GRID_HEIGHT):
            self.walls[y][0][LEFT] = True
            self.walls[y][GRID_WIDTH - 1][RIGHT] = True
    
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
        # elif direction == DOWN and row < GRID_HEIGHT - 1:
        #     self.walls[row + 1][col][UP] = True
        # elif direction == LEFT and col > 0:
        #     self.walls[row][col - 1][RIGHT] = True
        # elif direction == RIGHT and col < GRID_WIDTH - 1:
        #     self.walls[row][col + 1][LEFT] = True

    def can_move(self, row, col, direction):
        """Returns True if no wall blocks the movement in that direction."""
        return not self.walls[row][col][direction]

    def draw_board(self, surface):
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

    def draw_target(screen, target_manager):
        x, y = target_manager.target_position
        color = target_manager.target_color
        label = target_manager.target_type

        # Get pixel position
        px = x * TILE_SIZE
        py = y * TILE_SIZE

        # Draw filled circle
        pygame.draw.circle(screen, pygame.Color(color), (px + TILE_SIZE // 2, py + TILE_SIZE // 2), TILE_SIZE // 2 - 4)

        # Draw label on top
        font = pygame.font.SysFont("Arial", 18, bold=True)
        text_surface = font.render(label, True, (0, 0, 0))  # Black text
        text_rect = text_surface.get_rect(center=(px + TILE_SIZE // 2, py + TILE_SIZE // 2))
        screen.blit(text_surface, text_rect)

