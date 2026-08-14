# Renders grid
# Stores wall data

import json
import os

import numpy as np
import pygame

from src.core.logger import logger
from src.game.config import (
    BLACK,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DOWN,
    GREY,
    LEFT,
    LEVEL_FILE,
    RIGHT,
    TILE_SIZE,
    UP,
)

current_dir = os.path.dirname(__file__)


def default_level_file():
    """Absolute path to the configured level.

    Resolved on call rather than at import. As a default argument this was
    evaluated once when the module first loaded, which froze `LEVEL_FILE` for the
    life of the process and made the level impossible to change at runtime (or
    from a test).
    """

    return os.path.join(current_dir, "levels", LEVEL_FILE)


class BouncePadManager:

    def __init__(self, bounce_pad_file=None):
        self.bounce_pads = dict()

        if bounce_pad_file is None:
            bounce_pad_file = default_level_file()

        self._load_bounce_pads(bounce_pad_file=bounce_pad_file)
        
        
    def _load_bounce_pads(self, bounce_pad_file):
        
        if not os.path.exists(bounce_pad_file):
            logger.error(f"Bounce Pad file '{bounce_pad_file}' not found.")
            return

        with open(bounce_pad_file, "r") as f:
            data = json.load(f)

        for pad in data.get("bounce_pads", []):
            self.bounce_pads[(pad["col"], pad["row"])] = {
                'orientation': pad["orientation"],
                'color': pad["color"],
                'redirect': {
                    UP: pad[UP],
                    RIGHT: pad[RIGHT],
                    DOWN: pad[DOWN],
                    LEFT: pad[LEFT]
                }
            }
    
        
    def handle_bounce_pad(self, x, y, incoming_direction, robot_color):
        # Return a new direction if pad applies, otherwise return None
        pad = self.bounce_pads.get((x, y))
        if pad and pad['color'] == robot_color:
            return pad['redirect'].get(incoming_direction)  # e.g., 'up' → 'left'
        return None


    def draw_bounce_pads(self, screen):
        if not self.bounce_pads:
            return
        
        for (x, y), pad in self.bounce_pads.items():
            self.draw_single_bounce_pad(screen, x, y, pad['color'], pad['orientation'])

    def draw_single_bounce_pad(self, screen, x, y, color, orientation):
        center_x = x * TILE_SIZE + TILE_SIZE // 2
        center_y = y * TILE_SIZE + TILE_SIZE // 2

        # Define thin rectangle (width × height)
        width = TILE_SIZE // 8
        height = TILE_SIZE

        # Determine angle from incoming -> launch direction
        dir_to_angle = {
            "bl_tr": -45,
            "tl_br": 45,
        }

        angle = dir_to_angle.get(orientation)
        if angle is None:
            return  # Skip if direction combo is unsupported

        # Create a surface for the bounce pad
        pad_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(pad_surf, pygame.Color(color), pad_surf.get_rect())

        # Rotate the surface
        rotated = pygame.transform.rotate(pad_surf, angle)

        # Adjust position to center
        rect = rotated.get_rect(center=(center_x, center_y))
        screen.blit(rotated, rect)

    
    
class Board:
    
    def __init__(self, wall_file=None):

        if wall_file is None:
            wall_file = default_level_file()

        # Initialize a 2D grid with empty wall data for each tile
        # Each cell is a dict with {direction: True/False}
        self.walls = [[{UP: False, RIGHT: False, DOWN: False, LEFT: False}
                       for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

        self._add_border_walls()
        self._load_custom_walls(wall_file=wall_file)

        # Walls and pads describe the same level and must come from the same
        # file. Constructed with no argument, this silently read the module
        # default instead, so Board(wall_file=X) mixed X's walls with the
        # default level's pads.
        self.bounce_pad_manager = BouncePadManager(wall_file)
    
    
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
            logger.error(f"Wall file '{wall_file}' not found.")
            return

        with open(wall_file, "r") as f:
            data = json.load(f)

        for wall in data.get("walls", []):
            row, col, direction = wall["row"], wall["col"], wall["dir"]
            self.add_wall(row, col, direction)


    def add_wall(self, row, col, direction):
        """Raise a wall on one side of one cell.

        Deliberately does *not* mirror the wall into the neighbouring cell. A
        commented-out mirroring block used to sit here, which read as an
        unfinished feature -- as though walls were one-sided and robots could
        slip through them from the far side.

        They are not: every level file already lists both halves of every wall
        (verified across all four levels -- 0 unpaired). Mirroring here would be
        redundant, and re-enabling it would silently change the physics the
        current checkpoints were trained against.

        `tests/test_board.py` pins the level-data symmetry this relies on, since
        it is now the only thing making walls two-sided.
        """

        self.walls[row][col][direction] = True


    def can_move(self, col, row, direction):
        
        """Returns True if no wall blocks the movement in that direction."""
        
        return not self.walls[row][col][direction]
        

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

