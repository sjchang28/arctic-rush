# penguin class
import numpy as np
import random
import pygame
from config import BOARD_WIDTH, BOARD_HEIGHT, TILE_SIZE, ALL_DIRECTIONS, SWITCH, PENGUIN_COLORS, COLOR_MAP, GREY, PINK

class Penguin:
    def __init__(self, color, x=None, y=None):
        self.color = color
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y

        self.penguinLeftTarget = False
        
        
    def __repr__(self):
        return f"Penguin-{self.color} @ ({self.x},{self.y})"
    
    def get_position(self):
        return tuple([self.x, self.y])
    
    def get_prev_position(self):
        return tuple([self.prev_x, self.prev_y])
    
    def update_position(self, new_x, new_y):
        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = new_x, new_y

    def reset_penguin_target(self): # reset penguin position on every new target
        self.penguinLeftTarget = False
        self.update_position(self.x, self.y)

    def move_until_blocked(self, simulated, direction, board, other_penguins):
        if direction not in ALL_DIRECTIONS:
            return False
               
        penguin_moved = False
        dx, dy = 0, 0
        if direction == 'up': dy = -1
        elif direction == 'down': dy = 1
        elif direction == 'left': dx = -1
        elif direction == 'right': dx = 1

        tmp_x, tmp_y = self.x, self.y
        while True:
            new_x = tmp_x + dx
            new_y = tmp_y + dy

            # Check walls
            if not board.can_move(tmp_x, tmp_y, direction):
                break

            # Check penguin collision
            if any(r.x == new_x and r.y == new_y for r in other_penguins if r != self):
                break

            # Move
            tmp_x, tmp_y = new_x, new_y
            penguin_moved = True

        if not simulated and penguin_moved:
            self.update_position(new_x=tmp_x, new_y=tmp_y)

        return penguin_moved

    def is_target_reached(self, target):        
        penguinOnTarget = self.get_position() == (target.x, target.y)
        penguinWasOnTarget = self.get_prev_position() == (target.x, target.y)

        # Track if penguin was forced to leave the target before reaching it officially
        if penguinWasOnTarget and not penguinOnTarget:
            self.penguinLeftTarget = True

        # Case 1: Normal — penguin moves onto the target
        if penguinOnTarget and not penguinWasOnTarget and not self.penguinLeftTarget:
            if target.color.upper() == "ANY" or target.color.lower() == self.color.lower():
                return True

        # Case 2: Moved off target and came back
        if penguinOnTarget and self.penguinLeftTarget:
            if target.color.upper() == "ANY" or target.color.lower() == self.color.lower():
                return True

        # Case 3: Spawned on the target and hasn't moved yet
        return False

    
    def draw(self, surface):
        # Calculate pixel position for center of the cell
        px = self.x * TILE_SIZE + TILE_SIZE // 2
        py = self.y * TILE_SIZE + TILE_SIZE // 2
        radius = TILE_SIZE // 2 - 4  # Padding for clean look

        # Draw the penguin
        pygame.draw.circle(surface, PENGUIN_COLORS.get(self.color, GREY), (px, py), radius)

class PenguinManager:
    def __init__(self, board, penguins):
        self.board = board
        self.penguins = penguins
        self._initialize_penguin_positions()
    
    def flatten_penguins(self):
        penguin_array = []

        for penguin in self.penguins:
            row, col = penguin.x, penguin.y
            color_idx = COLOR_MAP[penguin.color.lower()]
            penguin_array.extend([row, col, color_idx])

        np_penguins = np.array(penguin_array)
        return np_penguins

    def _initialize_penguin_positions(self):
        forbidden_positions = {(7, 7), (7, 8), (8, 7), (8, 8)} # Avoid center
        available_positions = [
            (x, y)
            for x in range(BOARD_WIDTH)
            for y in range(BOARD_HEIGHT)
            if (x, y) not in forbidden_positions
        ]

        random.shuffle(available_positions)
        for penguin, position in zip(self.penguins, available_positions):
            if penguin.x is None or penguin.y is None:
                penguin.x, penguin.y = position
    
    def get_number_of_penguins(self):
        return len(self.penguins)

    def reset_penguins(self):
        for penguin in self.penguins:
            penguin.reset_penguin_target()

    def get_all_legal_moves(self, selected_idx):
        legal_moves = []

        for i, penguin in enumerate(self.penguins):
            for direction in ALL_DIRECTIONS:
                if penguin.move_until_blocked(simulated=True, direction=direction, board=self.board, other_penguins=self.penguins):
                    legal_moves.append((i, direction))
            if i != selected_idx:
                legal_moves.append((i, SWITCH))

        return legal_moves

    def draw_penguins(self, surface, selected_idx=0):
        for i, penguin in enumerate(self.penguins):
            if i == selected_idx:
                # Full cell position in pixels
                px = penguin.x * TILE_SIZE
                py = penguin.y * TILE_SIZE

                # Target rectangle size (smaller than full cell)
                padding = 2
                targeted_penguin_size = TILE_SIZE - 2 * padding
                targeted_penguin_rect = pygame.Rect(px + padding, py + padding, targeted_penguin_size, targeted_penguin_size)

                # Draw the inner rectangle for the target
                pygame.draw.rect(surface, PINK, targeted_penguin_rect, 3)

            penguin.draw(surface)