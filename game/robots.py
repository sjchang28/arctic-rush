# robot class
import numpy as np
import random
import pygame
from config import BOARD_WIDTH, BOARD_HEIGHT, TILE_SIZE, ALL_DIRECTIONS, SWITCH, ROBOT_COLORS, COLOR_MAP, GREY, PINK

class Robot:
    def __init__(self, color, x=None, y=None):
        self.color = color
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y

        self.robotLeftTarget = False
        
        
    def __repr__(self):
        return f"Robot-{self.color} @ ({self.x},{self.y})"
    
    def get_position(self):
        return tuple([self.x, self.y])
    
    def get_prev_position(self):
        return tuple([self.prev_x, self.prev_y])
    
    def update_position(self, new_x, new_y):
        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = new_x, new_y

    def reset_robot_target(self): # reset robot position on every new target
        self.robotLeftTarget = False
        self.update_position(self.x, self.y)

    def move_until_blocked(self, simulated, direction, board, other_robots):
        if direction not in ALL_DIRECTIONS:
            return False
               
        robot_moved = False
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

            # Check robot collision
            if any(r.x == new_x and r.y == new_y for r in other_robots if r != self):
                break

            # Move
            tmp_x, tmp_y = new_x, new_y
            robot_moved = True
            
            # Check for bounce pad and update direction if needed
            new_direction = board.bounce_pad_manager.handle_bounce_pad(tmp_x, tmp_y, direction, self.color)
            if new_direction and new_direction != direction:
                # Update direction and deltas
                direction = new_direction
                dx, dy = 0, 0
                if direction == 'up': dy = -1
                elif direction == 'down': dy = 1
                elif direction == 'left': dx = -1
                elif direction == 'right': dx = 1


        if not simulated and robot_moved:
            self.update_position(new_x=tmp_x, new_y=tmp_y)

        return robot_moved

    def is_target_reached(self, target):        
        robotOnTarget = self.get_position() == (target.x, target.y)
        robotWasOnTarget = self.get_prev_position() == (target.x, target.y)

        # Track if robot was forced to leave the target before reaching it officially
        if robotWasOnTarget and not robotOnTarget:
            self.robotLeftTarget = True

        # Case 1: Normal — robot moves onto the target
        if robotOnTarget and not robotWasOnTarget and not self.robotLeftTarget:
            if target.color.upper() == "ANY" or target.color.lower() == self.color.lower():
                return True

        # Case 2: Moved off target and came back
        if robotOnTarget and self.robotLeftTarget:
            if target.color.upper() == "ANY" or target.color.lower() == self.color.lower():
                return True

        # Case 3: Spawned on the target and hasn't moved yet
        return False

    
    def draw(self, surface):
        # Calculate pixel position for center of the cell
        px = self.x * TILE_SIZE + TILE_SIZE // 2
        py = self.y * TILE_SIZE + TILE_SIZE // 2
        radius = TILE_SIZE // 2 - 4  # Padding for clean look

        # Draw the robot
        pygame.draw.circle(surface, ROBOT_COLORS.get(self.color, GREY), (px, py), radius)

class RobotManager:
    def __init__(self, board, robots):
        self.board = board
        self.robots = robots
        self._initialize_robot_positions()
    
    def flatten_robots(self):
        robot_array = []

        for robot in self.robots:
            row, col = robot.x, robot.y
            color_idx = COLOR_MAP[robot.color.lower()]
            robot_array.extend([row, col, color_idx])

        np_robots = np.array(robot_array)
        return np_robots

    def _initialize_robot_positions(self):
        forbidden_positions = {(7, 7), (7, 8), (8, 7), (8, 8)} # Avoid center
        available_positions = [
            (x, y)
            for x in range(BOARD_WIDTH)
            for y in range(BOARD_HEIGHT)
            if (x, y) not in forbidden_positions
        ]

        random.shuffle(available_positions)
        for robot, position in zip(self.robots, available_positions):
            if robot.x is None or robot.y is None:
                robot.x, robot.y = position
    
    def get_number_of_robots(self):
        return len(self.robots)

    def reset_robots(self):
        for robot in self.robots:
            robot.reset_robot_target()

    def get_all_legal_moves(self, selected_idx):
        legal_moves = []

        for i, robot in enumerate(self.robots):
            for direction in ALL_DIRECTIONS:
                if robot.move_until_blocked(simulated=True, direction=direction, board=self.board, other_robots=self.robots):
                    legal_moves.append((i, direction))
            if i != selected_idx:
                legal_moves.append((i, SWITCH))

        return legal_moves

    def draw_robots(self, surface, selected_idx=0):
        for i, robot in enumerate(self.robots):
            if i == selected_idx:
                # Full cell position in pixels
                px = robot.x * TILE_SIZE
                py = robot.y * TILE_SIZE

                # Target rectangle size (smaller than full cell)
                padding = 2
                targeted_robot_size = TILE_SIZE - 2 * padding
                targeted_robot_rect = pygame.Rect(px + padding, py + padding, targeted_robot_size, targeted_robot_size)

                # Draw the inner rectangle for the target
                pygame.draw.rect(surface, PINK, targeted_robot_rect, 3)

            robot.draw(surface)