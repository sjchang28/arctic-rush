# robot class
import random

import numpy as np
import pygame

from src.game.config import (
    ALL_DIRECTIONS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    COLOR_MAP,
    DOWN,
    GREY,
    LEFT,
    PINK,
    RIGHT,
    ROBOT_COLORS,
    SWITCH,
    TILE_SIZE,
    UP,
    center_squares,
)


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

    def _initialize_robot_positions(self, force: bool = False, forbidden_extra=()):
        """Place robots on random free squares.

        `force=True` re-rolls robots that already have a position. Without it this
        is a no-op on an env that has already been played, which is why
        `RicochetRobotsEnv.reset` could not actually reset.

        `forbidden_extra` keeps robots off squares the caller wants left empty --
        the current target, so an episode cannot start already solved.
        """
        forbidden_positions = set(center_squares())  # Avoid center
        forbidden_positions.update(forbidden_extra)
        available_positions = [
            (x, y)
            for x in range(BOARD_WIDTH)
            for y in range(BOARD_HEIGHT)
            if (x, y) not in forbidden_positions
        ]

        random.shuffle(available_positions)
        for robot, position in zip(self.robots, available_positions):
            if force or robot.x is None or robot.y is None:
                robot.x, robot.y = position
                robot.prev_x, robot.prev_y = position
                robot.robotLeftTarget = False

    def get_number_of_robots(self):
        return len(self.robots)

    def reset_robots(self):
        for robot in self.robots:
            robot.reset_robot_target()

    def _predecessor_squares(self, robot, direction):
        """Squares from which `robot` would slide to its current square going `direction`.

        Walks backwards along the incoming ray, stopping at the first wall or
        occupied cell. Squares behind a bounce pad are excluded: a pad rewrites
        the direction mid-slide, so a plain backward walk is not its inverse.
        """

        opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}[direction]
        dx, dy = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}[opposite]

        occupied = {(r.x, r.y) for r in self.robots}
        pads = self.board.bounce_pad_manager.bounce_pads

        squares = []
        x, y = robot.x, robot.y

        while True:
            new_x, new_y = x + dx, y + dy

            if not (0 <= new_x < BOARD_WIDTH and 0 <= new_y < BOARD_HEIGHT):
                break
            # The forward move (new -> current) must not be blocked by a wall.
            if not self.board.can_move(new_x, new_y, direction):
                break
            if (new_x, new_y) in occupied or (new_x, new_y) in pads:
                break

            squares.append((new_x, new_y))
            x, y = new_x, new_y

        return squares

    def reverse_scramble(self, num_moves, rng=random, required_first=None,
                         solver_index=None, solver_bias=0.0, avoid_square=None):
        """Walk the board backwards from a solved position.

        Reverse curriculum: rather than hoping random play stumbles onto the goal,
        generate positions a known number of moves away from it and grow that
        number as the agent improves. The value net gets a usable gradient long
        before forward exploration would ever find the target.

        A move is only reversible if the robot is currently *blocked* in that
        direction -- otherwise it would not have stopped where it is.

        `required_first` pins the first backward move to one robot. The caller
        needs that: if the very first backward move displaces some *other* robot,
        the goal robot is left standing on the target and the generated position
        is degenerate rather than one move from solved.

        `solver_bias` is the probability that any *later* backward move is also
        drawn from the goal robot's candidates, rather than from all four robots
        uniformly. Be aware of what this does and does not buy: measured with
        `game.solver.shortest_solution_length` on level_01, the output of this
        function is one move from solved roughly 85% of the time regardless of
        `num_moves` (tested 2 to 32) and regardless of this bias. Sliding is
        long-range, so a robot anywhere along the target's row or column is still
        one move out, and a backward random walk barely ever leaves that set.

        Treat this as a shallow-position generator, not a difficulty dial. The
        curriculum in `src.game.curriculum.CurriculumGenerator.generate` uses it
        only for depth 1-2 and gets the deeper levels by solving random positions
        instead.

        `avoid_square` is refused as a destination for the goal robot -- putting
        it back on the target would produce an already-solved position.
        """

        for move_number in range(num_moves):

            only_robot = required_first if move_number == 0 else None
            candidates, solver_candidates = self._backward_candidates(
                only_robot=only_robot, solver_index=solver_index,
                avoid_square=avoid_square,
            )

            if not candidates:
                break

            pool = candidates
            if solver_candidates and rng.random() < solver_bias:
                pool = solver_candidates

            index, square = pool[rng.randrange(len(pool))]
            robot = self.robots[index]
            robot.x, robot.y = square
            robot.prev_x, robot.prev_y = square
            robot.robotLeftTarget = False


    def _backward_candidates(self, only_robot=None, solver_index=None, avoid_square=None):
        """Every (robot, square) a single backward move could undo.

        Returns (all candidates, those belonging to the goal robot). `only_robot`
        restricts the search to one robot, which is how the first backward move
        is pinned to the goal robot.
        """

        candidates = []
        solver_candidates = []

        for index, robot in enumerate(self.robots):

            if only_robot is not None and index != only_robot:
                continue

            for direction in ALL_DIRECTIONS:

                # If the robot could still move this way, it cannot have
                # arrived here by sliding in this direction.
                if robot.move_until_blocked(
                    simulated=True, direction=direction,
                    board=self.board, other_robots=self.robots
                ):
                    continue

                for square in self._predecessor_squares(robot, direction):

                    if (index == solver_index and avoid_square is not None
                            and square == avoid_square):
                        continue

                    candidates.append((index, square))

                    if index == solver_index:
                        solver_candidates.append((index, square))

        return candidates, solver_candidates

    def get_all_legal_moves(self, selected_idx=None):
        """Legal (robot_index, direction) pairs.

        `selected_idx=None` (the AI path) returns moves only. Passing an index
        additionally offers SWITCH to every other robot, which is what the human
        pygame path models -- the AI action space has no SWITCH, so a robot index
        on a move action is honoured directly by the environment.
        """
        legal_moves = []

        for i, robot in enumerate(self.robots):
            for direction in ALL_DIRECTIONS:
                if robot.move_until_blocked(simulated=True, direction=direction, board=self.board, other_robots=self.robots):
                    legal_moves.append((i, direction))
            if selected_idx is not None and i != selected_idx:
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
