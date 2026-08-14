import collections
import random

import gymnasium
from gymnasium import spaces
import numpy as np

from src.game.robots import Robot
from src.game.game import AI_Game
from src.game.solver import SEARCH_EXHAUSTED, shortest_solution_length

from src.config import (
    AI_ACTION_SPACE_SIZE,
    AI_OBSERVATION_SHAPE,
    ALL_DIRECTIONS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    COLOR_MAP,
    INT2DIRECTION,
    NUMBER_OF_DIRECTIONS,
    NUMBER_OF_ROBOTS,
    REWARD_PER_MOVE,
    REWARD_REPEAT_STATE,
    REWARD_SOLVE,
    settings,
)

def build_observation(wall_planes, robot_positions, target, move_counter):
    """Assemble the observation planes from primitive state.

    Kept free of the env object so a stored trajectory can be re-rendered under a
    *different* target -- which is what hindsight relabelling needs, and what
    lets a game keep positions rather than 13 baked planes per step.

    Args:
        wall_planes: (4, H, W) float32, constant for the level.
        robot_positions: [(x, y), ...] indexed by robot colour.
        target: (x, y, colour_index); colour_index >= NUMBER_OF_ROBOTS means "any".
        move_counter: moves played so far.
    """

    robot_planes = np.zeros((NUMBER_OF_ROBOTS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for color_index, (x, y) in enumerate(robot_positions):
        robot_planes[color_index, y, x] = 1.0

    target_x, target_y, target_color = target
    target_planes = np.zeros((NUMBER_OF_ROBOTS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    if target_color >= NUMBER_OF_ROBOTS:
        # "any" -- every colour satisfies this target.
        target_planes[:, target_y, target_x] = 1.0
    else:
        target_planes[target_color, target_y, target_x] = 1.0

    move_plane = np.full(
        (1, BOARD_HEIGHT, BOARD_WIDTH),
        move_counter / max(settings.MAX_TOTAL_MOVES_PER_GAME, 1),
        dtype=np.float32,
    )

    return np.concatenate([wall_planes, robot_planes, target_planes, move_plane], axis=0)


class RicochetRobotsEnv(gymnasium.Env):

    """
    Custom Environment for Ricochet robots compatible with OpenAI Gym.

    The agent controls all four robots directly: action `i * 4 + d` moves robot
    `i` in direction `d` until it is blocked. There is no selected-robot state and
    no SWITCH action -- those belong to the human pygame path, where a player must
    tab to a robot before moving it. Modelling them here cost a 4x-redundant
    action space and let the agent burn its move budget on no-ops.
    """

    def __init__(self, render_ai : bool=False):

        super().__init__()

        # Init board
        self.robots = [
            Robot("red"),
            Robot("blue"),
            Robot("green"),
            Robot("yellow")
        ]

        self.render_ai = render_ai
        self.game = AI_Game(self.robots, render_pygame=self.render_ai)

        self.visited_states = set()

        self.move_counter = 0

        # Reverse-curriculum depth: 0 generates fully random start positions.
        # The training loop ramps this up as the solved rate climbs.
        self.curriculum_moves = settings.CURRICULUM_START_MOVES

        # Measured optimal solution length of the position the episode started
        # from. None when it was not measured (curriculum off, or verification
        # disabled). This -- not the scramble depth -- is the honest difficulty.
        self.last_optimal_depth = None

        # Verified start positions, keyed by optimal depth. Solving a candidate
        # exactly costs milliseconds at depth 3 but seconds at depth 6, which is
        # more than a whole episode; without reuse, verification would dominate
        # the run. Positions are kept with the target they were verified against,
        # since difficulty is a property of the pair.
        self._depth_pool = {}

        # Action space: 4 robots x 4 directions.
        self.action_space = spaces.Discrete(AI_ACTION_SPACE_SIZE)

        # Observation space: stacked board planes (see config.AI_OBSERVATION_SHAPE)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=AI_OBSERVATION_SHAPE, dtype=np.float32
        )

        # Wall planes never change within a level, so build them once.
        self._wall_planes = self._build_wall_planes()

        self.reset()


    def set_curriculum_moves(self, moves: int):

        """Ramp the reverse-curriculum depth. Takes effect on the next reset."""

        self.curriculum_moves = max(0, int(moves))


    def reset(self):

        # Pick the target first so robots can be kept off it -- starting an episode
        # already standing on the goal would hand out a free win.
        self.game.target_deck.set_new_target()

        target = self.game.target_deck.current_target

        if self.curriculum_moves > 0:
            self._reset_from_curriculum(target)
        else:
            self._random_once(target)
            self.last_optimal_depth = None

        self.visited_states.clear()
        self.move_counter = 0

        return self._get_obs()


    def measure_optimal_depth(self, max_depth=None):
        """True optimal solution length of the current position, or None.

        `None` means the bounded search found nothing -- the position is deeper
        than `max_depth`, or deeper than the node budget could confirm. Read it
        as "deeper than that", never as "unsolvable".

        `max_depth` is worth passing whenever the caller only needs to know
        whether a position is a given depth rather than exactly how deep it is.
        BFS here branches by 16 and costs an order of magnitude more per level,
        so bounding it at the depth under test is the difference between a
        millisecond and several seconds per candidate.
        """

        return shortest_solution_length(
            self.game.board,
            self.robots,
            self.game.target_deck.current_target,
            max_depth=settings.SOLVER_MAX_DEPTH if max_depth is None else max_depth,
            node_budget=settings.SOLVER_NODE_BUDGET,
        )


    def _reset_from_curriculum(self, target):
        """Start a position whose *measured* optimal depth is `curriculum_moves`.

        Generate several candidate positions, solve each exactly, and keep the
        one closest to the requested depth. `last_optimal_depth` records what was
        actually produced -- that number, not the requested one, is what the
        promotion gate and TensorBoard read.

        Three generators, because none covers the range alone. Measured on
        level_01 against BFS, 60-80 samples per cell:

          * Backward scramble from solved (sampling predecessor states): ~85% of
            its output is one move from the goal, and that does not change with
            scramble length (tested 2 -> 32) or with how strongly the goal robot
            is favoured. Sliding is long-range, so a robot anywhere on the
            target's row or column is still one move out, and a backward random
            walk essentially never escapes that set. Kept only as a reliable
            depth-1-to-2 source.

          * Forward walk from solved -- start on the goal and play `n` random
            *legal* moves. Unlike the backward scramble this actually responds to
            `n`: mean optimal depth 4.5 at n=16, 6.4 at n=64. Moves are not
            invertible here, so `n` is not the resulting depth and the walk can
            wander back onto the goal; the solver settles it either way.

          * Uniformly random placement: mean optimal depth 5.8, but with a fixed
            distribution that cannot be asked for more.

        The forward walk is what reaches the deep tail: at n=64-128 it produces a
        position needing 8+ moves 31-35% of the time against random placement's
        21%, and 10+ moves 6-13% against 3%. That difference is the curriculum's
        ceiling, so deep levels are walked and shallow ones are scrambled, with
        random placement mixed in throughout for variety.
        """

        matching = [
            index for index, robot in enumerate(self.robots)
            if target.color.upper() == "ANY" or robot.color.lower() == target.color.lower()
        ]
        solver_index = matching[0] if matching else None

        if not settings.CURRICULUM_VERIFY_DEPTH or solver_index is None:
            self._random_once(target)
            self.last_optimal_depth = None
            return

        pool = self._depth_pool.setdefault(
            self.curriculum_moves,
            collections.deque(maxlen=max(1, settings.CURRICULUM_POOL_SIZE)),
        )

        # Draw from the pool once it holds enough positions to be varied, but
        # keep generating on a fraction of resets so it goes on growing and the
        # agent is not shown the same handful of boards for a whole level.
        if (len(pool) >= settings.CURRICULUM_POOL_MIN
                and random.random() > settings.CURRICULUM_POOL_REFRESH):
            self._apply_pooled(random.choice(pool))
            self.last_optimal_depth = self.curriculum_moves
            return

        best_positions, best_error, best_depth = None, None, None

        # Shallow levels are a thin slice -- the generators mostly land on depth 1
        # or on 3 and up, so depth 2 in particular needs more draws to hit. It can
        # afford them: verifying a depth-2 candidate costs single-digit
        # milliseconds against most of a second at depth 6.
        attempts = max(1, settings.CURRICULUM_SCRAMBLE_ATTEMPTS)
        if self.curriculum_moves <= 2:
            attempts *= 4

        for attempt in range(attempts):

            # Alternate the depth-seeking generator with plain random placement:
            # the first supplies the difficulty, the second keeps the positions
            # from all sharing a common ancestor in the solved state.
            if attempt % 2 == 1:
                self._random_once(target)
            elif self.curriculum_moves <= 2:
                self._scramble_once(target, solver_index)
            else:
                self._forward_walk_once(target, solver_index)

            # Only the verdict "is this position exactly `curriculum_moves`
            # deep?" is needed, so the search stops one level past it. Solving to
            # the true depth of a deep position costs seconds; this costs
            # milliseconds, and every episode pays it.
            depth = self.measure_optimal_depth(max_depth=self.curriculum_moves + 1)

            # Gave up rather than concluded. Accepting these would quietly
            # reinstate the original bug -- an unverified position counted as a
            # deep one -- so they are not eligible at all.
            if depth == SEARCH_EXHAUSTED:
                continue

            # Proven deeper than the bound. Recorded as one past the level rather
            # than as its unknown true depth: the promotion gate reads these as a
            # floor, so understating is the safe direction.
            reached = self.curriculum_moves + 1 if depth is None else depth

            # Degenerate: a robot already sits on the goal, so there is no puzzle.
            if reached == 0:
                continue

            error = abs(reached - self.curriculum_moves)
            if best_error is None or error < best_error:
                best_error, best_depth = error, reached
                best_positions = [(r.x, r.y) for r in self.robots]

            if error == 0:
                break

        if best_positions is None:
            # Nothing could be verified within the budget -- the usual cause is a
            # level deep enough that the search gives up. Play a random position
            # and report the depth as unknown rather than inventing one; the
            # promotion gate refuses to advance on unlabelled episodes.
            self._random_once(target)
            self.last_optimal_depth = None
            return

        # The last attempt is not necessarily the best one, so restore explicitly.
        for robot, position in zip(self.robots, best_positions):
            robot.x, robot.y = position
            robot.prev_x, robot.prev_y = position
            robot.robotLeftTarget = False

        self.last_optimal_depth = best_depth

        # Only exact matches are worth keeping; a near miss would drift the
        # pool's difficulty away from the level it is filed under.
        if best_depth == self.curriculum_moves:
            pool.append(((target.x, target.y, target.color), list(best_positions)))


    def _apply_pooled(self, entry):
        """Install a previously verified (target, positions) pair.

        Difficulty belongs to the pair, not to the positions alone, so the target
        is restored alongside the robots -- the same mutation `restore` performs.
        """

        (target_x, target_y, target_color), positions = entry

        target = self.game.target_deck.current_target
        target.x, target.y, target.color = target_x, target_y, target_color

        for robot, position in zip(self.robots, positions):
            robot.x, robot.y = position
            robot.prev_x, robot.prev_y = position
            robot.robotLeftTarget = False


    def _random_once(self, target):
        """Place every robot on a random free square, keeping the goal clear."""

        self.game.robot_manager._initialize_robot_positions(
            force=True, forbidden_extra={(target.x, target.y)}
        )


    def _forward_walk_once(self, target, solver_index):
        """Start solved and play random legal moves until the position is messy.

        The walk length is scaled to the requested depth but deliberately much
        longer than it. Moves here are not invertible -- a slide runs until it is
        blocked -- so `n` steps away from the goal is not `n` moves back, and the
        walk can even wander onto the goal again. It is a mixing process, not a
        distance: the length only has to be enough to reach the depth being asked
        for, and the solver decides whether it did.

        The goal robot is moved first so the position does not start out solved,
        which is the walk's one systematic failure mode at short lengths.
        """

        self._random_once(target)

        solver = self.robots[solver_index]
        solver.x, solver.y = target.x, target.y
        for robot in self.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False

        manager = self.game.robot_manager
        steps = min(
            settings.CURRICULUM_WALK_MAX,
            max(settings.CURRICULUM_WALK_MIN,
                settings.CURRICULUM_WALK_PER_DEPTH * self.curriculum_moves),
        )

        for step in range(steps):

            moves = manager.get_all_legal_moves(selected_idx=None)

            if step == 0:
                moves = [move for move in moves if move[0] == solver_index] or moves

            if not moves:
                break

            robot_index, direction = moves[random.randrange(len(moves))]
            self.robots[robot_index].move_until_blocked(
                simulated=False, direction=direction,
                board=self.game.board, other_robots=self.robots,
            )

        for robot in self.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False


    def _scramble_once(self, target, solver_index):
        """Re-roll the robots and walk one fresh scramble backwards from solved.

        Re-rolling every attempt matters: repeating the scramble from the same
        blocker layout produces correlated candidates.
        """

        # Robots are placed off the target, so moving the solver onto it cannot
        # collide with anyone.
        self._random_once(target)

        solver = self.robots[solver_index]
        solver.x, solver.y = target.x, target.y

        for robot in self.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False

        self.game.robot_manager.reverse_scramble(
            self.curriculum_moves,
            required_first=solver_index,
            solver_index=solver_index,
            solver_bias=settings.CURRICULUM_SOLVER_BIAS,
            avoid_square=(target.x, target.y),
        )


    def _build_wall_planes(self):

        planes = np.zeros((len(ALL_DIRECTIONS), BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                cell = self.game.board.walls[row][col]
                for d, direction in enumerate(ALL_DIRECTIONS):
                    planes[d, row, col] = float(cell[direction])

        return planes


    def _get_obs(self):

        target = self.game.target_deck.current_target

        return build_observation(
            wall_planes=self._wall_planes,
            robot_positions=[(r.x, r.y) for r in self.robots],
            target=(target.x, target.y, COLOR_MAP[target.color.lower()]),
            move_counter=self.move_counter,
        )


    def wall_planes(self):
        """The level's wall planes. Constant within a level, so callers may hold
        a reference rather than a copy."""

        return self._wall_planes


    def target_tuple(self):

        target = self.game.target_deck.current_target
        return (target.x, target.y, COLOR_MAP[target.color.lower()])


    def _state_key(self):

        return tuple((p.x, p.y) for p in self.robots)


    def snapshot(self):
        """Capture the full mutable state of an episode.

        Ricochet Robots is deterministic and fully observable, and the entire
        episode state is four robot positions plus the goal and a little
        bookkeeping. That makes the real simulator cheap enough to search over
        directly -- which is what SEARCH_MODE=alphazero does instead of unrolling
        a learned dynamics model.
        """

        target = self.game.target_deck.current_target

        return {
            "positions": [(r.x, r.y) for r in self.robots],
            "prev_positions": [(r.prev_x, r.prev_y) for r in self.robots],
            "left_target": [r.robotLeftTarget for r in self.robots],
            "target": (target.x, target.y, target.color),
            "visited": set(self.visited_states),
            "move_counter": self.move_counter,
        }


    def restore(self, state):

        for robot, position, previous, left in zip(
            self.robots, state["positions"], state["prev_positions"], state["left_target"]
        ):
            robot.x, robot.y = position
            robot.prev_x, robot.prev_y = previous
            robot.robotLeftTarget = left

        target = self.game.target_deck.current_target
        target.x, target.y, target.color = state["target"]

        self.visited_states = set(state["visited"])
        self.move_counter = state["move_counter"]


    def step(self, action):

        robot_idx, direction = divmod(int(action), NUMBER_OF_DIRECTIONS)
        done = False

        # Rewards live on a [-1, 1] scale. Solve length is priced through the
        # per-move cost rather than a large terminal bonus, so the value target
        # stays small enough for the categorical head's support.
        reward = REWARD_PER_MOVE

        robot = self.robots[robot_idx]

        if robot.move_until_blocked(
            simulated=False,
            direction=INT2DIRECTION[int(direction)],
            board=self.game.board,
            other_robots=self.game.robot_manager.robots
        ):
            self.move_counter += 1

            if robot.is_target_reached(self.game.target_deck.current_target):
                reward += REWARD_SOLVE
                done = True

        # Revisiting a configuration is discouraged, but the penalty must not
        # replace the win: a solving move that happens to land on a previously
        # seen configuration used to be recorded as a penalty outright.
        state_key = self._state_key()
        if not done:
            if state_key in self.visited_states:
                reward += REWARD_REPEAT_STATE
            else:
                self.visited_states.add(state_key)

        obs = self._get_obs()
        return obs, reward, done, {}


    def render(self):

        if self.render_ai:
            self.game.render_ai_environment(0, self.move_counter)


    def close(self):

        if self.render_ai:
            self.game.close()
