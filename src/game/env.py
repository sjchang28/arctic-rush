
import collections
import random

import gymnasium
import numpy as np
from gymnasium import spaces

from src.game.config import (
    ALL_DIRECTIONS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    COLOR_MAP,
    CURRICULUM_MASTERY_THRESHOLD,
    CURRICULUM_MASTERY_WINDOW,
    CURRICULUM_REHEARSAL_RATE,
    CURRICULUM_START_MOVES,
    INT2DIRECTION,
    NUMBER_OF_DIRECTIONS,
    NUMBER_OF_ROBOTS,
    SOLVER_MAX_DEPTH,
    SOLVER_NODE_BUDGET,
)
from src.game.curriculum import CurriculumGenerator
from src.game.game import AI_Game
from src.game.robots import Robot
from src.game.solver import shortest_solution_length

# The learner's contract with the environment: what a move is worth and what
# shape the observation must be. These are the only names `game` takes from
# `model`, and `src/model/config.py` documents why they live there.
from src.model.config import (
    AI_ACTION_SPACE_SIZE,
    AI_OBSERVATION_SHAPE,
    MAX_TOTAL_MOVES_PER_GAME,
    REWARD_PER_MOVE,
    REWARD_REPEAT_STATE,
    REWARD_SOLVE,
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
        move_counter / max(MAX_TOTAL_MOVES_PER_GAME, 1),
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
        self.curriculum_moves = CURRICULUM_START_MOVES

        # Measured optimal solution length of the position the episode started
        # from. None when it was not measured (curriculum off, or verification
        # disabled). This -- not the scramble depth -- is the honest difficulty.
        self.last_optimal_depth = None

        # Whether the episode started from a rehearsed depth rather than the
        # current level. Those episodes are practice, and the promotion gate must
        # not read them as evidence about the level it is assessing.
        self.last_was_rehearsal = False

        # Per-depth recent results, and the depths whose rate has ever cleared
        # CURRICULUM_MASTERY_THRESHOLD. The second is what gets rehearsed.
        self._depth_results = {}
        self._mastered_depths = set()

        # Start-position generation, including its verified-position pool.
        self.curriculum = CurriculumGenerator(self)

        # Action space: 4 robots x 4 directions.
        self.action_space = spaces.Discrete(AI_ACTION_SPACE_SIZE)

        # Observation space: stacked board planes (see config.AI_OBSERVATION_SHAPE)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=AI_OBSERVATION_SHAPE, dtype=np.float32
        )

        # Wall planes never change within a level, so build them once.
        self._wall_planes = self._build_wall_planes()

        self.reset()


    @property
    def current_target(self):
        """The goal this episode is playing towards.

        Callers reached through `env.game.target_deck.current_target` at seven
        sites, which coupled them to the deck's internals for a value the
        environment can simply expose.
        """

        return self.game.target_deck.current_target


    def place_robots_randomly(self, forbidden_extra=()):
        """Re-roll every robot onto a random free square.

        A public seam for the curriculum generator, which previously reached
        into `env.game.robot_manager._initialize_robot_positions` -- through two
        objects and into a private method.
        """

        self.game.robot_manager._initialize_robot_positions(
            force=True, forbidden_extra=forbidden_extra
        )


    def set_curriculum_moves(self, moves: int):

        """Ramp the reverse-curriculum depth. Takes effect on the next reset."""

        self.curriculum_moves = max(0, int(moves))


    def record_result(self, depth, solved: bool):

        """Log the outcome of an episode played at `depth`.

        Both assessment and rehearsal episodes are recorded, each against the
        depth it was actually played at. Rehearsals keep a mastered depth's rate
        current -- without them the record would freeze at the moment the ramp
        left that level and never say anything again.
        """

        if depth is None or depth <= 0:
            return

        results = self._depth_results.setdefault(
            depth, collections.deque(maxlen=CURRICULUM_MASTERY_WINDOW))
        results.append(1.0 if solved else 0.0)

        if len(results) < CURRICULUM_MASTERY_WINDOW:
            return

        rate = sum(results) / len(results)
        if rate >= CURRICULUM_MASTERY_THRESHOLD:
            self._mastered_depths.add(depth)


    def mastered_depths(self):

        """Depths below the current level that the agent has mastered."""

        return sorted(d for d in self._mastered_depths if d < self.curriculum_moves)


    def _next_start_depth(self):

        """Depth to generate this episode's start position at.

        Usually the current level. On a `CURRICULUM_REHEARSAL_RATE` fraction of
        resets it is a mastered shallower depth instead -- levels the ramp has
        left are otherwise never seen again, and the network trains them away
        (see the note in `game.config`).

        Falls through to the current level when nothing has been mastered yet,
        which is the case early in a run and at the starting depth.
        """

        if random.random() >= CURRICULUM_REHEARSAL_RATE:
            return self.curriculum_moves

        candidates = self.mastered_depths()
        if not candidates:
            return self.curriculum_moves

        return random.choice(candidates)


    def reset(self):

        # Pick the target first so robots can be kept off it -- starting an episode
        # already standing on the goal would hand out a free win.
        self.game.target_deck.set_new_target()

        target = self.current_target

        if self.curriculum_moves > 0:
            depth = self._next_start_depth()
            self.last_was_rehearsal = depth != self.curriculum_moves
            self.last_optimal_depth = self.curriculum.generate(target, depth)
        else:
            self.curriculum.random_once(target)
            self.last_optimal_depth = None
            self.last_was_rehearsal = False

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
            self.current_target,
            max_depth=SOLVER_MAX_DEPTH if max_depth is None else max_depth,
            node_budget=SOLVER_NODE_BUDGET,
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

        target = self.current_target

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

        target = self.current_target
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

        target = self.current_target

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

        target = self.current_target
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

            if robot.is_target_reached(self.current_target):
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
