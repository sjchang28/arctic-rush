"""The board and its presentation.

Everything here is a fixed property of Ricochet Robots or of how it is drawn --
a 16x16 board with four robots and four directions -- so none of it is an
env-overridable setting. `src/model/config.py` derives the observation and
action shapes from these, which is what keeps the renderer and the learner
describing the same board.
"""

# Different Board Levels
"""_summary_
    level_01.json, level_02.json, & level_03.json basic levels that have different walls and target placements
    level_04.json contains bounce pads that redirect the robot depending on their incoming direction + color (* Implement bounce pads for ML model)
"""
LEVEL_FILE = "level_01.json"

# Board geometry
BOARD_WIDTH = 16
BOARD_HEIGHT = 16


def center_squares():
    """The blocked squares at the middle of the board.

    Ricochet Robots walls off a 2x2 block at the centre. This was written out as
    the literal `{(7, 7), (7, 8), (8, 7), (8, 8)}`, correct only for a 16x16
    board -- any other size left the block off-centre with no error, just
    quietly wrong robot placement. Derived from the dimensions instead.
    """

    low_x, low_y = (BOARD_WIDTH - 1) // 2, (BOARD_HEIGHT - 1) // 2

    return {
        (x, y)
        for x in (low_x, low_x + 1)
        for y in (low_y, low_y + 1)
    }

# PyGame Settings
TILE_SIZE = 40
TARGET_SIZE = 30
SCREEN_WIDTH = TILE_SIZE * BOARD_WIDTH * 1.5
SCREEN_HEIGHT = TILE_SIZE * BOARD_HEIGHT
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARK_GREY = (33, 33, 33)
PINK = (231, 50, 189)

# Robot identity. Insertion order is not cosmetic: it fixes each robot's plane
# index in the observation, so reordering this dict silently relabels the
# network's inputs.
ROBOT_COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0)
}
NUMBER_OF_ROBOTS = len(ROBOT_COLORS)

COLOR_MAP = {
    "red": 0,
    "blue": 1,
    "green": 2,
    "yellow": 3,
    "any": 4
}
INDEX_COLOR_MAP = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "any"
}

# Directions for readability
UP, RIGHT, DOWN, LEFT, SWITCH = 'up', 'right', 'down', 'left', 'switch'
ALL_MOVES = [UP, RIGHT, DOWN, LEFT, SWITCH]
NUMBER_OF_POSSIBLE_MOVES = len(ALL_MOVES)
ALL_DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
NUMBER_OF_DIRECTIONS = len(ALL_DIRECTIONS)
INT2DIRECTION = {
    0: UP,
    1: RIGHT,
    2: DOWN,
    3: LEFT,
    4: SWITCH
}
DIRECTION2INT = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3,
    SWITCH: 4
}


# ---------------------------------------------------------------------------
# Curriculum and exact solver.
#
# These describe how start positions are generated and measured, which is a
# property of the environment, not of the learner. They lived in
# `src/model/config.py`, so `src/game/env.py` imported twenty names from the
# model package and the environment could not be constructed -- or tested --
# without the learner's configuration. The dependency runs learner ->
# environment; this block moving here is what makes that true.
#
# `src/model/config.py` still owns the reward values and observation shape:
# those are genuinely the learner's contract with the environment rather than
# the environment's own rules.
# ---------------------------------------------------------------------------

# Curriculum, ramped by measured difficulty. Starts one move from solved --
# with a sparse reward and a fresh network, forward exploration essentially
# never reaches the goal -- and deepens a level each time the agent solves
# CURRICULUM_PROMOTE_THRESHOLD of the last CURRICULUM_PROMOTE_WINDOW episodes,
# so the task stays at the edge of what it can currently do.
#
# CURRICULUM_START_MOVES is an *optimal* solution length, not a scramble count.
# An earlier run ramped 2 -> 12 in ten consecutive windows without ever
# stalling, which looked like fast learning and was not: backward scrambling
# produces a position one move from solved ~85% of the time no matter how many
# moves it is given, so every level was serving depth-1 puzzles and the 96%
# solved rate was measured on them. Every position is now solved exactly by BFS
# before use, and a level that cannot generate or verify its own difficulty
# holds and logs why instead of promoting.
#
# Expect the ramp to stall around depth 8-10 on level_01: the level that asks
# for 8 achieves a mean of 7.5, and deeper levels fall further behind until the
# min-depth gate holds them. That stall is the honest signal, and it is a
# generator limit rather than a solver one -- raising SOLVER_NODE_BUDGET will
# not move it.
#
# Verification costs ~0.2 s per episode at depth 4 and ~1.3 s at depth 8,
# against a ~1.5 s episode. CURRICULUM_POOL_REFRESH is the dial: lower it to
# spend less time generating fresh positions, at the cost of variety.
CURRICULUM_START_MOVES = 1          # starting depth; 0 disables the curriculum
CURRICULUM_MAX_MOVES = 15           # deepest level the ramp will reach
CURRICULUM_PROMOTE_THRESHOLD = 0.75  # solved rate over the window that promotes
CURRICULUM_PROMOTE_WINDOW = 30      # episodes the promotion decision reads

# A backward move is not the inverse of a forward one in this game, and a
# uniformly sampled scramble mostly shuffles robots that are not the one heading
# for the goal. Left unverified the generator emits one-move positions labelled
# "depth 12", the promotion gate never stalls, and the solved rate measures
# nothing. These verify and correct that.
CURRICULUM_VERIFY_DEPTH = True      # measure true optimal depth with BFS
CURRICULUM_SCRAMBLE_ATTEMPTS = 4    # candidates per reset; closest one is kept
CURRICULUM_SOLVER_BIAS = 0.6        # chance a backward move uses the goal robot
CURRICULUM_MIN_DEPTH_RATIO = 0.75   # mean measured depth / level depth to promote

# Verifying a candidate costs ~2 ms at depth 1 and ~2.5 s at depth 6, so deep
# levels would spend more wall clock proving positions than training on them.
# Verified positions are therefore pooled and resampled.
CURRICULUM_POOL_SIZE = 256          # verified start positions retained per depth
CURRICULUM_POOL_MIN = 24            # entries before resampling beats generating
CURRICULUM_POOL_REFRESH = 0.15      # fraction of resets that generate anyway

# Forward walk from the solved position. Length is a mixing parameter, not a
# distance -- slides are not invertible, so n steps out is not n moves back.
# Longer walks reach the deep tail that random placement cannot: at n=64, 31% of
# positions need 8+ moves against random placement's 21%.
CURRICULUM_WALK_PER_DEPTH = 12      # forward-walk moves per unit of depth
CURRICULUM_WALK_MIN = 16
CURRICULUM_WALK_MAX = 128

# Exact BFS solver, used to label positions and to score solutions. Keep
# SOLVER_MAX_DEPTH at or above CURRICULUM_MAX_MOVES, or the deepest levels
# cannot be verified and so can never be promoted into.
SOLVER_MAX_DEPTH = 16
SOLVER_NODE_BUDGET = 15_000         # expansions before the solver reports "unknown"
