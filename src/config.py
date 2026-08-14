import os

from pydantic import Field
from pydantic_settings import BaseSettings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASCII_LEELA_BOT = r"""

 _   _ _     ___ _             _              _       _ 
| | | (_)   |_ _( )_ __ ___   | |    ___  ___| | __ _| |
| |_| | |    | ||/| '_ ` _ \  | |   / _ \/ _ \ |/ _` | |
|  _  | |_   | |  | | | | | | | |__|  __/  __/ | (_| |_|
|_| |_|_( ) |___| |_| |_| |_| |_____\___|\___|_|\__,_(_)
        |/                                              

           ||||||||||||,,
           |WWWWWWWWW|W|||,
           |_________|~WWW||,
            ~-_      ~_  ~WW||,
            __-~---__/ ~_  ~WW|,
        _-~~         ~~-_~_  ~W
  _--~~~~~~~~~~___       ~-~_/
 -                ~~~--_   ~_
|                       ~_   |
|   ____-------___        -_  |
|-~~              ~~--_     - |
 ~| ~--___________     |-_   ~_
   | \`~'/  \`~'_-~~  |  |~-_-
  _-~_~~~    ~~~   _-~  |  |
 ---.--__         ---.-~  |
 | |    -~~-----~~| |    -
 |_|__-~          |_|__-~

"""

# Different Board Levels
"""_summary_
    level_01.json, level_02.json, & level_03.json basic levels that have different walls and target placements
    level_04.json contains bounce pads that redirect the robot depending on their incoming direction + color (* Implement bounce pads for ML model)
"""
LEVEL_FILE = "level_01.json"

# PyGame Settings
TILE_SIZE = 40
TARGET_SIZE = 30
BOARD_WIDTH = 16
BOARD_HEIGHT = 16
SCREEN_WIDTH = TILE_SIZE * BOARD_WIDTH * 1.5
SCREEN_HEIGHT = TILE_SIZE * BOARD_HEIGHT
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARK_GREY = (33, 33, 33)
PINK = (231, 50, 189)
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

# The AI action space is (robot, direction) only: action id `i * 4 + d` means
# "move robot i in direction d". SWITCH exists solely for the human pygame path,
# where a player tabs to select a robot before moving it. Including SWITCH in the
# learning action space made 16 of 20 action ids alias onto 4 distinct effects,
# because the environment moved the *selected* robot and ignored the robot index
# the search had planned for.
AI_ACTION_SPACE_SIZE = NUMBER_OF_ROBOTS * NUMBER_OF_DIRECTIONS

# Observation is a stack of BOARD_HEIGHT x BOARD_WIDTH planes, not a flat vector.
# Wall-blocking and robot-blocking are long-range spatial relations; flattening
# them into 1040 numbers for an MLP threw away the structure that makes them
# learnable at all.
#
#   0-3   walls, one plane per direction (up, right, down, left)
#   4-7   robot occupancy, one plane per robot colour
#   8-11  target square, one plane per target colour ("any" marks all four)
#   12    move counter, broadcast and normalised by the move cap
AI_OBSERVATION_PLANES = (4 * 1) + NUMBER_OF_ROBOTS + NUMBER_OF_ROBOTS + 1
AI_OBSERVATION_SHAPE = (AI_OBSERVATION_PLANES, BOARD_HEIGHT, BOARD_WIDTH)
AI_OBSERVATION_SPACE_SIZE = AI_OBSERVATION_PLANES * BOARD_HEIGHT * BOARD_WIDTH

# Reward scale. The original scheme paid up to 1000 for a solve and -5 for a
# repeated state, so plain MSE on the value target dwarfed the policy loss by
# orders of magnitude and diverged to NaN within a handful of episodes.
REWARD_SOLVE = 1.0
REWARD_PER_MOVE = -0.01
REWARD_REPEAT_STATE = -0.04

class Settings(BaseSettings):
    """Every tunable for a training run, in one place.

    This is the single source of truth for run configuration. Values were
    previously split between here and `docker-compose.yaml`, which meant the
    defaults in this file described a run nobody was doing: the container
    overrode a dozen of them, so reading either file alone gave a wrong picture
    of what was actually training.

    Everything here is still env-overridable -- that is what lets parallel
    Docker/k8s runs differ without editing code. The rule is that an override
    should exist only where a value genuinely differs *between deployments*:

      * `docker-compose.yaml` sets `RUN_ID`, because that is what separates one
        run's checkpoints and logs from another's.
      * `Dockerfile` sets `MODEL_DIR` / `LOG_DIR`, because those are container
        paths onto the mounted volume.
      * Tuning knobs belong here, not in either of those.
    """

    # Run identity / output dirs
    RUN_ID: str = Field("local", description="Identifier for this training run")
    # Defaults point at the repository-level `data/` tree (one level above
    # `src/`), which is what the Docker/k8s volume mounts onto. Code lives in
    # `src/` and run artefacts live in `data/`; nothing writes into the package.
    MODEL_DIR: str = Field(
        default_factory=lambda: os.path.join(_REPO_ROOT, "data", "models"),
        description="Directory model checkpoints are written to/read from",
    )
    LOG_DIR: str = Field(
        default_factory=lambda: os.path.join(_REPO_ROOT, "data", "logs"),
        description="Directory training logs are written to",
    )

    # Train LEELA
    LEVEL_FILE: str = Field("level_01.json", description="Board level JSON to train/play on")
    WEIGHTS_FILE_PATH: str = Field("leela.pth", description="Filename of the model weights within MODEL_DIR")
    TOTAL_TIMESTEPS_FOR_TRAINING: int = 500 * 1000  # 500k timesteps
    TRAINING_EPISODES: int = Field(4000, description="Number of self-play/training episodes")
    NUM_ACTORS: int = Field(1, description="Number of parallel self-play actors")
    TOTAL_MCTS_EPISODES: int = Field(50, description="Simulations per move")
    # The curriculum tops out around depth 8-10, so a 25-move budget leaves ample
    # room to wander and still finish. A larger cap mostly buys longer failures.
    MAX_TOTAL_MOVES_PER_GAME: int = Field(25, description="Move cap per game")

    # Memory / VRAM tuning knobs
    # NUM_UNROLL_STEPS drives how many recurrent_inference steps are held in a single
    # autograd graph per batch sample, so it dominates training VRAM. The MuZero
    # reference uses 5; unrolling a whole game (100) is what OOMs a 6 GB card.
    # TD_STEPS is the bootstrap horizon for value targets (CPU-side cost only).
    # REPLAY_WINDOW_SIZE is how many finished games stay resident in host RAM.
    NUM_UNROLL_STEPS: int = 5
    TD_STEPS: int = 10
    REPLAY_WINDOW_SIZE: int = 100
    EMPTY_CACHE_EVERY_N_EPISODES: int = 10

    # Console logging cadence. Every episode prints one compact line; the rolling
    # averages are only worth a line every so often, otherwise they triple the
    # log volume and bury the per-episode results.
    LOG_SHORT_AVG_EVERY: int = Field(20, description="Episodes between short-window average lines")
    LOG_LONG_AVG_EVERY: int = Field(100, description="Episodes between long-window average lines")

    # Gradient steps taken per self-play episode. This used to be hard-coded to 1,
    # which meant a default 120-episode run performed 120 SGD steps in total while
    # the LR schedule was written against a 500k-step run.
    TRAIN_STEPS_PER_EPISODE: int = Field(40, description="Gradient steps per self-play episode")
    MIN_REPLAY_GAMES: int = Field(4, description="Games required in the buffer before training starts")

    # Checkpointing. One weights file, overwritten only when the model improves:
    # saving every episode meant a run that peaked and then degraded ended with
    # the degraded weights, since the last episode always won.
    SAVE_BEST_ONLY: bool = Field(True, description="Overwrite weights only on improvement")
    CHECKPOINT_WARMUP_EPISODES: int = Field(
        20, description="Episodes saved unconditionally so an early crash still leaves a file"
    )

    # Network
    NUM_CHANNELS: int = Field(64, description="Channel width of the residual trunk")
    NUM_BLOCKS: int = Field(4, description="Residual blocks in the representation trunk")
    VALUE_SUPPORT_SIZE: int = Field(
        10,
        description="Categorical value/reward support spans [-N, N] in transformed space",
    )

    # Search. Ricochet Robots is deterministic and fully observable with a cheap
    # exact simulator, so MuZero's learned dynamics is pure cost here -- it has to
    # rediscover wall and robot blocking before its search means anything.
    # Measured head to head on this task: 86% solved vs 33% at equal wall clock.
    # SEARCH_MODE=muzero still works if you want the comparison back.
    SEARCH_MODE: str = Field("alphazero", description="'muzero' (learned dynamics) or 'alphazero' (real environment)")
    USE_GUMBEL: bool = Field(True, description="Gumbel root search instead of Dirichlet noise + PUCT at the root")
    # Gumbel deliberately searches only the top-m sampled root actions. With a
    # 16-action space that is small enough to consider all of them, and doing so
    # means a one-move win can never be missed simply because it was not sampled.
    GUMBEL_NUM_CONSIDERED: int = Field(16, description="Root actions sampled without replacement for sequential halving")

    # Sample efficiency.
    # Reanalyse re-searches stored positions with the *learned* model, which is
    # not the model SEARCH_MODE=alphazero plans with, so it stays off by default.
    # Raise it alongside SEARCH_MODE=muzero.
    REANALYSE_FRACTION: float = Field(
        0.0, description="Fraction of training batches drawn from reanalysed (refreshed) targets"
    )
    CONSISTENCY_LOSS_WEIGHT: float = Field(
        2.0, description="Weight of the EfficientZero self-supervised consistency loss"
    )
    VALUE_PREFIX_DIM: int = Field(128, description="Hidden width of the value-prefix LSTM")
    LSTM_HORIZON_LEN: int = Field(
        5, description="Steps the value-prefix LSTM accumulates before its state is reset"
    )
    PRIORITIZED_REPLAY_ALPHA: float = Field(1.0, description="Replay priority exponent; 0 disables prioritisation")
    PRIORITIZED_REPLAY_BETA: float = Field(1.0, description="Importance-sampling correction exponent")
    USE_HER: bool = Field(True, description="Hindsight relabelling of failed episodes")
    HER_FRACTION: float = Field(0.5, description="Fraction of sampled failed games that get relabelled")
    # Curriculum, ramped by measured difficulty. Starts one move from solved --
    # with a sparse reward and a fresh network, forward exploration essentially
    # never reaches the goal -- and deepens a level each time the agent solves
    # CURRICULUM_PROMOTE_THRESHOLD of the last CURRICULUM_PROMOTE_WINDOW
    # episodes, so the task stays at the edge of what it can currently do.
    #
    # CURRICULUM_START_MOVES is an *optimal* solution length, not a scramble
    # count. An earlier run ramped 2 -> 12 in ten consecutive windows without
    # ever stalling, which looked like fast learning and was not: backward
    # scrambling produces a position one move from solved ~85% of the time no
    # matter how many moves it is given, so every level was serving depth-1
    # puzzles and the 96% solved rate was measured on them. Every position is now
    # solved exactly by BFS before use, and a level that cannot generate or
    # verify its own difficulty holds and logs why instead of promoting.
    #
    # Expect the ramp to stall around depth 8-10 on level_01: the level that asks
    # for 8 achieves a mean of 7.5, and deeper levels fall further behind until
    # the min-depth gate holds them. That stall is the honest signal, and it is a
    # generator limit rather than a solver one -- raising SOLVER_NODE_BUDGET will
    # not move it.
    #
    # Verification costs ~0.2 s per episode at depth 4 and ~1.3 s at depth 8,
    # against a ~1.5 s episode. CURRICULUM_POOL_REFRESH is the dial: lower it to
    # spend less time generating fresh positions, at the cost of variety.
    CURRICULUM_START_MOVES: int = Field(
        1, description="Curriculum depth as an optimal solution length; 0 disables the curriculum"
    )
    CURRICULUM_MAX_MOVES: int = Field(15, description="Deepest level the curriculum will ramp to")
    CURRICULUM_PROMOTE_THRESHOLD: float = Field(
        0.75, description="Solved rate over the recent window that promotes the curriculum a level"
    )
    CURRICULUM_PROMOTE_WINDOW: int = Field(
        30, description="Episodes of history the promotion decision is made over"
    )
    # A backward move is not the inverse of a forward one in this game, and a
    # uniformly sampled scramble mostly shuffles robots that are not the one
    # heading for the goal. Left unverified the generator emits one-move
    # positions labelled "depth 12", the promotion gate never stalls, and the
    # solved rate measures nothing. These settings verify and correct that.
    CURRICULUM_VERIFY_DEPTH: bool = Field(
        True, description="Measure each generated position's true optimal depth with BFS"
    )
    CURRICULUM_SCRAMBLE_ATTEMPTS: int = Field(
        4, description="Candidate positions generated per reset; the closest to the level is kept"
    )
    CURRICULUM_SOLVER_BIAS: float = Field(
        0.6, description="Probability a backward move is drawn from the goal robot's candidates"
    )
    CURRICULUM_MIN_DEPTH_RATIO: float = Field(
        0.75,
        description="Mean measured depth, as a fraction of level depth, required to promote",
    )
    # Verifying a candidate costs ~2 ms at depth 1 and ~2.5 s at depth 6, so
    # deep levels would spend more wall clock proving positions than training on
    # them. Verified positions are therefore pooled and resampled.
    CURRICULUM_POOL_SIZE: int = Field(256, description="Verified start positions retained per depth")
    CURRICULUM_POOL_MIN: int = Field(
        24, description="Pool entries required before positions are resampled rather than generated"
    )
    CURRICULUM_POOL_REFRESH: float = Field(
        0.15, description="Fraction of resets that generate a fresh position even with a full pool"
    )
    # Forward walk from the solved position. Length is a mixing parameter, not a
    # distance -- slides are not invertible, so n steps out is not n moves back.
    # Longer walks reach the deep tail that random placement cannot: at n=64,
    # 31% of positions need 8+ moves against random placement's 21%.
    CURRICULUM_WALK_PER_DEPTH: int = Field(12, description="Forward-walk moves per unit of depth")
    CURRICULUM_WALK_MIN: int = Field(16, description="Shortest forward walk")
    CURRICULUM_WALK_MAX: int = Field(128, description="Longest forward walk")

    # Exact BFS solver, used to label positions and to score solutions. Keep
    # SOLVER_MAX_DEPTH at or above CURRICULUM_MAX_MOVES, or the deepest levels
    # cannot be verified and so can never be promoted into.
    SOLVER_MAX_DEPTH: int = Field(16, description="Deepest solution the BFS solver will look for")
    SOLVER_NODE_BUDGET: int = Field(
        15_000, description="Node expansions before the solver gives up and reports 'unknown'"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


settings: Settings = Settings()