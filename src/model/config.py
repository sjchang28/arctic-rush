"""The learning contract and the fixed parts of the learner.

Two kinds of thing live here, and neither belongs in `src/config.py`:

  * the contract between the environment and the network -- action space,
    observation planes, reward scale. `src/game/env.py` imports these; that is
    not a layering violation, since the environment exists to feed the learner
    and one definition is what stops the encoder and the network disagreeing
    about plane 12.
  * architecture and optimisation constants that are properties of *this*
    model, not of a run. They were env-overridable fields once, which made a
    sweep look like it could vary the trunk width between two jobs sharing a
    checkpoint directory -- it cannot; the weights would not load.

Change these and you are changing the model, so every run afterwards is a new
experiment rather than a new configuration of the old one.
"""

from src.game.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUMBER_OF_DIRECTIONS,
    NUMBER_OF_ROBOTS,
)

# Filename of the weights within MODEL_DIR/<RUN_ID>. The directory is per-run
# and env-overridable; the filename inside it never needed to be.
WEIGHTS_FILE_PATH = "leela.pth"

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

# The curriculum tops out around depth 8-10, so a 25-move budget leaves ample
# room to wander and still finish. A larger cap mostly buys longer failures.
# It also normalises the move-counter plane, so changing it changes the
# observation -- another reason it is not a per-run dial.
MAX_TOTAL_MOVES_PER_GAME = 25

# Network architecture. Baked into the checkpoint: a run that changes these
# cannot resume from weights trained without them.
NUM_CHANNELS = 64           # channel width of the residual trunk
NUM_BLOCKS = 4              # residual blocks in the representation trunk
VALUE_SUPPORT_SIZE = 10     # categorical value/reward support spans [-N, N]
VALUE_PREFIX_DIM = 128      # hidden width of the value-prefix LSTM
LSTM_HORIZON_LEN = 5        # steps the prefix LSTM accumulates before reset

# Unroll / replay. NUM_UNROLL_STEPS drives how many recurrent_inference steps are
# held in a single autograd graph per batch sample, so it dominates training
# VRAM. The MuZero reference uses 5; unrolling a whole game (100) is what OOMs a
# 6 GB card. TD_STEPS is the bootstrap horizon for value targets (CPU-side cost
# only). REPLAY_WINDOW_SIZE is how many finished games stay resident in host RAM.
NUM_UNROLL_STEPS = 5
TD_STEPS = 10
REPLAY_WINDOW_SIZE = 100
MIN_REPLAY_GAMES = 4        # games required in the buffer before training starts
EMPTY_CACHE_EVERY_N_EPISODES = 10

# Search shaping. Gumbel deliberately searches only the top-m sampled root
# actions; with a 16-action space that is small enough to consider all of them,
# and doing so means a one-move win can never be missed simply because it was
# not sampled.
USE_GUMBEL = True
GUMBEL_NUM_CONSIDERED = 16

# Losses and sampling.
CONSISTENCY_LOSS_WEIGHT = 2.0     # EfficientZero self-supervised consistency loss
PRIORITIZED_REPLAY_ALPHA = 1.0    # priority exponent; 0 disables prioritisation
PRIORITIZED_REPLAY_BETA = 1.0     # importance-sampling correction exponent
USE_HER = True                    # hindsight relabelling of failed episodes
HER_FRACTION = 0.5                # fraction of sampled failed games relabelled

