import collections
from dataclasses import dataclass
from typing import Callable, Optional

from src.config import settings
from src.core.logger import logger
from src.game.config import CURRICULUM_START_MOVES
from src.model.config import (
    AI_ACTION_SPACE_SIZE,
    AI_OBSERVATION_PLANES,
    AI_OBSERVATION_SPACE_SIZE,
    CONSISTENCY_LOSS_WEIGHT,
    GUMBEL_NUM_CONSIDERED,
    LSTM_HORIZON_LEN,
    MAX_TOTAL_MOVES_PER_GAME,
    NUM_BLOCKS,
    NUM_CHANNELS,
    NUM_UNROLL_STEPS,
    REPLAY_WINDOW_SIZE,
    TD_STEPS,
    USE_GUMBEL,
    VALUE_PREFIX_DIM,
    VALUE_SUPPORT_SIZE,
)
from src.model.state import (
    RicochetRobotEnvironment,
    RicochetRobotsGame,
    action_to_index,
    index_to_action,
)

##########################
####### Helpers ##########

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


def visit_softmax_temperature(num_moves, training_steps):

    # higher temperature higher exploration
    if training_steps < int(50e3):
        return 2.0
    elif training_steps < int(100e3):
        return 1.0
    else:
        return 0.5


@dataclass
class MuZeroConfig:
    """Every knob one run is configured with.

    Was a 103-line `__init__` taking 21 positional parameters and assigning each
    to an identically named attribute, with a second 16-parameter constructor in
    `RicochetRobotsConfig` existing only to supply defaults. The parameter list
    and the assignment list had to be kept in step by hand, in two places.

    As a dataclass the defaults live on the fields themselves, so the subclass
    only states what it changes. Everything genuinely *derived* -- rather than
    passed -- is computed in `__post_init__`.
    """

    # Self-play
    action_space_size: int = AI_ACTION_SPACE_SIZE
    observation_space_size: int = AI_OBSERVATION_SPACE_SIZE
    max_moves: int = MAX_TOTAL_MOVES_PER_GAME
    discount: float = 1.0
    num_simulations: int = settings.TOTAL_MCTS_EPISODES
    num_actors: int = settings.NUM_ACTORS
    visit_softmax_temperature_fn: Callable = visit_softmax_temperature

    # Root prior exploration noise.
    dirichlet_alpha: float = 0.25
    root_exploration_fraction: float = 0.25

    # UCB formula
    pb_c_base: int = 19652
    pb_c_init: float = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling. This is not
    # strictly necessary, but establishes identical behaviour to AlphaZero in
    # board games.
    known_bounds: Optional[KnownBounds] = None

    # Training
    training_steps: int = int(500e3)
    checkpoint_interval: int = int(1e2)
    # window_size is how many finished games stay resident in host RAM.
    window_size: int = REPLAY_WINDOW_SIZE
    batch_size: int = 16

    # num_unroll_steps sets how many recurrent_inference steps live in one
    # autograd graph per batch sample, so it dominates training VRAM.
    num_unroll_steps: int = NUM_UNROLL_STEPS
    td_steps: int = TD_STEPS

    weight_decay: float = 1e-4
    momentum: float = 0.9

    training_episodes: int = settings.TRAINING_EPISODES
    hidden_layer_size: int = 64

    # Convolutional trunk over board planes (see config.AI_OBSERVATION_SHAPE).
    # None means "take the project default"; kept rather than defaulting
    # directly to the constant so an explicit None from a caller still resolves.
    observation_planes: int = AI_OBSERVATION_PLANES
    num_channels: Optional[int] = None
    num_blocks: Optional[int] = None
    value_support_size: Optional[int] = None

    # Weight of the EfficientZero self-supervised consistency loss.
    consistency_loss_weight: float = CONSISTENCY_LOSS_WEIGHT

    # Value-prefix head (EfficientZero): predicts the cumulative reward over a
    # window rather than the reward of one specific step.
    value_prefix_dim: int = VALUE_PREFIX_DIM
    lstm_horizon_len: int = LSTM_HORIZON_LEN

    # Gumbel root search (see mcts.run_gumbel_mcts).
    use_gumbel: bool = USE_GUMBEL
    gumbel_num_considered: int = GUMBEL_NUM_CONSIDERED

    # Exponential learning rate schedule
    lr_init: float = 0.005
    lr_decay_rate: float = 0.1
    lr_decay_steps: float = 100000

    # "muzero" searches the learned dynamics model; "alphazero" searches the
    # real simulator (see mcts.RealEnvironmentModel). Resolved in __post_init__
    # so a test that patches the setting before constructing is honoured.
    search_mode: Optional[str] = None

    def __post_init__(self):

        # `root_dirichlet_alpha` is the name the search reads; `dirichlet_alpha`
        # is the name callers pass. Kept as an alias rather than renamed so both
        # sides of that boundary stay readable.
        self.root_dirichlet_alpha = self.dirichlet_alpha

        self.window_size = int(self.window_size)

        if self.num_channels is None:
            self.num_channels = NUM_CHANNELS
        if self.num_blocks is None:
            self.num_blocks = NUM_BLOCKS
        if self.value_support_size is None:
            self.value_support_size = VALUE_SUPPORT_SIZE

        if self.search_mode is None:
            self.search_mode = settings.SEARCH_MODE

        if self.search_mode != "muzero":
            # AlphaZero plans with the real simulator, so the dynamics network is
            # never consulted at inference. Unrolling it during training would
            # spend most of the step budget fitting a model nothing uses, and
            # would have its reward and consistency terms compete with the value
            # and policy heads that actually decide play. Train on real states only.
            self.num_unroll_steps = 0
            self.consistency_loss_weight = 0.0

    # Action encoding lives in model.state so the environment wrapper and the
    # config cannot drift apart; re-exported here as staticmethods for callers
    # that reach for them through the config object.
    action_to_index = staticmethod(action_to_index)
    index_to_action = staticmethod(index_to_action)


@dataclass
class RicochetRobotsConfig(MuZeroConfig):
    """This game's run configuration.

    Every value it used to pass explicitly to `super().__init__` is now the
    default on `MuZeroConfig` itself, so this class holds only what is specific
    to Ricochet Robots: the live environment, the curriculum level and whether
    to draw anything.
    """

    render_mode: bool = False

    def __post_init__(self):

        super().__post_init__()

        self.game = None

        self.curriculum_moves = CURRICULUM_START_MOVES

        # One long-lived environment shared by every game. Building a fresh
        # RicochetRobotsEnv per game re-parsed the level JSON and re-initialised
        # pygame each time; `reset()` now genuinely resets, so it is reusable.
        self.environment = None


    def new_game(self):

        # Actors run concurrently, so a shared environment is only safe with one
        # of them. With more, each game builds and owns its own environment.
        if self.num_actors > 1:
            return self._new_owned_game()

        if self.environment is None:
            self.environment = RicochetRobotEnvironment(render_ai=self.render_mode)

        self.game = RicochetRobotsGame(
            self.action_space_size,
            self.discount,
            render_ai=self.render_mode,
            environment=self.environment,
        )

        return self.game


    def _new_owned_game(self):

        return RicochetRobotsGame(
            self.action_space_size,
            self.discount,
            render_ai=self.render_mode,
        )


    def new_episode(self):

        self.environment.reset()


    def set_curriculum_moves(self, moves: int):

        """Ramp the reverse-curriculum depth on the live environment."""

        self.curriculum_moves = moves

        if self.environment is not None:
            self.environment.env.set_curriculum_moves(moves)


    def finish_game(self):

        if self.environment is not None:
            self.environment.close()
            self.environment = None

        logger.info("[Finished Training.] Stay Golden, Ponyboy!")
        
        
def make_ricochet_config(render_ai=False) -> RicochetRobotsConfig:

    return RicochetRobotsConfig(render_mode=render_ai)

     
##### End Helpers ########
##########################
