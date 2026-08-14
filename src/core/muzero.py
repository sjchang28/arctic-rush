import os
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from src.core.state import (
    RicochetRobotEnvironment,
    RicochetRobotsGame,
    action_to_index,
    index_to_action,
)
from src.core.logger import logger

from src.config import (
    AI_ACTION_SPACE_SIZE,
    AI_OBSERVATION_PLANES,
    AI_OBSERVATION_SPACE_SIZE,
    settings,
)


##########################
####### Helpers ##########

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MuZeroConfig(object):
  
    def __init__(self,
                 action_space_size: int,
                 observation_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_unroll_steps: int,
                 window_size: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 training_episodes: int,
                 hidden_layer_size: int,
                 visit_softmax_temperature_fn,
                 observation_planes: int = AI_OBSERVATION_PLANES,
                 num_channels: int = None,
                 num_blocks: int = None,
                 value_support_size: int = None,
                 known_bounds: Optional[KnownBounds] = None):

        # Self-Play
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        # Training
        self.training_steps = int(500e3)
        self.checkpoint_interval = int(1e2)
        # window_size is how many finished games stay resident in host RAM.
        self.window_size = int(window_size)
        self.batch_size = batch_size

        # num_unroll_steps sets how many recurrent_inference steps live in one
        # autograd graph per batch sample, so it dominates training VRAM.
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.training_episodes = training_episodes

        self.hidden_layer_size = hidden_layer_size

        # Convolutional trunk over board planes (see config.AI_OBSERVATION_SHAPE).
        self.observation_planes = observation_planes
        self.num_channels = num_channels if num_channels is not None else settings.NUM_CHANNELS
        self.num_blocks = num_blocks if num_blocks is not None else settings.NUM_BLOCKS
        self.value_support_size = (
            value_support_size if value_support_size is not None else settings.VALUE_SUPPORT_SIZE
        )

        # Weight of the EfficientZero self-supervised consistency loss.
        self.consistency_loss_weight = settings.CONSISTENCY_LOSS_WEIGHT

        # Value-prefix head (EfficientZero): predicts the cumulative reward over a
        # window rather than the reward of one specific step.
        self.value_prefix_dim = settings.VALUE_PREFIX_DIM
        self.lstm_horizon_len = settings.LSTM_HORIZON_LEN

        # Gumbel root search (see mcts.run_gumbel_mcts).
        self.use_gumbel = settings.USE_GUMBEL
        self.gumbel_num_considered = settings.GUMBEL_NUM_CONSIDERED

        # "muzero" searches the learned dynamics model; "alphazero" searches the
        # real simulator (see mcts.RealEnvironmentModel).
        self.search_mode = settings.SEARCH_MODE

        if self.search_mode != "muzero":
            # AlphaZero plans with the real simulator, so the dynamics network is
            # never consulted at inference. Unrolling it during training would
            # spend most of the step budget fitting a model nothing uses, and
            # would have its reward and consistency terms compete with the value
            # and policy heads that actually decide play. Train on real states only.
            self.num_unroll_steps = 0
            self.consistency_loss_weight = 0.0

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps
    
    
    # Action encoding lives in core.state so the environment wrapper and the
    # config cannot drift apart; re-exported here as staticmethods for callers
    # that reach for them through the config object.
    action_to_index = staticmethod(action_to_index)
    index_to_action = staticmethod(index_to_action)


def visit_softmax_temperature(num_moves, training_steps):
    
    # higher temperature higher exploration
    if training_steps < int(50e3):
        return 2.0
    elif training_steps < int(100e3):
        return 1.0
    else:
        return 0.5
    

class RicochetRobotsConfig(MuZeroConfig):
    
    def __init__(self,
                action_space_size=AI_ACTION_SPACE_SIZE,
                observation_space_size=AI_OBSERVATION_SPACE_SIZE,
                max_moves=settings.MAX_TOTAL_MOVES_PER_GAME,
                discount=1.0,
                dirichlet_alpha=0.25,
                num_simulations=settings.TOTAL_MCTS_EPISODES,
                batch_size=16,
                td_steps=settings.TD_STEPS,
                num_unroll_steps=settings.NUM_UNROLL_STEPS,
                window_size=settings.REPLAY_WINDOW_SIZE,
                num_actors=settings.NUM_ACTORS,
                lr_init=0.005,
                lr_decay_steps=100000,
                training_episodes=settings.TRAINING_EPISODES,
                hidden_layer_size=64,
                visit_softmax_temperature_fn=visit_softmax_temperature,
                render_mode=False):

        super().__init__(
            action_space_size=action_space_size,
            observation_space_size=observation_space_size,
            max_moves=max_moves,
            discount=discount,
            dirichlet_alpha=dirichlet_alpha,
            num_simulations=num_simulations,
            batch_size=batch_size,
            td_steps=td_steps,
            num_unroll_steps=num_unroll_steps,
            window_size=window_size,
            num_actors=num_actors,
            lr_init=lr_init,
            lr_decay_steps=lr_decay_steps,
            training_episodes=training_episodes,
            hidden_layer_size=hidden_layer_size,
            visit_softmax_temperature_fn=visit_softmax_temperature_fn
        )
        
        self.game = None

        self.curriculum_moves = settings.CURRICULUM_START_MOVES

        # One long-lived environment shared by every game. Building a fresh
        # RicochetRobotsEnv per game re-parsed the level JSON and re-initialised
        # pygame each time; `reset()` now genuinely resets, so it is reusable.
        self.environment = None

        self.render_mode = render_mode


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
        
        
    def display_final_stats(self, rewards, losses):

        # Episodes before the replay buffer fills record no loss; drop those
        # pairs rather than plotting NaNs.
        paired = [(r, l) for r, l in zip(rewards, losses) if not np.isnan(l)]
        if not paired:
            logger.warning("No completed training steps to plot.")
            return

        # Sort by rewards, preserving index alignment
        sorted_rewards, sorted_losses = zip(*sorted(paired))
        losses = list(sorted_losses)

        plt.figure()
        plt.plot(sorted_rewards, sorted_losses, marker='o', label='Loss vs Reward')

        # Adding labels and title
        plt.xlabel('Rewards')
        plt.ylabel('Losses')
        plt.title('Loss Decreases as Rewards Increase')
        plt.legend()
        plt.grid(True)
        
        # Histogram - Rewards
        plt.figure()
        plt.boxplot([rewards], tick_labels=['Rewards'])
        
        plt.title("Boxplot of Rewards")
        plt.ylabel("Value")
        
        # Histogram - Losses
        plt.figure()
        plt.boxplot([losses], tick_labels=['Losses'])

        plt.title("Boxplot of Losses")
        plt.ylabel("Value")

        out_dir = os.path.join(settings.LOG_DIR, settings.RUN_ID)
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(1).savefig(os.path.join(out_dir, "loss_vs_reward.png"))
        plt.figure(2).savefig(os.path.join(out_dir, "rewards_boxplot.png"))
        plt.figure(3).savefig(os.path.join(out_dir, "losses_boxplot.png"))
        plt.close("all")
            
        
def make_ricochet_config(render_ai=False) -> MuZeroConfig:

    return RicochetRobotsConfig(render_mode=render_ai)

     
##### End Helpers ########
##########################
