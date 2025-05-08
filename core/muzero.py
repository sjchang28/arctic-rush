import collections
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from core.state import RicochetRobotsGame

from config import BOARD_WIDTH, BOARD_HEIGHT, NUMBER_OF_POSSIBLE_MOVES, NUMBER_OF_ROBOTS, TOTAL_MCTS_EPISODES, MAX_TOTAL_MOVES_PER_GAME


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
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 training_episodes: int,
                 hidden_layer_size: int,
                 visit_softmax_temperature_fn,
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
        self.window_size = int(500) # 1000
        self.batch_size = batch_size
        self.num_unroll_steps = MAX_TOTAL_MOVES_PER_GAME
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.training_episodes = training_episodes

        self.hidden_layer_size = hidden_layer_size

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps
    
    
    def action_to_index(robot_index: int, direction: int) -> int:
        
        return robot_index * NUMBER_OF_POSSIBLE_MOVES + direction


    def index_to_action(index: int) -> Tuple[int, int]:
        
        return divmod(index, NUMBER_OF_POSSIBLE_MOVES)


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
                action_space_size=(NUMBER_OF_ROBOTS * NUMBER_OF_POSSIBLE_MOVES),
                observation_space_size=(BOARD_WIDTH * BOARD_HEIGHT * 4) + 3 + (4 * 3) + 1 + 1,
                max_moves=MAX_TOTAL_MOVES_PER_GAME,
                discount=1.0,
                dirichlet_alpha=0.25,
                num_simulations=TOTAL_MCTS_EPISODES,
                batch_size=16,
                td_steps=MAX_TOTAL_MOVES_PER_GAME,  # Same as max_moves
                num_actors=1,
                lr_init=0.005,
                lr_decay_steps=100000,
                training_episodes=120,
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
            num_actors=num_actors,
            lr_init=lr_init,
            lr_decay_steps=lr_decay_steps,
            training_episodes=training_episodes,
            hidden_layer_size=hidden_layer_size,
            visit_softmax_temperature_fn=visit_softmax_temperature_fn
        )
        
        self.game = None
        
        self.render_mode = render_mode
  
  
    def new_game(self):
        
        self.game = RicochetRobotsGame(self.action_space_size, self.discount, render_ai=self.render_mode)
            
        return self.game
    
    
    def new_episode(self):
        
        self.game.environment.reset()
    
    
    def finish_game(self):
        
        self.game.environment.close()
        
        print("[Finished Training.] ☞ó ͜つò☞ Stay Golden, Ponyboy!")
        
        
    def display_final_stats(self, rewards, losses):
        
        # Sort by rewards, preserving index alignment
        combined = sorted(zip(rewards, losses))
        sorted_rewards, sorted_losses = zip(*combined)

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
        plt.boxplot([rewards], labels=['Rewards'])
        
        plt.title("Boxplot of Rewards")
        plt.ylabel("Value")
        
        # Histogram - Losses
        plt.figure()
        plt.boxplot([losses], labels=['Losses'])
        
        plt.title("Boxplot of Losses")
        plt.ylabel("Value")
        
        plt.show()   
            
        
def make_ricochet_config(render_ai=False) -> MuZeroConfig:

    return RicochetRobotsConfig(render_mode=render_ai)

     
##### End Helpers ########
##########################
