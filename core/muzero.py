import collections
from typing import Tuple, Optional

from config import BOARD_WIDTH, BOARD_HEIGHT, NUMBER_OF_PENGUINS, NUMBER_OF_POSSIBLE_MOVES, INDEX_COLOR_MAP, UP, RIGHT, DOWN, LEFT, TOTAL_MCTS_EPISODES, MAX_TOTAL_MOVES_PER_GAME

from game.penguins import Penguin
from game.target import Target
from core.state import RicochetGame


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
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1000)
        self.batch_size = batch_size
        self.num_unroll_steps = 500
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.training_episodes = training_episodes

        self.hidden_layer_size = hidden_layer_size

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

    
    def reconstruct_walls(self, flat):
        
        walls = []
        idx = 0
        for row in range(BOARD_HEIGHT):
            wall_row = []
            for col in range(BOARD_WIDTH):
                cell = {
                    UP: bool(flat[idx]),
                    RIGHT: bool(flat[idx+1]),
                    DOWN: bool(flat[idx+2]),
                    LEFT: bool(flat[idx+3]),
                }
                wall_row.append(cell)
                idx += 4
            walls.append(wall_row)
        return walls


    def reconstruct_penguins(self, flat):
        
        penguins = []
        for i in range(0, len(flat), 3):
            x = int(flat[i])
            y = int(flat[i+1])
            # You should define INDEX_TO_COLOR = {0: "red", ...}
            color = INDEX_COLOR_MAP[int(flat[i+2])]
            penguins.append(Penguin(x, y, color))
        return penguins


    def reconstruct_target(self, flat):
        x = int(flat[0])
        y = int(flat[1])
        color = INDEX_COLOR_MAP[int(flat[2])]
        return Target(x, y, color)


    def decode_state(self, encoded_state):
        
        # Constants
        board_flat_size = BOARD_WIDTH * BOARD_HEIGHT * 4  # 4 directions per cell
        penguin_flat_size = NUMBER_OF_PENGUINS * 3        # x, y, color
        target_flat_size = 3                              # x, y, color_idx

        # * Board
        board_flat = encoded_state[0:board_flat_size]
        walls = self.reconstruct_walls(board_flat)

        # * Penguins
        start = board_flat_size
        end = start + penguin_flat_size
        penguins_flat = encoded_state[start:end]
        penguins = self.reconstruct_penguins(penguins_flat)

        # * Target
        start = end
        end = start + target_flat_size
        target_flat = encoded_state[start:end]
        target = self.reconstruct_target(target_flat)

        # * Selected Penguin
        selected_penguin_idx = int(encoded_state[-2])

        # * Move Counter
        move_counter = int(encoded_state[-1])

        return walls, penguins, target, selected_penguin_idx, move_counter
    
    
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
    

class RicochetConfig(MuZeroConfig):
    
    def __init__(self,
                action_space_size=(NUMBER_OF_PENGUINS * NUMBER_OF_POSSIBLE_MOVES),
                observation_space_size=(BOARD_WIDTH * BOARD_HEIGHT * 4) + 3 + (4 * 3) + 1 + 1,
                max_moves=MAX_TOTAL_MOVES_PER_GAME,
                discount=1.0,
                dirichlet_alpha=0.25,
                num_simulations=TOTAL_MCTS_EPISODES,
                batch_size=128,
                td_steps=MAX_TOTAL_MOVES_PER_GAME,  # Same as max_moves
                num_actors=1,
                lr_init=0.005,
                lr_decay_steps=100000,
                training_episodes=100000,
                hidden_layer_size=64,
                visit_softmax_temperature_fn=visit_softmax_temperature):

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
  
  
    def new_game(self):

        self.game = RicochetGame(self.action_space_size, self.discount)
        return self.game
    
    
    def new_episode(self):
        
        self.game.environment.reset()
    
    
    def finish_game(self):
        
        self.game.environment.close()
    
        
        
def make_ricochet_config() -> MuZeroConfig:

    return RicochetConfig()

     
##### End Helpers ########
##########################
