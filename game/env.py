import gymnasium
from gymnasium import spaces
import numpy as np

from game.penguins import Penguin
from game.game import Game

from config import BOARD_WIDTH, BOARD_HEIGHT, INT2DIRECTION, NUMBER_OF_PENGUINS, NUMBER_OF_POSSIBLE_MOVES

class RicochetPenguinsEnv(gymnasium.Env):
    
    """
    Custom Environment for Ricochet penguins compatible with OpenAI Gym.
    AI controls all 4 penguins at once, moving each penguin in turn.
    """

    def __init__(self, render_mode=None):
        
        super().__init__()

        # Init board
        self.penguins = [
            Penguin("red"),
            Penguin("blue"),
            Penguin("green"),
            Penguin("yellow")
        ]
        self.render_mode = render_mode
        self.game = Game(self.penguins)

        self.visited_states = set()
        
        self.selected_penguin_idx = 0
        self.move_counter = 0

        # Action space: 4 penguins, each with 5 actions (Up, Down, Left, Right, Switch)
        # Action: [penguin_index, direction (0=Up, 1=Down, 2=Left, 3=Right, 4=Switch)]
        self.action_space = spaces.MultiDiscrete([NUMBER_OF_PENGUINS, 5])

        # Observation space: flattened board state + current_target (x,y,color) + positions of 4 penguins + penguin_idx + move_counter
        self.observation_space = spaces.Box(
            low=0, high=255, shape=((BOARD_WIDTH * BOARD_HEIGHT * 4) + (3) + (NUMBER_OF_PENGUINS * 3) + (1) + (1),), dtype=np.float32
        )

        self.reset()
        
        self.active_gameplay = False
        
        
    def reset(self):
        
        # Initialize the game state
        self.game.penguin_manager._initialize_penguin_positions()
        self.game.target_deck.set_new_target()
        self.visited_states.clear()
        
        self.selected_penguin_idx = 0
        self.move_counter = 0 

        return self._get_obs()


    def _get_obs(self):
        
        flat_walls = self.game.board.flatten_walls()
        flat_target = self.game.target_deck.flatten_current_target()
        flat_penguins = self.game.penguin_manager.flatten_penguins()
        flat_selected_penguin = np.array([self.selected_penguin_idx])
        flat_move_counter = np.array([self.move_counter])

        return np.concatenate([flat_walls, flat_target, flat_penguins, flat_selected_penguin, flat_move_counter], axis=0)


    def step(self, action):
        penguin_idx, direction = divmod(action, NUMBER_OF_POSSIBLE_MOVES)
        done = False
        reward = 0

        # Handle tabbing (direction 4 means switch penguin)
        if direction == 4:
            if penguin_idx != self.selected_penguin_idx:
                self.selected_penguin_idx = penguin_idx
            reward = 0
            done = False
        else:
            selected_penguin = self.penguins[self.selected_penguin_idx]
            state_key = (tuple((p.x, p.y) for p in self.penguins), self.selected_penguin_idx)

            # Move until blocked
            if selected_penguin.move_until_blocked(
                simulated=False,
                direction=INT2DIRECTION[int(direction)],
                board=self.game.board,
                other_penguins=self.game.penguin_manager.penguins
            ):
                self.move_counter += 1

                if selected_penguin.is_target_reached(
                    self.game.target_deck.current_target
                ):
                    reward = max(1000 - self.move_counter * 10, 100)
                    done = True

            # Check if state was repeated
            if state_key in self.visited_states:
                reward = -5
            else:
                self.visited_states.add(state_key)

        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode=None):
        
        if mode == 'human':
            self.active_gameplay = True
            self.game.render_ai_environment(self.selected_penguin_idx, self.move_counter)
    
    
    def close(self):
        
        if self.active_gameplay:
            self.game.close()
