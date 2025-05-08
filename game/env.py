import gymnasium
from gymnasium import spaces
import numpy as np

from game.robots import Robot
from game.game import AI_Game

from config import BOARD_WIDTH, BOARD_HEIGHT, INT2DIRECTION, NUMBER_OF_ROBOTS, NUMBER_OF_POSSIBLE_MOVES

class RicochetRobotsEnv(gymnasium.Env):
    
    """
    Custom Environment for Ricochet robots compatible with OpenAI Gym.
    AI controls all 4 robots at once, moving each robot in turn.
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
        
        self.selected_robot_idx = 0
        self.move_counter = 0

        # Action space: 4 robots, each with 5 actions (Up, Down, Left, Right, Switch)
        # Action: [robot_index, direction (0=Up, 1=Down, 2=Left, 3=Right, 4=Switch)]
        self.action_space = spaces.MultiDiscrete([NUMBER_OF_ROBOTS, 5])

        # Observation space: flattened board state + current_target (x,y,color) + positions of 4 robots + robot_idx + move_counter
        self.observation_space = spaces.Box(
            low=0, high=255, shape=((BOARD_WIDTH * BOARD_HEIGHT * 4) + (3) + (NUMBER_OF_ROBOTS * 3) + (1) + (1),), dtype=np.float32
        )

        self.reset()
        
        
    def reset(self):
        
        # Initialize the game state
        self.game.robot_manager._initialize_robot_positions()
        self.game.target_deck.set_new_target()
        self.visited_states.clear()
        
        self.selected_robot_idx = 0
        self.move_counter = 0 

        return self._get_obs()


    def _get_obs(self):
        
        flat_walls = self.game.board.flatten_walls()
        flat_target = self.game.target_deck.flatten_current_target()
        flat_robots = self.game.robot_manager.flatten_robots()
        flat_selected_robot = np.array([self.selected_robot_idx])
        flat_move_counter = np.array([self.move_counter])

        return np.concatenate([flat_walls, flat_target, flat_robots, flat_selected_robot, flat_move_counter], axis=0)


    def step(self, action):
        robot_idx, direction = divmod(action, NUMBER_OF_POSSIBLE_MOVES)
        done = False
        reward = 0

        # Handle tabbing (direction 4 means switch robot)
        if direction == 4:
            if robot_idx != self.selected_robot_idx:
                self.selected_robot_idx = robot_idx
            reward = 0
            done = False
        else:
            selected_robot = self.robots[self.selected_robot_idx]
            state_key = (tuple((p.x, p.y) for p in self.robots), self.selected_robot_idx)

            # Move until blocked
            if selected_robot.move_until_blocked(
                simulated=False,
                direction=INT2DIRECTION[int(direction)],
                board=self.game.board,
                other_robots=self.game.robot_manager.robots
            ):
                self.move_counter += 1

                if selected_robot.is_target_reached(
                    self.game.target_deck.current_target
                ):
                    reward = max(1000 - (self.move_counter * 10), 100)
                    done = True

            # Check if state was repeated
            if state_key in self.visited_states:
                reward = -5
            else:
                self.visited_states.add(state_key)

        obs = self._get_obs()
        return obs, reward, done, {}


    def render(self):
        
        if self.render_ai:
            self.game.render_ai_environment(self.selected_robot_idx, self.move_counter)
    
    
    def close(self):
        
        if self.render_ai:
            self.game.close()
