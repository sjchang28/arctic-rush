from typing import List

from game.env import RicochetRobotsEnv

from config import NUMBER_OF_POSSIBLE_MOVES, DIRECTION2INT

   
class Environment(object):
  
    """The environment MuZero is interacting with."""

    def step(self, action):
        
        pass
    
    
class GymEnvironment(Environment):
  
    """The openAI gym environment MuZero is interacting with."""

    def __init__(self):
        
        self.env = None
        
    
    def step(self, action):
        
        pass
    
    
class RicochetRobotEnvironment(GymEnvironment):
  
    """The openAI Ricochet gym environment MuZero is interacting with."""

    def __init__(self, render_ai: bool=False):
        
        super().__init__()
        
        self.render_ai = render_ai
        self.env = RicochetRobotsEnv(render_ai=self.render_ai)
        
    
    def step(self, action):
        
        return self.env.step(action)


    def reset(self):
        
        return self.env.reset()


    def terminal(self):
        
        # Game specific termination rules.
        pass


    def action_to_index(self, robot_index: int, direction: int) -> int:
        
        return robot_index * NUMBER_OF_POSSIBLE_MOVES + direction
    
    
    def legal_actions(self):
        
        encoded_legal_moves = []
        
        legal_moves = self.env.game.robot_manager.get_all_legal_moves(self.env.selected_robot_idx)

        for robot_idx, direction in legal_moves:
            encoded_idx = self.action_to_index(robot_idx, DIRECTION2INT[direction])
            encoded_legal_moves.append(encoded_idx)
        
        return encoded_legal_moves


    def render(self):
        
        if self.render_ai:
            self.env.render()


    def close(self):
        
        self.env.close()
    

class Game(object):
    
    """A single episode of interaction with the environment."""

    def create_environment(self):
        
        pass

class GymGame(Game):
    
    """A single episode of interaction with an openAI gym environment."""
    
    def __init__(self):
        
        self.env = None
        
        
    def create_gym_environment(self):
        
        pass
    
class RicochetRobotsGame(GymGame):
    
    def __init__(self, action_space_size: int, discount: float, render_ai: bool=False):

        super().__init__()
        
        self.render_ai = render_ai
        self.observations = []
        self.environment = self.create_environment()
        
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = False
    
    
    def add_observation(self, observation):
        
        self.observations.append(observation)
    
    
    def create_environment(self) -> RicochetRobotEnvironment:
        
        # Game specific environment. 
        game = RicochetRobotEnvironment(render_ai=self.render_ai)
        self.add_observation(game.env.reset())
        return game


    def is_terminal(self) -> bool:
        
        # Game specific termination rules.
        return self.done


    def legal_actions(self) -> List[int]:
        
        # Game specific calculation of legal actions.
        return self.environment.legal_actions()


    def apply(self, action: int):
        
        """Apply action to the environment and store the result."""
        
        observation, reward, done, _ = self.environment.step(action)
        self.observations.append(observation)
        self.history.append(action)
        self.rewards.append(reward)
        self.done = done


    def store_search_statistics(self, root):
        
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (int(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())


    def make_image(self, state_index: int):
        
        # Game specific feature planes.
        return self.observations[state_index]


    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int, action_space_size: int):
        
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i  # pytype: disable=unsupported-operands

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = None

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, []))
        
        return targets


    def action_history(self):
        
        from core.mcts import ActionHistory
        
        return ActionHistory(self.history, self.action_space_size)
    
    
    def total_rewards(self):
        
        return sum(self.rewards)
    
    
    def total_moves(self):
        
        return len(self.rewards)