import numpy as np
import collections, math
from typing import List, Optional

from core.muzero import MuZeroConfig
from core.network import Network, NetworkOutput


##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value
    

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[int], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: int):
    self.history.append(action)

  def last_action(self) -> int:
    return self.history[-1]

  def action_space(self) -> List[int]:
    return [int(i) for i in range(self.action_space_size)]
  

class Node(object):
  
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

##### End Helpers ########
##########################


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.

def run_mcts(config: MuZeroConfig, 
             root: Node, 
             action_history: ActionHistory,
             network: Network):
    
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     history.last_action())
        expand_node(node, history.action_space(), network_output)

        backpropagate(search_path, 
                      network_output.value,
                      config.discount, 
                      min_max_stats)


# Select the child with the highest UCB score.

def select_child(config: MuZeroConfig,  node: Node, min_max_stats: MinMaxStats):
    
    _, action, child = max((ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items())
    return action, child
    
    # The score for a node is based on its value, plus an exploration bonus based on the prior.
    

def ucb_score(config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward + config.discount * child.value())
    else:
        value_score = 0
  
    return prior_score + value_score

    
# At the end of a simulation, we propagate the evaluation all the way up the tree to the root.

def backpropagate(search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats):
    
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value
        

# We expand a node using the value, reward and policy prediction obtained from the neural network.

def expand_node(node: Node, actions: List[int], network_output: NetworkOutput):
    """
    Expands the given node using the network's predicted policy logits, hidden state, and reward.
    
    Args:
        node (Node): The current node to expand.
        actions (List[int]): Legal actions from the current node.
        network_output (NetworkOutput): Contains the predicted reward, policy logits, and hidden state.
    """
    # Assign the hidden state and predicted reward to the node
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward

    # Extract and normalize policy logits for legal actions
    policy = {action: math.exp(network_output.policy_logits[action]) for action in actions}
    policy_sum = sum(policy.values())

    # Create children nodes with normalized prior probabilities
    for action, prob in policy.items():
        node.children[action] = Node(prior=prob / policy_sum)


######### End Self-Play ##########
##################################