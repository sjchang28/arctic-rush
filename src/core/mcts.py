import numpy as np
import collections, math
from typing import List, Optional

from src.core.muzero import MuZeroConfig
from src.core.network import Network, NetworkOutput


##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


def to_scalar(value) -> float:
  """Unwrap a network output into a plain Python float.

  The search tree must never hold torch tensors: a tensor produced inside
  inference keeps its autograd graph (and the GPU activations behind it) alive
  for as long as the node is referenced, and finished games live in the replay
  buffer for hundreds of episodes.
  """
  return value.item() if hasattr(value, 'item') else float(value)


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
  
  def __init__(self, prior: float, logit: float = 0.0):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    # The raw policy logit, kept alongside the normalised prior because the
    # Gumbel root formulas operate on logits, not probabilities.
    self.logit = logit
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0
    # Only the real-environment back-end can know a position is terminal; the
    # learned model has no notion of one.
    self.terminal = False
    # Value-prefix bookkeeping: the LSTM state at this node and the cumulative
    # reward predicted since the window began. `reward` is the difference
    # between this node's prefix and its parent's.
    self.reward_hidden = None
    self.value_prefix = 0.0

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
             network: Network,
             model=None):

    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        _simulate(config, root, action_history, network, min_max_stats, model)

    if model is not None:
        model.restore_root()

    return min_max_stats


# Select the child with the highest UCB score.

def select_child(config: MuZeroConfig,  node: Node, min_max_stats: MinMaxStats):
    
    # Key on the score alone so ties never fall through to comparing Node objects.
    action, child = max(node.children.items(),
                        key=lambda item: ucb_score(config, node, item[1], min_max_stats))
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
    # Assign the hidden state and predicted reward to the node. The hidden state
    # is detached because it is fed back into the dynamics network on the next
    # simulation; the reward is stored as a float so the tree holds no tensors.
    hidden_state = network_output.hidden_state
    node.hidden_state = hidden_state.detach() if hasattr(hidden_state, 'detach') else hidden_state
    node.reward = to_scalar(network_output.reward)

    # Softmax the raw logits over the given (legal) actions. The network emits
    # logits, so this is the single place the policy becomes a distribution;
    # subtracting the max keeps exp() from overflowing on confident policies.
    if not actions:
        return

    logits = [network_output.policy_logits[action] for action in actions]
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_logits)

    # Create children nodes with normalized prior probabilities
    for action, logit, exp_logit in zip(actions, logits, exp_logits):
        node.children[action] = Node(prior=exp_logit / total, logit=logit)


##################################
####### Search back-ends #########
#
# The tree code below is shared by two ways of answering "what does this action
# lead to?".
#
#   SEARCH_MODE=muzero    ask the learned dynamics network (`recurrent_inference`)
#   SEARCH_MODE=alphazero ask the real simulator
#
# Ricochet Robots is deterministic and fully observable and the simulator is a
# cheap loop over board cells, so a learned dynamics model buys nothing here and
# costs a great deal: it has to rediscover wall and robot blocking from scratch
# before the search means anything. The AlphaZero back-end exists as the honest
# control -- same trunk, same replay buffer, same training loop, exact successors.


class RealEnvironmentModel(object):
    """Expands search nodes by replaying actions through the real environment.

    Episode state is four robot positions plus the goal, so snapshot/restore is
    cheap enough to re-run a path from the root on every simulation. That keeps
    the tree free of environment handles and avoids needing an undo operation.
    """

    def __init__(self, environment):

        self.environment = environment
        self.root_snapshot = environment.snapshot()

    def reset_root(self):

        self.root_snapshot = self.environment.snapshot()

    def rollout(self, path_actions):

        self.environment.restore(self.root_snapshot)

        reward = 0.0
        done = False

        for action in path_actions:
            _, reward, done, _ = self.environment.step(action)
            if done:
                break

        legal = [] if done else self.environment.legal_actions()

        return self.environment.observation(), float(reward), done, legal

    def restore_root(self):

        self.environment.restore(self.root_snapshot)


def expand_root(config, root, network, game, model=None):
    """Populate the root node for either search back-end."""

    legal_actions = game.legal_actions()
    if not legal_actions:
        return legal_actions

    network_output = network.initial_inference(game.make_image(-1))
    expand_node(root, legal_actions, network_output)

    # The root starts a value-prefix window: nothing accumulated, zeroed LSTM.
    root.reward_hidden = network_output.reward_hidden
    root.value_prefix = 0.0
    root.reward = 0.0

    if model is not None:
        model.reset_root()

    return legal_actions


def _evaluate_leaf(config, node, path_actions, parent, network, model):
    """Expand `node` and return the value estimate backed up from it."""

    if model is None:
        # Learned model: one step of dynamics from the parent's hidden state.
        #
        # The reward head predicts a *value prefix* -- the cumulative reward since
        # the LSTM state was last reset -- so this node's own reward is the
        # difference against its parent's prefix. The window restarts every
        # `lstm_horizon_len` steps down the path, matching the reset schedule the
        # training unroll uses.
        depth = len(path_actions)

        if (depth - 1) % config.lstm_horizon_len == 0:
            reward_hidden = None       # None -> a freshly zeroed LSTM state
            parent_prefix = 0.0
        else:
            reward_hidden = parent.reward_hidden
            parent_prefix = parent.value_prefix

        network_output = network.recurrent_inference(
            parent.hidden_state, path_actions[-1], reward_hidden)

        expand_node(node, [int(i) for i in range(config.action_space_size)], network_output)

        node.reward_hidden = network_output.reward_hidden
        node.value_prefix = to_scalar(network_output.reward)
        node.reward = node.value_prefix - parent_prefix

        return to_scalar(network_output.value)

    # Real model: replay the path and evaluate the true successor state.
    observation, reward, done, legal = model.rollout(path_actions)

    network_output = network.initial_inference(observation)

    node.reward = reward
    node.terminal = done

    if done:
        # Nothing follows a solved position, so the tree stops here and the
        # bootstrap value is exactly zero rather than a network guess.
        return 0.0

    expand_node(node, legal, network_output)
    node.reward = reward

    return to_scalar(network_output.value)


def _simulate(config, root, action_history, network, min_max_stats, model,
              forced_first_action=None):
    """One simulation from the root, optionally forced through a given action."""

    node = root
    search_path = [node]
    path_actions = []

    if forced_first_action is not None:
        node = root.children[forced_first_action]
        search_path.append(node)
        path_actions.append(forced_first_action)

    while node.expanded():
        action, node = select_child(config, node, min_max_stats)
        search_path.append(node)
        path_actions.append(action)

    if not path_actions:
        return

    parent = search_path[-2]
    value = _evaluate_leaf(config, node, path_actions, parent, network, model)

    backpropagate(search_path, value, config.discount, min_max_stats)


##################################
####### Gumbel MuZero ############
#
# Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022).
#
# Standard MuZero relies on many simulations plus Dirichlet noise at the root and
# offers no guarantee at small budgets: with few simulations the visit-count
# distribution can easily be *worse* than the network's own policy. Gumbel MuZero
# replaces root exploration with Gumbel top-k sampling and sequential halving,
# which does guarantee a policy improvement at any budget -- including budgets of
# 16-32 simulations, which is what a single local GPU can actually afford here.
#
# It also changes what the policy target is. Instead of visit fractions, the
# target is softmax(logits + sigma(completed Q)), which is defined for every
# action rather than only the ones that happened to be visited.

C_VISIT = 50.0
C_SCALE = 1.0


def _sigma(q_value: float, max_visit_count: int) -> float:
    """Monotone transform applied to Q before it is added to the logits."""

    return (C_VISIT + max_visit_count) * C_SCALE * q_value


def _normalised_q(config, child: Node, min_max_stats: MinMaxStats) -> float:

    return min_max_stats.normalize(child.reward + config.discount * child.value())


def _completed_q_values(config, root: Node, min_max_stats: MinMaxStats):
    """Q for every child, with unvisited children filled in by the mixed value.

    The mixed value interpolates between the root's own value estimate and the
    average Q of the children the search did visit, weighted by how much prior
    mass those children carry. This is what makes the improved policy defined
    over the whole action space at tiny simulation budgets.
    """

    total_visits = sum(child.visit_count for child in root.children.values())

    visited_prior_mass = sum(
        child.prior for child in root.children.values() if child.visit_count > 0
    )
    weighted_q = sum(
        child.prior * _normalised_q(config, child, min_max_stats)
        for child in root.children.values() if child.visit_count > 0
    )

    root_value = min_max_stats.normalize(root.value())

    if visited_prior_mass > 0:
        mixed_value = (root_value + (total_visits / visited_prior_mass) * weighted_q) / (1 + total_visits)
    else:
        mixed_value = root_value

    return {
        action: (_normalised_q(config, child, min_max_stats)
                 if child.visit_count > 0 else mixed_value)
        for action, child in root.children.items()
    }


def run_gumbel_mcts(config, root: Node, action_history: ActionHistory, network, model=None):
    """Gumbel root search with sequential halving.

    Returns (chosen_action, improved_policy) where `improved_policy` is a list of
    length `action_space_size` suitable as a training target.
    """

    min_max_stats = MinMaxStats(config.known_bounds)

    actions = list(root.children.keys())
    if not actions:
        return None, [0.0] * config.action_space_size

    # Gumbel noise is the *only* source of root exploration here; Dirichlet noise
    # on the priors is not used and would double-count.
    gumbel = {action: np.random.gumbel() for action in actions}
    logits = {action: root.children[action].logit for action in actions}

    num_considered = min(config.gumbel_num_considered, len(actions))

    # Top-m sampling without replacement is exactly argsort(gumbel + logit).
    considered = sorted(actions, key=lambda a: gumbel[a] + logits[a], reverse=True)[:num_considered]

    simulations_left = config.num_simulations
    num_phases = max(1, int(math.ceil(math.log2(num_considered)))) if num_considered > 1 else 1

    remaining = list(considered)
    phase = 0

    while True:

        # Divide the *remaining* budget across the *remaining* phases, so later
        # phases -- which have fewer surviving actions -- get proportionally more
        # visits each and the whole budget is spent. Dividing the shrinking budget
        # by the fixed total phase count instead left roughly a third of the
        # simulations unused, which at a 32-simulation budget is a lot to waste.
        phases_left = max(1, num_phases - phase)
        visits_per_action = max(1, simulations_left // (phases_left * max(len(remaining), 1)))

        for action in remaining:
            for _ in range(visits_per_action):
                if simulations_left <= 0:
                    break
                _simulate(config, root, action_history, network, min_max_stats,
                          model, forced_first_action=action)
                simulations_left -= 1

        phase += 1

        if len(remaining) <= 1 or simulations_left <= 0:
            break

        max_visit_count = max(child.visit_count for child in root.children.values())
        remaining.sort(
            key=lambda a: gumbel[a] + logits[a] + _sigma(
                _normalised_q(config, root.children[a], min_max_stats), max_visit_count),
            reverse=True,
        )
        remaining = remaining[:max(1, len(remaining) // 2)]

    # Sequential halving can converge to one survivor with budget still unspent.
    # Spending it deepens the principal variation rather than throwing it away.
    while simulations_left > 0 and remaining:
        _simulate(config, root, action_history, network, min_max_stats,
                  model, forced_first_action=remaining[0])
        simulations_left -= 1

    max_visit_count = max(
        (child.visit_count for child in root.children.values()), default=0
    )

    chosen = max(
        remaining,
        key=lambda a: gumbel[a] + logits[a] + _sigma(
            _normalised_q(config, root.children[a], min_max_stats), max_visit_count),
    )

    # Improved policy target: softmax(logit + sigma(completed Q)) over legal
    # actions. Unlike visit fractions this is well defined for every action, not
    # only for the handful the search had budget to touch.
    if model is not None:
        model.restore_root()

    completed_q = _completed_q_values(config, root, min_max_stats)
    scores = {a: logits[a] + _sigma(completed_q[a], max_visit_count) for a in actions}

    max_score = max(scores.values())
    exponentiated = {a: math.exp(s - max_score) for a, s in scores.items()}
    total = sum(exponentiated.values())

    improved_policy = [0.0] * config.action_space_size
    for action, value in exponentiated.items():
        improved_policy[action] = value / total

    return chosen, improved_policy


######### End Self-Play ##########
##################################