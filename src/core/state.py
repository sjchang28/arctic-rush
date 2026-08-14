import copy
from typing import List

import numpy as np

from src.game.env import RicochetRobotsEnv, build_observation

from src.config import (
    DIRECTION2INT,
    NUMBER_OF_DIRECTIONS,
    REWARD_PER_MOVE,
    REWARD_REPEAT_STATE,
    REWARD_SOLVE,
)


def action_to_index(robot_index: int, direction: int) -> int:
    """Encode (robot, direction) as a single AI action id."""

    return robot_index * NUMBER_OF_DIRECTIONS + direction


def index_to_action(index: int):
    """Inverse of `action_to_index`: returns (robot_index, direction)."""

    return divmod(int(index), NUMBER_OF_DIRECTIONS)

   
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

        return action_to_index(robot_index, direction)


    def legal_actions(self):

        encoded_legal_moves = []

        # selected_idx=None -> moves only, no SWITCH: the AI action space is
        # (robot, direction) and the environment honours the robot index directly.
        legal_moves = self.env.game.robot_manager.get_all_legal_moves(selected_idx=None)

        for robot_idx, direction in legal_moves:
            encoded_idx = self.action_to_index(robot_idx, DIRECTION2INT[direction])
            encoded_legal_moves.append(encoded_idx)

        return encoded_legal_moves


    def robot_positions(self):

        return [(r.x, r.y) for r in self.env.robots]


    def snapshot(self):

        return self.env.snapshot()


    def restore(self, state):

        self.env.restore(state)


    def observation(self):

        return self.env._get_obs()


    def wall_planes(self):

        return self.env.wall_planes()


    def target_tuple(self):

        return self.env.target_tuple()


    def optimal_depth(self):

        """Measured optimal solution length of the current episode's start
        position, or None when it was not measured."""

        return self.env.last_optimal_depth


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

    """A single self-play trajectory.

    The trajectory stores *primitive* state -- robot positions per step plus the
    goal -- rather than baked observation planes. Two reasons: 13 float planes per
    step for a hundred steps across a hundred resident games is a lot of host RAM
    for data that reconstructs in microseconds; and hindsight relabelling needs to
    re-render a trajectory under a different goal, which is impossible once the
    goal is baked into the stored planes.
    """

    def __init__(self, action_space_size: int, discount: float, render_ai: bool=False, environment=None):

        super().__init__()

        self.render_ai = render_ai

        # A shared environment is owned by the caller (see RicochetRobotsConfig):
        # it is reset per game and outlives the game, so this game must not close it.
        self.owns_environment = environment is None
        if environment is None:
            self.environment = self.create_environment()
        else:
            self.environment = environment
            self.environment.reset()

        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount
        self.done = False

        # Level geometry is constant, so hold a reference rather than a copy.
        self._wall_planes = self.environment.wall_planes()
        self.target = self.environment.target_tuple()
        self.positions = [self.environment.robot_positions()]

        # How hard this episode actually was. `total_moves()` on its own says
        # nothing -- the gap between it and this is the score that matters.
        self.optimal_depth = self.environment.optimal_depth()


    def create_environment(self) -> RicochetRobotEnvironment:

        # Game specific environment.
        return RicochetRobotEnvironment(render_ai=self.render_ai)


    def release_environment(self):

        """Drop the pygame environment once the game is only training data.

        Idempotent: `RicochetRobotsConfig.finish_game` may also close it.
        Rendering runs keep their environment so the display stays alive.
        """

        if self.render_ai or self.environment is None:
            return

        if self.owns_environment:
            self.environment.close()
        self.environment = None


    def is_terminal(self) -> bool:

        # Game specific termination rules.
        return self.done


    def legal_actions(self) -> List[int]:

        # Game specific calculation of legal actions.
        return self.environment.legal_actions()


    def apply(self, action: int):

        """Apply action to the environment and store the result."""

        _, reward, done, _ = self.environment.step(action)

        self.positions.append(self.environment.robot_positions())
        self.history.append(action)
        self.rewards.append(reward)
        self.done = done


    def store_search_statistics(self, root, policy=None):

        """Record the search's policy target and root value.

        `policy` lets Gumbel search supply its completed-Q improved policy, which
        is defined over every action; falling back to visit fractions is only
        correct when the simulation budget was large enough for them to mean
        something.
        """

        if policy is not None:
            self.child_visits.append(list(policy))
        else:
            sum_visits = sum(child.visit_count for child in root.children.values())
            action_space = (int(index) for index in range(self.action_space_size))
            self.child_visits.append([
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ])

        self.root_values.append(root.value())


    ##### Observations #####

    def make_image(self, state_index: int):

        # Game specific feature planes, rebuilt from stored primitive state.
        index = state_index % len(self.positions)

        return build_observation(
            wall_planes=self._wall_planes,
            robot_positions=self.positions[index],
            target=self.target,
            move_counter=index,
        )


    ##### Targets #####

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                    action_space_size: int, lstm_horizon_len: int = 5):

        """Value / value-prefix / policy targets for `num_unroll_steps + 1` steps.

        Returns numpy arrays plus a policy mask. States past the end of the game
        are absorbing: their value target is 0 and they are masked out of the
        policy loss rather than being handed an empty list the loss cannot use.

        The reward target is a **value prefix**: the cumulative reward since the
        start of the current accumulation window, resetting every
        `lstm_horizon_len` steps to match the LSTM state resets in the training
        unroll and in the search tree.
        """

        values = np.zeros(num_unroll_steps + 1, dtype=np.float32)
        rewards = np.zeros(num_unroll_steps + 1, dtype=np.float32)
        policies = np.zeros((num_unroll_steps + 1, action_space_size), dtype=np.float32)
        policy_mask = np.zeros(num_unroll_steps + 1, dtype=np.float32)

        value_prefix = 0.0

        for k, current_index in enumerate(range(state_index, state_index + num_unroll_steps + 1)):

            # The value target is the discounted root value of the search tree N
            # steps into the future, plus the discounted sum of all rewards until then.
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount ** td_steps
            else:
                value = 0.0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            # Value prefix: reset at the start of each window, then accumulate.
            if k > 0:
                if (k - 1) % lstm_horizon_len == 0:
                    value_prefix = 0.0
                if 0 < current_index <= len(self.rewards):
                    value_prefix += self.rewards[current_index - 1]
                rewards[k] = value_prefix

            if current_index < len(self.root_values):
                values[k] = value
                policies[k] = self.child_visits[current_index]
                policy_mask[k] = 1.0

        return values, rewards, policies, policy_mask


    def make_actions(self, state_index: int, num_unroll_steps: int):

        """Actions for the unroll, right-padded past the end of the game.

        Padding actions are masked out of the consistency loss; they still drive
        the dynamics network so the unroll keeps a fixed shape.
        """

        actions = np.zeros(num_unroll_steps, dtype=np.int64)
        mask = np.zeros(num_unroll_steps, dtype=np.float32)

        for k, index in enumerate(range(state_index, state_index + num_unroll_steps)):
            if index < len(self.history):
                actions[k] = self.history[index]
                mask[k] = 1.0

        return actions, mask


    def make_next_observations(self, state_index: int, num_unroll_steps: int):

        """Observations the unrolled hidden states should match, for the
        EfficientZero consistency loss."""

        if num_unroll_steps == 0:
            # SEARCH_MODE=alphazero trains no dynamics model, so there is no
            # unroll and nothing to be consistent with.
            return (
                np.zeros((0,) + self.make_image(0).shape, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        observations = np.stack([
            self.make_image(min(state_index + k + 1, len(self.positions) - 1))
            for k in range(num_unroll_steps)
        ])

        mask = np.array(
            [1.0 if state_index + k + 1 < len(self.positions) else 0.0
             for k in range(num_unroll_steps)],
            dtype=np.float32,
        )

        return observations, mask


    ##### Hindsight relabelling #####

    def hindsight_relabel(self, rng=None):

        """Return a copy of this trajectory re-labelled with an achieved goal.

        The target is resampled every episode, which makes this a goal-conditioned
        task -- the canonical HER setting. A failed episode still demonstrates how
        to reach wherever the robots actually ended up, so re-labelling the goal to
        an achieved square converts a sparse-reward failure into a success. On a
        puzzle where random play almost never reaches the real goal, this is the
        difference between a learning signal and none.

        Returns None when no usable relabelling exists (nothing moved, or the
        episode already succeeded).
        """

        rng = rng if rng is not None else np.random

        if self.done or not self.history:
            return None

        # Candidate goals: squares a robot arrived at after some move, other than
        # the square it was already on. Pick one and truncate the trajectory there.
        candidates = []
        for step, action in enumerate(self.history):
            robot_idx, _ = index_to_action(action)
            arrived = self.positions[step + 1][robot_idx]
            if arrived != self.positions[step][robot_idx]:
                candidates.append((step, robot_idx, arrived))

        if not candidates:
            return None

        step, robot_idx, arrived = candidates[rng.randint(len(candidates))]

        relabelled = copy.copy(self)
        relabelled.environment = None
        relabelled.target = (arrived[0], arrived[1], robot_idx)
        # Measured against the original goal, so meaningless under the new one.
        relabelled.optimal_depth = None
        relabelled.positions = self.positions[:step + 2]
        relabelled.history = self.history[:step + 1]
        relabelled.child_visits = self.child_visits[:step + 1]
        relabelled.done = True

        # Recompute rewards under the new goal: the truncated final move now solves.
        rewards = []
        visited = {tuple(self.positions[0])}
        for i in range(len(relabelled.history)):
            reward = REWARD_PER_MOVE
            if i == len(relabelled.history) - 1:
                reward += REWARD_SOLVE
            else:
                key = tuple(relabelled.positions[i + 1])
                if key in visited:
                    reward += REWARD_REPEAT_STATE
                else:
                    visited.add(key)
            rewards.append(reward)

        relabelled.rewards = rewards

        # Note the one approximation here: `child_visits` was searched against the
        # original goal, so the policy targets are the behaviour policy rather
        # than a policy re-searched under the new goal. The action sequence does
        # reach the relabelled goal, so the targets remain informative, but they
        # are not the improved policy a fresh search would give. Re-searching
        # every relabelled trajectory would cost as much as generating it.

        # Root values were searched against the *original* goal, so they are not
        # valid bootstraps for the relabelled return. Recompute them as the actual
        # discounted return from each position, which is exact here because the
        # relabelled trajectory is known to end in a solve.
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.discount * running
            returns.append(running)
        relabelled.root_values = list(reversed(returns))

        return relabelled


    ##### Misc #####

    def action_history(self):

        from core.mcts import ActionHistory

        return ActionHistory(self.history, self.action_space_size)


    def total_rewards(self):

        return sum(self.rewards)


    def total_moves(self):

        return len(self.rewards)
