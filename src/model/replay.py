"""Prioritised replay buffer.

Split out of `network.py`, which held a replay buffer, a torch module and a
checkpoint manager in one 674-line file. The three share no state and change for
unrelated reasons.
"""

import threading

import numpy as np
import torch

from src.model.config import (
    HER_FRACTION,
    LSTM_HORIZON_LEN,
    PRIORITIZED_REPLAY_ALPHA,
    PRIORITIZED_REPLAY_BETA,
    USE_HER,
)
from src.model.device import gpu_device
from src.model.muzero import RicochetRobotsConfig
from src.model.state import RicochetRobotsGame


class ReplayBuffer(object):

    def __init__(self, config: RicochetRobotsConfig):

        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

        # Priority per game, refreshed from the training loss (see update_priorities).
        self.priorities = []
        self.alpha = PRIORITIZED_REPLAY_ALPHA
        self.beta = PRIORITIZED_REPLAY_BETA

        # Self-play actors are threads (see train.launch_selfplay_jobs), so
        # concurrent save_game calls can interleave the pop/append below. Note
        # the GIL means those actors overlap I/O only -- they are not a way to
        # parallelise CPU-bound search. Use processes for that.
        self._lock = threading.Lock()

    def save_game(self, game):

        with self._lock:
            if len(self.buffer) >= self.window_size:
                self.buffer.pop(0)
                self.priorities.pop(0)
            self.buffer.append(game)
            # New games enter at the current maximum priority so they are seen
            # at least once before their priority is estimated.
            self.priorities.append(max(self.priorities, default=1.0))

    def __len__(self):

        return len(self.buffer)

    def sample_game_indices(self, count: int):
        """Sample game indices proportionally to priority, with IS weights."""

        priorities = np.asarray(self.priorities, dtype=np.float64)

        if self.alpha <= 0 or priorities.sum() <= 0:
            probabilities = np.full(len(priorities), 1.0 / len(priorities))
        else:
            scaled = priorities ** self.alpha
            probabilities = scaled / scaled.sum()

        indices = np.random.choice(len(probabilities), size=count, p=probabilities)

        weights = (1.0 / (len(probabilities) * probabilities[indices])) ** self.beta
        weights = weights / weights.max()

        return indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):

        with self._lock:
            for index, priority in zip(indices, priorities):
                if index < len(self.priorities):
                    self.priorities[index] = float(priority) + 1e-6

    def sample_position(self, game) -> int:

        # A game solved on its first move has a single position; `total_moves()-1`
        # is then an empty range, so clamp rather than raise.
        num_positions = max(game.total_moves() - 1, 1)
        return int(np.random.choice(num_positions))

    def last_game(self) -> RicochetRobotsGame:

        return self.buffer[-1]

    def sample_batch(self, num_unroll_steps: int, td_steps: int, action_space_size: int):
        """Draw a batch as stacked tensors.

        The previous implementation returned a list of per-sample tuples that the
        training loop then looped over one at a time, so a batch of 16 ran 16
        separate forward passes. Everything here is stacked and unrolled once.
        """

        indices, weights = self.sample_game_indices(self.batch_size)

        images, next_images, next_masks = [], [], []
        actions, action_masks = [], []
        target_values, target_rewards, target_policies, policy_masks = [], [], [], []

        use_her = USE_HER and HER_FRACTION > 0

        for index in indices:
            game = self.buffer[index]

            # Hindsight relabelling: a failed episode still shows how to reach the
            # square the robot actually stopped on, which is a real success under
            # a relabelled goal (see RicochetRobotsGame.hindsight_relabel).
            if use_her and not game.is_terminal() and np.random.rand() < HER_FRACTION:
                relabelled = game.hindsight_relabel()
                if relabelled is not None:
                    game = relabelled

            position = self.sample_position(game)

            images.append(game.make_image(position))

            game_actions, action_mask = game.make_actions(position, num_unroll_steps)
            actions.append(game_actions)
            action_masks.append(action_mask)

            values, rewards, policies, policy_mask = game.make_target(
                position, num_unroll_steps, td_steps, action_space_size,
                lstm_horizon_len=LSTM_HORIZON_LEN,
            )
            target_values.append(values)
            target_rewards.append(rewards)
            target_policies.append(policies)
            policy_masks.append(policy_mask)

            observations, next_mask = game.make_next_observations(position, num_unroll_steps)
            next_images.append(observations)
            next_masks.append(next_mask)

        def stack(arrays, dtype=torch.float32):
            return torch.as_tensor(np.stack(arrays), dtype=dtype, device=gpu_device())

        return {
            "indices": indices,
            "weights": torch.as_tensor(weights, device=gpu_device()),
            "observations": stack(images),
            "next_observations": stack(next_images),
            "next_mask": stack(next_masks),
            "actions": stack(actions, dtype=torch.long),
            "action_mask": stack(action_masks),
            "target_values": stack(target_values),
            "target_rewards": stack(target_rewards),
            "target_policies": stack(target_policies),
            "policy_mask": stack(policy_masks),
        }
