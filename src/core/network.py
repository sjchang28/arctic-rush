import json
import math
import os
import threading
import numpy as np
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from src.core.muzero import RicochetRobotsConfig
from src.core.state import RicochetRobotsGame
from src.core.support import support_to_scalar
from src.core.logger import logger

from src.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUMBER_OF_DIRECTIONS,
    NUMBER_OF_ROBOTS,
    settings,
)


##########################
####### Helpers ##########

GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# An action is encoded for the dynamics network as 8 broadcast planes: a one-hot
# over the robot and a one-hot over the direction. Encoding the flat action id as
# 16 planes instead would be both larger and structureless.
ACTION_PLANES = NUMBER_OF_ROBOTS + NUMBER_OF_DIRECTIONS


class NetworkOutput(NamedTuple):

    # Scalars in real units, for the search tree.
    value: torch.Tensor
    reward: torch.Tensor

    # Logits, for training.
    policy_logits: Dict[int, float]
    policy_tensor: torch.Tensor
    value_logits: torch.Tensor
    reward_logits: torch.Tensor

    hidden_state: torch.Tensor

    # LSTM state of the value-prefix head, carried along a search path.
    reward_hidden: Optional[tuple] = None


def scale_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
    """Min-max scale each sample's hidden state into [0, 1].

    From the MuZero appendix. Without it the state drifts in magnitude across
    recurrent applications of the dynamics network, and `num_unroll_steps`
    applications is enough for that to matter.
    """

    batch = hidden_state.shape[0]
    flat = hidden_state.view(batch, -1)

    minimum = flat.min(dim=1, keepdim=True)[0]
    maximum = flat.max(dim=1, keepdim=True)[0]

    scaled = (flat - minimum) / (maximum - minimum + 1e-5)
    return scaled.view_as(hidden_state)


def encode_action_planes(action, batch_size: int, device) -> torch.Tensor:
    """Broadcast (robot, direction) one-hots over the board."""

    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, device=device)
    action = action.to(device).long().reshape(-1)

    if action.numel() == 1 and batch_size > 1:
        action = action.expand(batch_size)

    robot_idx = torch.div(action, NUMBER_OF_DIRECTIONS, rounding_mode="floor")
    direction = action % NUMBER_OF_DIRECTIONS

    robot_one_hot = F.one_hot(robot_idx, NUMBER_OF_ROBOTS).float()
    direction_one_hot = F.one_hot(direction, NUMBER_OF_DIRECTIONS).float()

    planes = torch.cat((robot_one_hot, direction_one_hot), dim=1)
    return planes[:, :, None, None].expand(-1, -1, BOARD_HEIGHT, BOARD_WIDTH)


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x):

        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + x)


def _trunk(in_channels: int, channels: int, blocks: int) -> nn.Sequential:

    layers = [
        nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(1, channels),
        nn.ReLU(inplace=True),
    ]
    layers += [ResidualBlock(channels) for _ in range(blocks)]
    return nn.Sequential(*layers)


class _Head(nn.Module):
    """1x1 conv bottleneck into a linear output."""

    def __init__(self, channels: int, reduced: int, out_features: int):

        super().__init__()

        self.conv = nn.Conv2d(channels, reduced, 1, bias=False)
        self.norm = nn.GroupNorm(1, reduced)
        self.fc = nn.Linear(reduced * BOARD_HEIGHT * BOARD_WIDTH, out_features)

    def forward(self, x):

        out = F.relu(self.norm(self.conv(x)))
        return self.fc(out.flatten(1))


class ReplayBuffer(object):

    def __init__(self, config: RicochetRobotsConfig):

        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

        # Priority per game, refreshed from the training loss (see update_priorities).
        self.priorities = []
        self.alpha = settings.PRIORITIZED_REPLAY_ALPHA
        self.beta = settings.PRIORITIZED_REPLAY_BETA

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

        use_her = settings.USE_HER and settings.HER_FRACTION > 0

        for index in indices:
            game = self.buffer[index]

            # Hindsight relabelling: a failed episode still shows how to reach the
            # square the robot actually stopped on, which is a real success under
            # a relabelled goal (see RicochetRobotsGame.hindsight_relabel).
            if use_her and not game.is_terminal() and np.random.rand() < settings.HER_FRACTION:
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
                lstm_horizon_len=settings.LSTM_HORIZON_LEN,
            )
            target_values.append(values)
            target_rewards.append(rewards)
            target_policies.append(policies)
            policy_masks.append(policy_mask)

            observations, next_mask = game.make_next_observations(position, num_unroll_steps)
            next_images.append(observations)
            next_masks.append(next_mask)

        def stack(arrays, dtype=torch.float32):
            return torch.as_tensor(np.stack(arrays), dtype=dtype, device=GPU_DEVICE)

        return {
            "indices": indices,
            "weights": torch.as_tensor(weights, device=GPU_DEVICE),
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


class Network(nn.Module):

    def __init__(self, config):

        super(Network, self).__init__()

        self.action_space_size = config.action_space_size
        self.support_size = config.value_support_size
        self.support_dim = 2 * config.value_support_size + 1

        channels = config.num_channels
        blocks = config.num_blocks

        self.tot_training_steps = 0

        # Representation: board planes -> hidden state, kept spatial. The board is
        # spatial and wall/robot blocking is a long-range relation between cells;
        # flattening it into a vector for an MLP threw that structure away.
        self.representation_net = _trunk(config.observation_planes, channels, blocks)

        # Dynamics: (hidden state, action planes) -> next hidden state
        self.dynamics_net = _trunk(channels + ACTION_PLANES, channels, blocks)

        # Prediction heads. Value and reward are categorical over a transformed
        # support rather than scalar regressions (see core.support).
        self.policy_net = _Head(channels, 4, config.action_space_size)
        self.value_net = _Head(channels, 4, self.support_dim)

        # Value prefix (EfficientZero) rather than a per-step reward head. The
        # LSTM predicts the *cumulative* reward accumulated since its state was
        # last reset, so the model no longer has to pin down exactly which step a
        # reward landed on -- only that it arrives somewhere in the window. The
        # state is carried along a search path and reset every
        # `lstm_horizon_len` steps, in both the tree and the training unroll.
        self.value_prefix_dim = config.value_prefix_dim
        self.lstm_horizon_len = config.lstm_horizon_len

        self.reward_feature = _Head(channels, 4, self.value_prefix_dim)
        self.value_prefix_lstm = nn.LSTM(self.value_prefix_dim, self.value_prefix_dim)
        self.value_prefix_head = nn.Sequential(
            nn.LayerNorm(self.value_prefix_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.value_prefix_dim, self.support_dim),
        )

        # EfficientZero's self-supervised consistency loss: a SimSiam projector
        # and predictor over the hidden state. This is what forces the dynamics
        # network to actually model the board rather than drifting into whatever
        # minimises value error.
        projection_dim = channels * BOARD_HEIGHT * BOARD_WIDTH
        self.projector = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )

        # The boot checklist (core.boot.report_ready) reports the device once the
        # network exists; this fires per Network construction.
        logger.debug(f"Ready to begin training on device {str(GPU_DEVICE).upper()} ...")
        self.to(GPU_DEVICE)

    ##### Persistence #####

    def save_model(self, path=settings.WEIGHTS_FILE_PATH):

        run_dir = os.path.join(settings.MODEL_DIR, settings.RUN_ID)
        os.makedirs(run_dir, exist_ok=True)
        model_weights_file = os.path.join(run_dir, path)

        torch.save(self.state_dict(), model_weights_file)

    def load_model(self, path=settings.WEIGHTS_FILE_PATH):

        model_weights_file = os.path.join(settings.MODEL_DIR, settings.RUN_ID, path)

        self.load_state_dict(torch.load(model_weights_file))
        self.eval()

    ##### Core functions #####

    def representation(self, observation: torch.Tensor) -> torch.Tensor:

        return scale_hidden_state(self.representation_net(observation))

    def dynamics(self, hidden_state: torch.Tensor, action) -> torch.Tensor:

        action_planes = encode_action_planes(action, hidden_state.shape[0], hidden_state.device)
        combined = torch.cat((hidden_state, action_planes), dim=1)

        return scale_hidden_state(self.dynamics_net(combined))

    def initial_reward_hidden(self, batch_size: int, device=None):
        """A zeroed value-prefix LSTM state: the start of an accumulation window."""

        device = device if device is not None else GPU_DEVICE
        zeros = torch.zeros(1, batch_size, self.value_prefix_dim, device=device)

        return (zeros, zeros.clone())

    def value_prefix(self, hidden_state: torch.Tensor, reward_hidden):
        """Cumulative-reward logits for this state, plus the advanced LSTM state."""

        if reward_hidden is None:
            reward_hidden = self.initial_reward_hidden(hidden_state.shape[0], hidden_state.device)

        feature = self.reward_feature(hidden_state).unsqueeze(0)  # [1, B, D]
        output, next_reward_hidden = self.value_prefix_lstm(feature, reward_hidden)

        return self.value_prefix_head(output.squeeze(0)), next_reward_hidden

    def project(self, hidden_state: torch.Tensor, with_predictor: bool = True) -> torch.Tensor:
        """SimSiam projection of a hidden state, used by the consistency loss."""

        projection = self.projector(hidden_state.flatten(1))
        return self.predictor(projection) if with_predictor else projection

    def _as_batch(self, observation) -> torch.Tensor:

        if not isinstance(observation, torch.Tensor):
            observation = torch.as_tensor(np.asarray(observation), dtype=torch.float32)

        # Move unconditionally: a caller passing an already-built CPU tensor used
        # to skip the transfer and blow up inside the first layer.
        observation = observation.to(GPU_DEVICE, dtype=torch.float32)

        if observation.dim() == 3:
            observation = observation.unsqueeze(0)  # Add batch dimension

        return observation

    def _predict(self, hidden_state, reward_logits, reward_hidden=None) -> NetworkOutput:

        value_logits = self.value_net(hidden_state)
        policy_logits = self.policy_net(hidden_state)

        # Raw logits everywhere. The search normalises the policy itself, over
        # legal actions only (see mcts.expand_node).
        policy_dict = {a: policy_logits[0, a].item() for a in range(self.action_space_size)}

        return NetworkOutput(
            value=support_to_scalar(value_logits, self.support_size).detach(),
            reward=support_to_scalar(reward_logits, self.support_size).detach(),
            policy_logits=policy_dict,
            policy_tensor=policy_logits,
            value_logits=value_logits,
            reward_logits=reward_logits,
            hidden_state=hidden_state,
            reward_hidden=reward_hidden,
        )

    def initial_inference(self, observation) -> NetworkOutput:

        hidden_state = self.representation(self._as_batch(observation))
        batch = hidden_state.shape[0]

        # The value prefix is zero at the root of a window: nothing has been
        # accumulated yet. Encode that as a confident distribution on the zero bin
        # so the tree reads back exactly 0.
        reward_logits = torch.zeros(batch, self.support_dim, device=hidden_state.device)
        reward_logits[:, self.support_size] = 1e4

        return self._predict(
            hidden_state,
            reward_logits,
            self.initial_reward_hidden(batch, hidden_state.device),
        )

    def recurrent_inference(self, hidden_state, action, reward_hidden=None) -> NetworkOutput:

        if not isinstance(hidden_state, torch.Tensor):
            hidden_state = torch.as_tensor(hidden_state, dtype=torch.float32)
        hidden_state = hidden_state.to(GPU_DEVICE, dtype=torch.float32)

        if hidden_state.dim() == 3:
            hidden_state = hidden_state.unsqueeze(0)

        next_hidden_state = self.dynamics(hidden_state, action)

        # `reward` on the output is the accumulated value prefix, not the reward
        # of this single step. The tree recovers the step reward by differencing
        # against the parent's prefix (see mcts._evaluate_leaf).
        reward_logits, next_reward_hidden = self.value_prefix(next_hidden_state, reward_hidden)

        return self._predict(next_hidden_state, reward_logits, next_reward_hidden)

    def forward(self, observation):

        hidden_state = self.representation(self._as_batch(observation))
        return hidden_state, self.value_net(hidden_state), self.policy_net(hidden_state)

    def training_steps(self) -> int:

        # How many steps / batches the network has been trained for.
        return self.tot_training_steps


class SharedStorage:

    def __init__(self, config, path=settings.WEIGHTS_FILE_PATH):
        self.networks = {}  # Dictionary to store networks by step
        self.latest_step = 0

        model_path = os.path.join(settings.MODEL_DIR, settings.RUN_ID, path)

        self.latest_network = Network(config)

        # What the weights on disk actually scored, so a resumed run has to beat
        # the checkpoint it inherited rather than replacing it with its first
        # noisy window. Only meaningful alongside the weights it describes -- a
        # marker left behind by a deleted checkpoint would otherwise set a bar
        # that a freshly initialised network can never clear.
        if os.path.exists(model_path):
            self.latest_network.load_model(path)
            self.best_level, self.best_score = self._read_best_marker()
        else:
            self.best_level, self.best_score = 0, float("-inf")

        self.writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, settings.RUN_ID, "tensorboard"))


    def get_latest_network(self) -> Network:

        return self.latest_network


    def save_network(self, step: int, network: Network):

        self.networks[step] = network
        if step > self.latest_step:
            self.latest_step = step
            self.latest_network = network


    def get_network(self, step: int) -> Network:

        return self.networks.get(step, self.latest_network)


    def log_scalars(self, step, reward=None, loss=None, elapsed_min=None,
                    solved_rate=None, solution_length=None, loss_parts=None,
                    curriculum_moves=None, optimal_depth=None, excess_moves=None):

        if reward is not None:
            self.writer.add_scalar("episode/reward", reward, step)
        # Episodes before the buffer fills record no loss; NaN poisons the chart.
        if loss is not None and not math.isnan(loss):
            self.writer.add_scalar("episode/loss", loss, step)
        if elapsed_min is not None:
            self.writer.add_scalar("episode/elapsed_minutes", float(elapsed_min), step)
        if solved_rate is not None:
            self.writer.add_scalar("episode/solved_rate", solved_rate, step)
        if solution_length is not None:
            self.writer.add_scalar("episode/solution_length", solution_length, step)
        if curriculum_moves is not None:
            self.writer.add_scalar("episode/curriculum_depth", curriculum_moves, step)
        # Plot against curriculum_depth: the two diverging means the scramble is
        # generating easier positions than the level it is labelled with.
        if optimal_depth is not None:
            self.writer.add_scalar("episode/optimal_depth", optimal_depth, step)
        # Moves spent above the optimum. Zero is optimal play; solved_rate alone
        # cannot distinguish that from a win that wandered.
        if excess_moves is not None:
            self.writer.add_scalar("episode/excess_moves", excess_moves, step)
        if loss_parts:
            for name, value in loss_parts.items():
                if not math.isnan(value):
                    self.writer.add_scalar(f"loss/{name}", value, step)


    def save_if_best(self, level, score, episode):

        """Overwrite the single weights file only when the model has improved.

        One file, as before -- but written on merit rather than on every episode.
        A run that peaks and then degrades used to end with the degraded weights,
        because the last episode always won.

        "Best" cannot be the solved rate alone. Solving 95% at curriculum depth 2
        is not better than solving 60% at depth 6, so a plain rate comparison
        would freeze the file at the first easy level and never write again.
        Progress is therefore ranked by (level, rate) lexicographically:

          * A deeper level always beats a shallower one, and resets the bar --
              the first score at a new depth is by definition its best so far.
              This also means the weights that earned a promotion are saved
              immediately, which is exactly when they are worth keeping.
          * Within a level, only an improved rate writes. Degradation cannot
              overwrite a better checkpoint.

        Early episodes save unconditionally so that a crash in the first few
        minutes still leaves a usable file.
        """

        if not settings.SAVE_BEST_ONLY:
            self.latest_network.save_model()
            return True

        warming_up = episode <= settings.CHECKPOINT_WARMUP_EPISODES

        # A new depth is a new scale; last level's rate is not a bar to clear.
        if level > self.best_level:
            self.best_level = level
            self.best_score = float("-inf")

        improved = score > self.best_score

        if not (warming_up or improved):
            return False

        if improved:
            self.best_score = score

        self.latest_network.save_model()
        self._write_best_marker(episode)

        if improved and not warming_up:
            logger.debug(
                f"[Checkpoint] Episode {episode}: saved at level {level}, "
                f"solved {score:.0%}."
            )

        return True


    def _write_best_marker(self, episode):

        """Record what the saved weights scored, next to the weights themselves.

        Without this a resumed run starts with `best_score` unset, and its first
        noisy window overwrites a checkpoint that took hours to earn.
        """

        run_dir = os.path.join(settings.MODEL_DIR, settings.RUN_ID)
        os.makedirs(run_dir, exist_ok=True)

        marker = {"level": self.best_level, "score": self.best_score, "episode": episode}

        with open(os.path.join(run_dir, "best.json"), "w") as handle:
            json.dump(marker, handle)


    def _read_best_marker(self):

        path = os.path.join(settings.MODEL_DIR, settings.RUN_ID, "best.json")

        try:
            with open(path) as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            return 0, float("-inf")

        return marker.get("level", 0), marker.get("score", float("-inf"))


    def update_elapsed_time(self, new_time):

        run_log_dir = os.path.join(settings.LOG_DIR, settings.RUN_ID)
        os.makedirs(run_log_dir, exist_ok=True)
        elapsed_time_file = os.path.join(run_log_dir, "elapsed_time.txt")

        try:
            # Read the current elapsed time from the file (if it exists)
            with open(elapsed_time_file, 'r') as file:
                current_time = file.read().strip()

            # Update the time by adding the new value
            updated_time = float(current_time) + float(new_time) if current_time else new_time
            logger.debug(f"Total Elapsed Time: {updated_time} minutes [+{new_time} min/ep]")

        except FileNotFoundError:
            # If the file doesn't exist, start with the new time
            updated_time = new_time
            logger.warning(f"File not found. Starting with new time: {updated_time} minutes")

        # Write the updated time back to the file
        with open(elapsed_time_file, 'w') as file:
            file.write(str(updated_time))

        self.writer.flush()


##### End Helpers ########
##########################
