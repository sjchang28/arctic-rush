"""The MuZero network: representation, dynamics, prediction.

This file used to be 674 lines holding three unrelated things -- a replay
buffer, this torch module, and a checkpoint/TensorBoard manager. They are now
`replay.py`, this, and `storage.py`.
"""

import os
from typing import Dict, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import settings
from src.core.logger import logger
from src.game.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUMBER_OF_DIRECTIONS,
    NUMBER_OF_ROBOTS,
)
from src.model.config import (
    WEIGHTS_FILE_PATH,
)
from src.model.device import gpu_device
from src.model.support import support_to_scalar

##########################
####### Helpers ##########


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
        logger.debug(f"Ready to begin training on device {str(gpu_device()).upper()} ...")
        self.to(gpu_device())

    ##### Persistence #####

    def save_model(self, path=WEIGHTS_FILE_PATH):

        run_dir = os.path.join(settings.MODEL_DIR, settings.RUN_ID)
        os.makedirs(run_dir, exist_ok=True)
        model_weights_file = os.path.join(run_dir, path)

        torch.save(self.state_dict(), model_weights_file)

    def load_model(self, path=WEIGHTS_FILE_PATH):

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

        device = device if device is not None else gpu_device()
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
        observation = observation.to(gpu_device(), dtype=torch.float32)

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
        hidden_state = hidden_state.to(gpu_device(), dtype=torch.float32)

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
