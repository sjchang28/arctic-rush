"""Search and network-head tests.

Each test here pins a bug from the Phase 1 correctness pass.
"""

import math

import numpy as np
import pytest
import torch

from src.config import AI_OBSERVATION_SHAPE
from src.core.mcts import Node, expand_node, run_gumbel_mcts, run_mcts
from src.core.muzero import make_ricochet_config
from src.core.network import Network
from src.core.support import support_to_scalar
from src.core.train import select_action, softmax_sample


@pytest.fixture(scope="module")
def config():
    return make_ricochet_config()


@pytest.fixture(scope="module")
def network(config):
    return Network(config)


def _fake_output(logits, action_space_size):
    """A NetworkOutput-shaped stub carrying known logits."""

    from core.network import NetworkOutput

    return NetworkOutput(
        value=torch.zeros(1),
        reward=torch.zeros(1),
        policy_logits={a: logits[a] for a in range(action_space_size)},
        policy_tensor=torch.zeros(1, action_space_size),
        value_logits=torch.zeros(1, 3),
        reward_logits=torch.zeros(1, 3),
        hidden_state=torch.zeros(1, 4, 2, 2),
    )


def test_softmax_sample_returns_an_action_id_not_a_list_index():
    """The root is expanded over legal actions only, so the position of a child
    in the visit-count list is not its action id. `softmax_sample` used to return
    the position, which self-play then applied to the environment as an action."""

    # Non-contiguous action ids, with all the visits on the last one.
    distribution = [(0, 7), (0, 9), (50, 13)]

    sampled = {softmax_sample(distribution, temperature=0.5) for _ in range(50)}

    assert sampled <= {7, 9, 13}
    assert 13 in sampled


def test_softmax_sample_respects_temperature():
    """Exponentiating raw visit counts collapsed the distribution to one-hot and
    made the temperature schedule inert."""

    distribution = [(30, 0), (20, 1)]

    hot = [softmax_sample(distribution, temperature=2.0) for _ in range(400)]
    cold = [softmax_sample(distribution, temperature=0.1) for _ in range(400)]

    # Higher temperature must pick the less-visited action more often.
    assert hot.count(1) > cold.count(1)


def test_expand_node_priors_are_softmax_over_legal_actions(config):
    logits = [float(a) for a in range(config.action_space_size)]
    legal = [0, 2, 3, 7]

    node = Node(0)
    expand_node(node, legal, _fake_output(logits, config.action_space_size))

    assert set(node.children) == set(legal)

    priors = np.array([node.children[a].prior for a in legal])
    assert priors.sum() == pytest.approx(1.0)

    expected = np.exp([logits[a] for a in legal])
    expected /= expected.sum()
    np.testing.assert_allclose(priors, expected, rtol=1e-6)


def test_expand_node_priors_are_not_near_uniform_for_confident_logits(config):
    """exp() applied to already-softmaxed probabilities lands in [1, e], so every
    prior came out near-uniform and the policy head never steered the search."""

    logits = [0.0] * config.action_space_size
    logits[3] = 8.0
    legal = list(range(config.action_space_size))

    node = Node(0)
    expand_node(node, legal, _fake_output(logits, config.action_space_size))

    assert node.children[3].prior > 0.9


def test_expand_node_survives_large_logits(config):
    """Softmax must subtract the max before exponentiating."""

    logits = [900.0] * config.action_space_size
    node = Node(0)
    expand_node(node, [0, 1], _fake_output(logits, config.action_space_size))

    assert all(math.isfinite(child.prior) for child in node.children.values())


def test_policy_head_emits_logits_not_probabilities(config, network):
    obs = torch.zeros(AI_OBSERVATION_SHAPE)
    out = network.initial_inference(obs)

    values = np.array(list(out.policy_logits.values()))

    # A softmaxed head sums to 1 and is non-negative; logits generally are neither.
    assert not (values.sum() == pytest.approx(1.0) and (values >= 0).all())


def test_inference_paths_agree_on_policy_representation(config, network):
    """initial_inference softmaxed its output while recurrent_inference did not,
    so the two paths fed different quantities into the same tree."""

    obs = torch.zeros(AI_OBSERVATION_SHAPE)
    initial = network.initial_inference(obs)
    recurrent = network.recurrent_inference(initial.hidden_state, 0)

    for out in (initial, recurrent):
        values = np.array(list(out.policy_logits.values()))
        assert not (values.sum() == pytest.approx(1.0) and (values >= 0).all())


def test_value_and_reward_heads_can_represent_negative_numbers(config, network):
    """Both heads ended in nn.ReLU, clamping them to >= 0 while the environment
    emits a negative reward for repeated states."""

    device = next(network.parameters()).device
    hidden = torch.rand(200, config.num_channels, 16, 16, device=device)

    with torch.no_grad():
        values = support_to_scalar(network.value_net(hidden), network.support_size)

        prefix_logits, _ = network.value_prefix(
            hidden, network.initial_reward_hidden(hidden.shape[0], device))
        rewards = support_to_scalar(prefix_logits, network.support_size)

    assert values.min().item() < 0, "value head cannot output a negative value"
    assert rewards.min().item() < 0, "value-prefix head cannot output a negative reward"


def test_value_prefix_accumulates_along_a_path(config, network):
    """The reward head predicts a cumulative prefix, and the tree recovers each
    step's own reward by differencing against its parent."""

    obs = torch.zeros(AI_OBSERVATION_SHAPE)

    initial = network.initial_inference(obs)
    assert initial.reward.item() == pytest.approx(0.0, abs=1e-3), "a window starts at zero"

    first = network.recurrent_inference(initial.hidden_state, 0, initial.reward_hidden)
    second = network.recurrent_inference(first.hidden_state, 1, first.reward_hidden)

    # Carrying the LSTM state must change the prediction; a reset must not depend
    # on it. If these matched, the LSTM state would be doing nothing.
    reset = network.recurrent_inference(first.hidden_state, 1, None)
    assert second.reward.item() != pytest.approx(reset.reward.item(), abs=1e-6)


def test_select_action_returns_an_action_present_in_the_tree(config, network):
    game = config.new_game()
    legal = game.legal_actions()

    root = Node(0)
    expand_node(root, legal, network.initial_inference(game.make_image(-1)))
    run_mcts(config, root, game.action_history(), network)

    for _ in range(20):
        action = select_action(config, len(game.history), root, network)
        assert action in root.children
        assert action in legal
