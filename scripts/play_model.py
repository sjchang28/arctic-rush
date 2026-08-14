"""Manual visual playback of a trained checkpoint.

Run it directly: `python -m scripts.play_model`.

This lived at `tests/test.py`, inside pytest's `testpaths`, despite being a
pygame demo rather than a test. It was only excluded from collection by the
accident of its filename not matching `test_*.py` -- rename it to anything more
descriptive and pytest would have started importing pygame and the whole
training stack during collection.
"""

import time

import pygame
import torch

from src.core.logger import logger
from src.model.mcts import Node, RealEnvironmentModel, expand_root, run_gumbel_mcts, run_mcts
from src.model.muzero import make_ricochet_config
from src.model.network import Network
from src.model.train import select_action


@torch.no_grad()
def run_model_test(weights_path=None, render_fps=5, max_moves=None):
    """Play one episode with the trained network and render it with pygame.

    Actions come from the same search self-play uses, not from a bare argmax over
    the policy head: the policy is a prior for the search, and reading it alone
    throws away everything planning contributes.
    """

    config = make_ricochet_config(render_ai=True)
    if max_moves is not None:
        config.max_moves = max_moves

    network = Network(config)
    if weights_path is not None:
        network.load_state_dict(torch.load(weights_path, map_location="cpu"))
    network.eval()

    game = config.new_game()
    model = RealEnvironmentModel(game.environment) if config.search_mode == "alphazero" else None

    clock = pygame.time.Clock()

    while not game.is_terminal() and len(game.history) < config.max_moves:

        root = Node(0)
        legal_actions = expand_root(config, root, network, game, model)
        if not legal_actions:
            logger.warning("No legal moves available; stopping.")
            break

        if config.use_gumbel:
            action, _ = run_gumbel_mcts(config, root, game.action_history(), network, model)
        else:
            run_mcts(config, root, game.action_history(), network, model)
            action = select_action(config, len(game.history), root, network)

        game.apply(action)
        game.environment.render()
        clock.tick(render_fps)

    logger.info(
        f"Episode done in {len(game.history)} moves. "
        f"Solved: {game.is_terminal()}. Total reward: {game.total_rewards():.3f}"
    )

    time.sleep(1)
    game.release_environment()


if __name__ == "__main__":
    run_model_test()
