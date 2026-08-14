"""Shared fixtures.

Two things here have to happen before anything under `src` is imported, so they
run at module scope rather than in a fixture:

  * `LOG_DIR` -- `src/core/logger.py` opens its file sink at import time, so by
    the time any fixture runs the log file already exists. Redirecting the
    setting later moves nothing. Pointing the environment variable at a temp
    directory first is the only thing that actually stops a test run writing
    into the repository's `data/logs/`.

  * `SDL_VIDEODRIVER` -- lets the pygame-backed tests construct a display
    without a window server.
"""

import os
import tempfile

# Must precede `import src.*` anywhere in the test session.
os.environ.setdefault("LOG_DIR", os.path.join(tempfile.gettempdir(), "arctic_rush_test_logs"))
os.environ.setdefault("RUN_ID", "pytest")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from src.game.env import RicochetRobotsEnv  # noqa: E402

SEED = 20260812


# The suite draws on three independent random streams and every one of them has
# to be pinned, or the tests that assert on a threshold flake:
#
#   * `random` -- the curriculum generators. The depth-2 level is a thin enough
#     slice that an unseeded
#     `test_curriculum_generates_positions_near_the_requested_depth[2]` fails
#     outright roughly one run in four.
#   * `numpy` -- action sampling in `softmax_sample` and Dirichlet noise.
#   * `torch` -- network weight initialisation. Missing this one is why
#     `test_overfits_a_single_fixed_position` and
#     `test_alphazero_search_mode_finds_a_one_move_solution` could each fail on
#     one run and pass on the next: both assert on what an initial network can
#     learn or solve, which depends entirely on how it was initialised.
#
# Tests that need variety still get it -- successive draws within a test differ,
# only the run does not differ from the last one.
def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# Both scopes are needed, and the session one is not redundant.
#
# pytest sets higher-scoped fixtures up first, so a module-scoped fixture such as
# `test_learning.trained_setup` -- which builds a network and plays four games --
# is constructed *before* any function-scoped autouse fixture runs. Seeding only
# per-function therefore left exactly the expensive, network-initialising
# fixtures unseeded, which is why the learning tests still flaked after the first
# attempt at pinning the RNG.
@pytest.fixture(scope="session", autouse=True)
def _deterministic_rng_session():
    _seed_everything()


@pytest.fixture(autouse=True)
def _deterministic_rng():
    _seed_everything()


@pytest.fixture
def env():
    e = RicochetRobotsEnv(render_ai=False)
    yield e
    e.close()


@pytest.fixture
def fake_config():
    """Stand-in for the training config that `maybe_promote_curriculum` mutates.

    The promotion gate only ever reads and writes `curriculum_moves`; building a
    real `RicochetRobotsConfig` would drag a network into a test about a
    threshold.
    """

    class FakeConfig:
        curriculum_moves = 8

        def set_curriculum_moves(self, moves):
            self.curriculum_moves = moves

    return FakeConfig()
