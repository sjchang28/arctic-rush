"""The environment must work on every level, not just level_01.

`LEVEL_FILE` was baked into module-level default arguments, so the level was
fixed at import and no test could vary it. Now that `default_level_file()`
resolves per call, these run the real environment against all four levels --
including level_04, whose bounce pads change how a robot slides and had never
been exercised by the env, the solver or the curriculum.
"""

import pytest

from src.game.env import RicochetRobotsEnv
from src.game.solver import SEARCH_EXHAUSTED

ALL_LEVELS = ["level_01.json", "level_02.json", "level_03.json", "level_04.json"]


@pytest.fixture
def env_on_level(monkeypatch):
    """An environment built against a named level.

    `LEVEL_FILE` is patched on `src.game.board`, which is where
    `default_level_file()` reads it from, before the env (and so the Board and
    TargetDeck) is constructed.
    """

    created = []

    def _build(level):
        monkeypatch.setattr("src.game.board.LEVEL_FILE", level)
        env = RicochetRobotsEnv(render_ai=False)
        created.append(env)
        return env

    yield _build

    for env in created:
        env.close()


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_environment_resets_and_steps_on_every_level(level, env_on_level):
    env = env_on_level(level)

    observation = env.reset()
    assert observation.shape == env.observation_space.shape

    for action in range(env.action_space.n):
        env.reset()
        observation, reward, done, _ = env.step(action)
        assert observation.shape == env.observation_space.shape
        assert isinstance(done, bool)


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_episodes_never_start_already_solved(level, env_on_level):
    """The curriculum must not hand out a free win on any level."""

    env = env_on_level(level)

    for _ in range(10):
        env.reset()
        target = (env.current_target.x, env.current_target.y)
        assert target not in {r.get_position() for r in env.robots}


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_curriculum_produces_a_verified_depth_on_every_level(level, env_on_level):
    """Level_04's bounce pads make some backward moves non-invertible, so the
    generators have to cope with a board the reverse scramble cannot fully walk."""

    env = env_on_level(level)
    env.set_curriculum_moves(1)

    depths = []
    for _ in range(8):
        env.reset()
        depths.append(env.last_optimal_depth)

    verified = [d for d in depths if d is not None]

    assert verified, f"{level} verified no depth at all"
    assert all(d >= 1 for d in verified), f"{level} produced a solved start position"


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_solver_agrees_with_the_board_on_every_level(level, env_on_level):
    env = env_on_level(level)
    env.set_curriculum_moves(1)
    env.reset()

    depth = env.measure_optimal_depth(max_depth=3)

    assert depth != 0, "a start position must not already be solved"
    assert depth is None or depth == SEARCH_EXHAUSTED or depth >= 1


def test_bounce_pads_are_actually_loaded_on_level_04(env_on_level):
    """Guards the whole point of parametrising: if level_04 quietly loaded
    level_01's board, every test above would pass without testing pads."""

    env = env_on_level("level_04.json")

    assert env.game.board.bounce_pad_manager.bounce_pads, "level_04 loaded no pads"


def test_level_01_has_no_bounce_pads(env_on_level):
    env = env_on_level("level_01.json")

    assert not env.game.board.bounce_pad_manager.bounce_pads
