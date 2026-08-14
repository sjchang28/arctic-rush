"""Exact-solver and curriculum-difficulty tests.

These pin the failure the reverse curriculum had in its first form: scramble
depth is only an upper bound on true difficulty, so the generator produced
one-move positions labelled "depth 12", the promotion gate cleared every window
it saw, and the 96% solved rate described puzzles far easier than the level it
was recorded against.
"""

import random

import numpy as np
import pytest

from src.game.config import CURRICULUM_MIN_DEPTH_RATIO
from src.game.solver import SEARCH_EXHAUSTED, shortest_solution_length


def _optimal(env, max_depth=8, node_budget=30_000):
    return shortest_solution_length(
        env.game.board, env.robots, env.game.target_deck.current_target,
        max_depth=max_depth, node_budget=node_budget,
    )


def test_solver_reports_zero_on_a_solved_position(env):
    """A matching robot already on the goal is degenerate, not a puzzle."""

    target = env.game.target_deck.current_target
    matching = [i for i, r in enumerate(env.robots)
                if target.color.upper() == "ANY" or r.color.lower() == target.color.lower()]
    assert matching

    env.robots[matching[0]].x = target.x
    env.robots[matching[0]].y = target.y

    assert _optimal(env) == 0


def test_solver_does_not_mutate_the_live_robots(env):
    """The solver runs on stand-ins; the episode it was asked about is untouched."""

    before = [(r.x, r.y) for r in env.robots]
    _optimal(env)
    assert [(r.x, r.y) for r in env.robots] == before


def test_solver_agrees_with_the_environment(env, monkeypatch):
    """The optimum the solver reports must be achievable by the real `step`.

    Not just a number: replay a shortest solution through the environment and
    require it to terminate in exactly that many moves.
    """

    monkeypatch.setattr("src.game.env.CURRICULUM_START_MOVES", 2)
    env.set_curriculum_moves(2)

    # Replaying through the environment is exponential, so this needs a shallow
    # position. Resample until one turns up rather than skipping -- a test that
    # skips on a bad draw silently stops testing anything.
    for _ in range(20):
        env.reset()
        optimal = _optimal(env, max_depth=3)
        if optimal not in (None, SEARCH_EXHAUSTED, 0):
            break
    else:
        pytest.fail("curriculum never produced a position solvable in 3 moves or fewer")

    # Breadth-first replay through the environment itself, which is a different
    # code path from the solver's stand-in robots.
    start = env.snapshot()
    frontier = [([], start)]

    for _ in range(optimal):
        nxt = []
        for history, state in frontier:
            for action in range(env.action_space.n):
                env.restore(state)
                _, _, done, _ = env.step(action)
                if done:
                    assert len(history) + 1 == optimal, (
                        f"environment solved in {len(history) + 1}, solver said {optimal}"
                    )
                    return
                nxt.append((history + [action], env.snapshot()))
        frontier = nxt

    pytest.fail(f"solver claimed {optimal} moves but the environment never solved in that many")


def test_forward_walk_responds_to_its_length(env, monkeypatch):
    """The property the backward scramble lacks, and the reason it was added.

    A longer scramble produces the same distribution as a short one; a longer
    forward walk produces genuinely harder positions. The curriculum's reach
    depends on that, so it is asserted rather than assumed.
    """

    monkeypatch.setattr("src.game.curriculum.CURRICULUM_WALK_MIN", 0)
    monkeypatch.setattr("src.game.curriculum.CURRICULUM_WALK_MAX", 512)

    target = env.game.target_deck.current_target
    matching = [i for i, r in enumerate(env.robots)
                if target.color.upper() == "ANY" or r.color.lower() == target.color.lower()]
    solver_index = matching[0]

    def mean_depth(steps, trials=20):
        monkeypatch.setattr("src.game.curriculum.CURRICULUM_WALK_PER_DEPTH", steps)
        env.set_curriculum_moves(1)  # walk length becomes `steps * 1`
        depths = []
        for _ in range(trials):
            env.curriculum.forward_walk_once(target, solver_index)
            depth = _optimal(env, max_depth=8)
            depths.append(8 if depth in (None, SEARCH_EXHAUSTED) else depth)
        return float(np.mean(depths))

    assert mean_depth(48) > mean_depth(2)


@pytest.mark.parametrize("requested", [1, 2, 4, 6])
def test_curriculum_generates_positions_near_the_requested_depth(env, monkeypatch, requested):
    """The bug this whole change exists for.

    Before verification, every level -- including depth 12 -- was served
    positions that were overwhelmingly one move from solved, so the solved rate
    described a puzzle far easier than its label and the curriculum promoted
    straight to the ceiling in ten consecutive windows.
    """

    monkeypatch.setattr("src.game.curriculum.CURRICULUM_VERIFY_DEPTH", True)
    monkeypatch.setattr("src.game.curriculum.CURRICULUM_SCRAMBLE_ATTEMPTS", 4)
    monkeypatch.setattr("src.game.env.SOLVER_NODE_BUDGET", 15_000)

    env.set_curriculum_moves(requested)

    depths = []
    for _ in range(12):
        env.reset()
        # An episode must never start already solved.
        if env.last_optimal_depth is not None:
            assert env.last_optimal_depth >= 1
            depths.append(env.last_optimal_depth)

    assert depths, f"level {requested} verified nothing at all"

    mean_depth = float(np.mean(depths))
    assert mean_depth >= CURRICULUM_MIN_DEPTH_RATIO * requested, (
        f"level {requested} produced positions averaging {mean_depth:.1f} "
        f"optimal moves: {depths}"
    )


def test_reverse_scramble_is_a_shallow_generator(env):
    """Pins the measurement the curriculum design rests on.

    The scramble is kept only for depth 1-2. If this ever stops holding -- a
    board with enough walls to trap a backward walk, say -- the curriculum could
    lean on it again, so the claim is asserted rather than left in a comment.
    """

    target = env.game.target_deck.current_target
    matching = [i for i, r in enumerate(env.robots)
                if target.color.upper() == "ANY" or r.color.lower() == target.color.lower()]
    solver_index = matching[0]

    rng = random.Random(0)
    depths = []

    for _ in range(16):
        env.place_robots_randomly(forbidden_extra={(target.x, target.y)})
        env.robots[solver_index].x, env.robots[solver_index].y = target.x, target.y
        for robot in env.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False

        # Deliberately over-scrambled: depth is not what this dial controls.
        env.game.robot_manager.reverse_scramble(
            16, rng=rng, required_first=solver_index,
            solver_index=solver_index, solver_bias=0.6,
            avoid_square=(target.x, target.y),
        )

        depth = _optimal(env, max_depth=6)
        # Anything the bounded search could not pin down is counted generously
        # against the claim being made, so this cannot pass by accident.
        depths.append(6 if depth in (None, SEARCH_EXHAUSTED) else depth)

    assert float(np.mean(depths)) < 3.0, (
        f"16-move scramble now averages {np.mean(depths):.1f} optimal moves; "
        f"the curriculum's generator choice should be revisited"
    )


def test_curriculum_holds_when_positions_are_too_shallow(monkeypatch, fake_config):
    """Promotion must refuse a level whose generated positions are trivial.

    Previously the gate saw only the solved rate, so it promoted on every window
    -- ten levels in ten windows, never once stalling, on positions that were
    mostly one move from solved.
    """

    import collections

    from src.model.promotion import maybe_promote_curriculum

    monkeypatch.setattr("src.model.promotion.CURRICULUM_PROMOTE_WINDOW", 10)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_PROMOTE_THRESHOLD", 0.75)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_MIN_DEPTH_RATIO", 0.75)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_MAX_MOVES", 12)

    config = fake_config
    solved = collections.deque([1.0] * 10, maxlen=100)

    shallow = collections.deque([1] * 10, maxlen=100)
    assert not maybe_promote_curriculum(config, solved, shallow, episode=10)
    assert config.curriculum_moves == 8

    honest = collections.deque([8] * 10, maxlen=100)
    assert maybe_promote_curriculum(config, solved, honest, episode=20)
    assert config.curriculum_moves == 9


def test_promotion_holds_when_depth_could_not_be_verified(monkeypatch, fake_config):
    """Unmeasured is not the same as hard.

    Past a certain depth the solver gives up rather than concluding. Promoting on
    those episodes would mean promoting on difficulty nobody established -- the
    original bug wearing a different hat -- so the gate stalls instead.
    """

    import collections

    from src.model.promotion import maybe_promote_curriculum

    monkeypatch.setattr("src.model.promotion.CURRICULUM_VERIFY_DEPTH", True)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_PROMOTE_WINDOW", 10)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_PROMOTE_THRESHOLD", 0.75)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_MAX_MOVES", 12)

    config = fake_config
    solved = collections.deque([1.0] * 10, maxlen=100)

    # Only two of ten episodes carried a verified depth.
    unverified = collections.deque([None] * 8 + [8, 8], maxlen=100)
    assert not maybe_promote_curriculum(config, solved, unverified, episode=10)
    assert config.curriculum_moves == 8


def test_verified_positions_are_pooled_and_reused(env, monkeypatch):
    """Verification is amortised, and reuse must not degrade the labels.

    Solving a candidate exactly costs seconds at depth 6, which is longer than an
    episode; without reuse the curriculum would cost more than the training it
    exists to shape.
    """

    monkeypatch.setattr("src.game.curriculum.CURRICULUM_VERIFY_DEPTH", True)
    monkeypatch.setattr("src.game.curriculum.CURRICULUM_SCRAMBLE_ATTEMPTS", 4)
    monkeypatch.setattr("src.game.env.SOLVER_NODE_BUDGET", 15_000)
    monkeypatch.setattr("src.game.curriculum.CURRICULUM_POOL_MIN", 2)
    monkeypatch.setattr("src.game.curriculum.CURRICULUM_POOL_REFRESH", 0.15)

    # Only exact matches are pooled, and the generators hit an exact depth well
    # under half the time, so the count is not the assertion -- the labels are.
    requested = 4
    env.set_curriculum_moves(requested)

    for _ in range(25):
        env.reset()

    pool = env.curriculum.depth_pool.get(requested)
    assert pool and len(pool) >= 2

    # Every pooled entry must genuinely be the depth it is filed under, target
    # included -- difficulty belongs to the pair, not to the robots alone.
    for entry in list(pool)[:6]:
        env.curriculum.apply_pooled(entry)
        assert _optimal(env, max_depth=requested + 1) == requested
