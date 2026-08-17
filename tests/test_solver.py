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
        depths = []
        for _ in range(trials):
            # Depth 1, so the walk length is `steps * 1`.
            env.curriculum.forward_walk_once(target, solver_index, 1)
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


# ---------------------------------------------------------------------------
# Promotion cadence, demotion, and rehearsal of mastered depths.
#
# These pin the second failure the curriculum had, found in the runs to
# 2026-08-17: the gate promoted on variance rather than competence, the ramp had
# no way back down once it did, and the level it left stopped being generated at
# all and was trained away.
# ---------------------------------------------------------------------------


def _gate(monkeypatch, window=10, promote=0.85, demote=0.35, start=1, maximum=12):
    monkeypatch.setattr("src.model.promotion.CURRICULUM_PROMOTE_WINDOW", window)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_PROMOTE_THRESHOLD", promote)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_DEMOTE_THRESHOLD", demote)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_START_MOVES", start)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_MAX_MOVES", maximum)
    monkeypatch.setattr("src.model.promotion.CURRICULUM_MIN_DEPTH_RATIO", 0.75)


def test_promotion_is_decided_once_per_window(monkeypatch, fake_config):
    """The bug: read after every episode, a trailing window is not a measurement.

    It asks whether *any* window has ever crossed the bar, re-rolled every
    episode, which a marginal agent clears on variance alone. Every promotion
    past depth 3 in the 2026-08-17 runs fired at exactly the minimum passing
    count -- never with margin.
    """

    import collections

    from src.model.promotion import maybe_promote_curriculum

    _gate(monkeypatch, window=10)

    config = fake_config
    solved = collections.deque([1.0] * 10, maxlen=100)
    depths = collections.deque([8] * 10, maxlen=100)

    # A passing window, but not on the cadence: no decision is taken.
    for episode in (11, 15, 19):
        assert not maybe_promote_curriculum(config, solved, depths, episode)
        assert config.curriculum_moves == 8

    assert maybe_promote_curriculum(config, solved, depths, episode=20)
    assert config.curriculum_moves == 9


def test_a_failing_level_demotes(monkeypatch, fake_config):
    """Promotion used to be one-way, so a level entered too early was permanent.

    The 2026-08-17 runs promoted to depth 5 on a marginal window, collapsed to
    20-40%, and stayed there for their remaining ~1400 episodes.
    """

    import collections

    from src.model.promotion import maybe_promote_curriculum

    _gate(monkeypatch, window=10)

    config = fake_config
    solved = collections.deque([1.0] * 2 + [0.0] * 8, maxlen=100)
    depths = collections.deque([8] * 10, maxlen=100)

    assert maybe_promote_curriculum(config, solved, depths, episode=20)
    assert config.curriculum_moves == 7

    # The window described the level just left, so it must not carry over.
    assert not solved and not depths


def test_demotion_stops_at_the_starting_depth(monkeypatch, fake_config):
    import collections

    from src.model.promotion import maybe_promote_curriculum

    _gate(monkeypatch, window=10, start=1)

    config = fake_config
    config.set_curriculum_moves(1)

    solved = collections.deque([0.0] * 10, maxlen=100)
    depths = collections.deque([1] * 10, maxlen=100)

    assert not maybe_promote_curriculum(config, solved, depths, episode=20)
    assert config.curriculum_moves == 1


def test_the_two_thresholds_leave_a_band_that_holds(monkeypatch, fake_config):
    """Hysteresis. With the thresholds close together a level sitting between
    them would promote, fail, demote and promote again indefinitely."""

    import collections

    from src.model.promotion import maybe_promote_curriculum

    _gate(monkeypatch, window=10, promote=0.85, demote=0.35)

    config = fake_config
    solved = collections.deque([1.0] * 6 + [0.0] * 4, maxlen=100)  # 60%
    depths = collections.deque([8] * 10, maxlen=100)

    assert not maybe_promote_curriculum(config, solved, depths, episode=20)
    assert config.curriculum_moves == 8
    # Held, not reset: the window is still describing the level being played.
    assert len(solved) == 10


def test_a_depth_is_rehearsed_once_it_is_mastered(env, monkeypatch):
    """Positions are generated only at the current level, and the replay buffer
    holds 100 games, so within 100 episodes of a promotion no example of the
    previous level survives anywhere and the network trains it away."""

    monkeypatch.setattr("src.game.env.CURRICULUM_MASTERY_WINDOW", 4)
    monkeypatch.setattr("src.game.env.CURRICULUM_MASTERY_THRESHOLD", 0.95)

    env.set_curriculum_moves(6)

    # Nothing established yet, so there is nothing worth rehearsing.
    assert env.mastered_depths() == []

    for _ in range(4):
        env.record_result(2, solved=True)
    assert env.mastered_depths() == [2]

    # A depth it is merely decent at is not mastered.
    for solved in (True, True, True, False):
        env.record_result(3, solved=solved)
    assert env.mastered_depths() == [2]

    # ...and the current level is not something to rehearse against itself.
    for _ in range(4):
        env.record_result(6, solved=True)
    assert env.mastered_depths() == [2]


def test_mastery_is_sticky_so_a_slipping_depth_can_recover(env, monkeypatch):
    """A depth dropped for slipping would never be practised again, so it could
    never recover -- which is the forgetting this exists to prevent."""

    monkeypatch.setattr("src.game.env.CURRICULUM_MASTERY_WINDOW", 4)
    monkeypatch.setattr("src.game.env.CURRICULUM_MASTERY_THRESHOLD", 0.95)

    env.set_curriculum_moves(6)

    for _ in range(4):
        env.record_result(2, solved=True)
    assert env.mastered_depths() == [2]

    for _ in range(4):
        env.record_result(2, solved=False)
    assert env.mastered_depths() == [2]


def test_rehearsal_depths_are_drawn_from_the_mastered_set(env, monkeypatch):
    monkeypatch.setattr("src.game.env.CURRICULUM_MASTERY_WINDOW", 4)
    monkeypatch.setattr("src.game.env.CURRICULUM_REHEARSAL_RATE", 1.0)

    env.set_curriculum_moves(6)

    # Nothing mastered: rehearsal has nothing to draw from and falls through to
    # the current level rather than inventing one.
    assert env._next_start_depth() == 6

    for depth in (1, 2):
        for _ in range(4):
            env.record_result(depth, solved=True)

    assert {env._next_start_depth() for _ in range(40)} <= {1, 2}

    monkeypatch.setattr("src.game.env.CURRICULUM_REHEARSAL_RATE", 0.0)
    assert env._next_start_depth() == 6
