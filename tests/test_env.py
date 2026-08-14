"""Environment-semantics tests.

Each test here pins a bug that made learning impossible before the Phase 1
correctness pass; they fail against the pre-fix environment.
"""

import numpy as np
import pytest

from src.game.config import INT2DIRECTION, NUMBER_OF_DIRECTIONS
from src.model.config import AI_ACTION_SPACE_SIZE
from src.model.state import action_to_index, index_to_action


def _place(env, positions):
    """Put robots on exact squares, keeping prev == current."""

    for robot, (x, y) in zip(env.robots, positions):
        robot.x, robot.y = x, y
        robot.prev_x, robot.prev_y = x, y
        robot.robotLeftTarget = False


def _landing_square(env, robot_idx, direction):
    """Where robot `robot_idx` would stop moving in `direction`, or None."""

    robot = env.robots[robot_idx]
    before = robot.get_position()
    moved = robot.move_until_blocked(
        simulated=True, direction=direction, board=env.game.board,
        other_robots=env.game.robot_manager.robots,
    )
    assert robot.get_position() == before, "simulated move must not mutate the robot"
    if not moved:
        return None

    # Re-run unsimulated on a scratch copy of the position to learn the square.
    robot.move_until_blocked(
        simulated=False, direction=direction, board=env.game.board,
        other_robots=env.game.robot_manager.robots,
    )
    landing = robot.get_position()
    robot.x, robot.y = before
    robot.prev_x, robot.prev_y = before
    return landing


def test_action_space_is_robot_times_direction(env):
    assert AI_ACTION_SPACE_SIZE == len(env.robots) * NUMBER_OF_DIRECTIONS == 16
    assert env.action_space.n == AI_ACTION_SPACE_SIZE


def test_action_encoding_roundtrips():
    for robot_idx in range(4):
        for direction in range(NUMBER_OF_DIRECTIONS):
            assert index_to_action(action_to_index(robot_idx, direction)) == (robot_idx, direction)


def test_move_action_moves_the_named_robot(env):
    """The env must honour the robot index the search planned for.

    It previously moved `self.selected_robot_idx` regardless, so 16 of the 20
    action ids aliased down to 4 distinct effects.
    """

    moved_any = False

    for robot_idx in range(len(env.robots)):
        for direction in range(NUMBER_OF_DIRECTIONS):
            env.reset()
            before = [r.get_position() for r in env.robots]

            if _landing_square(env, robot_idx, INT2DIRECTION[direction]) is None:
                continue

            env.step(action_to_index(robot_idx, direction))
            after = [r.get_position() for r in env.robots]

            assert after[robot_idx] != before[robot_idx], (
                f"robot {robot_idx} should have moved {INT2DIRECTION[direction]}"
            )
            for other in range(len(env.robots)):
                if other != robot_idx:
                    assert after[other] == before[other], "only the named robot may move"
            moved_any = True

    assert moved_any, "no legal move was exercised"


def test_reset_rerandomizes_robot_positions(env):
    """`_initialize_robot_positions` only assigned robots whose x was None, so
    reset() was a no-op on an env that had already been played."""

    layouts = set()
    for _ in range(10):
        env.reset()
        layouts.add(tuple(r.get_position() for r in env.robots))

    assert len(layouts) > 1


def test_reset_clears_left_target_flag(env):
    for robot in env.robots:
        robot.robotLeftTarget = True

    env.reset()

    assert not any(r.robotLeftTarget for r in env.robots)


def test_robots_never_start_on_the_target(env):
    for _ in range(25):
        env.reset()
        target = (env.game.target_deck.current_target.x, env.game.target_deck.current_target.y)
        assert target not in {r.get_position() for r in env.robots}


def test_win_reward_survives_a_repeated_state(env):
    """A solving move that lands on a previously seen configuration used to be
    recorded as -5, because the repeat penalty overwrote the win reward."""

    env.reset()

    # Find any robot/direction whose landing square we can make the target.
    for robot_idx in range(len(env.robots)):
        for direction in range(NUMBER_OF_DIRECTIONS):
            landing = _landing_square(env, robot_idx, INT2DIRECTION[direction])
            if landing is None or landing == env.robots[robot_idx].get_position():
                continue

            target = env.game.target_deck.current_target
            target.x, target.y = landing
            target.color = env.robots[robot_idx].color

            # Pre-seed the post-move configuration so the repeat branch fires.
            positions = [r.get_position() for r in env.robots]
            positions[robot_idx] = landing
            env.visited_states = {tuple(positions)}

            _, reward, done, _ = env.step(action_to_index(robot_idx, direction))

            assert done, "robot should have reached the target"
            assert reward > 0, f"win reward was overwritten by the repeat penalty: {reward}"
            return

    pytest.skip("no usable move found on this randomly generated layout")


def test_observation_is_float32_and_correctly_sized(env):
    obs = env.reset()
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
