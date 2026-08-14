"""Board, wall, bounce-pad and target-deck tests.

`src/game/board.py`, `robots.py` and `target.py` had no direct tests at all --
they were only ever exercised transitively through `RicochetRobotsEnv`. Bounce
pads, the most recent feature, were untested entirely, and only `level_01` was
ever loaded by anything.
"""

import json
import os

import pytest

from src.game.board import Board, BouncePadManager, default_level_file
from src.game.config import (
    ALL_DIRECTIONS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    center_squares,
)
from src.game.robots import Robot, RobotManager
from src.game.target import TargetDeck

LEVELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "src", "game", "levels")
ALL_LEVELS = ["level_01.json", "level_02.json", "level_03.json", "level_04.json"]

OPPOSITE = {UP: (DOWN, 0, -1), DOWN: (UP, 0, 1), LEFT: (RIGHT, -1, 0), RIGHT: (LEFT, 1, 0)}


def level_path(name):
    return os.path.join(LEVELS_DIR, name)


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_level_wall_data_is_symmetric(level):
    """The invariant that makes walls two-sided.

    `Board.add_wall` sets the wall on one side of one cell only; a mirroring
    block existed but was commented out. That is safe *because* every level file
    already lists both halves of every wall. Nothing enforced that, so a
    hand-edited level could introduce a wall a robot passes through from one
    side and not the other -- a bug that would surface only as strange play.
    """

    with open(level_path(level), "r") as handle:
        walls = {(w["row"], w["col"], w["dir"]) for w in json.load(handle)["walls"]}

    unpaired = []
    for row, col, direction in walls:
        opposite, dcol, drow = OPPOSITE[direction]
        neighbour_row, neighbour_col = row + drow, col + dcol
        inside = 0 <= neighbour_row < BOARD_HEIGHT and 0 <= neighbour_col < BOARD_WIDTH
        if inside and (neighbour_row, neighbour_col, opposite) not in walls:
            unpaired.append((row, col, direction))

    assert not unpaired, f"{level} has walls with no mirrored twin: {unpaired[:5]}"


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_board_loads_every_level(level):
    """Only level_01 was ever loaded; the others were never proven to parse."""

    board = Board(wall_file=level_path(level))

    assert len(board.walls) == BOARD_HEIGHT
    assert all(len(row) == BOARD_WIDTH for row in board.walls)
    assert all(set(cell) == set(ALL_DIRECTIONS) for row in board.walls for cell in row)


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_border_walls_enclose_the_board(level):
    """No robot may leave the board on any level."""

    board = Board(wall_file=level_path(level))

    for x in range(BOARD_WIDTH):
        assert not board.can_move(x, 0, UP)
        assert not board.can_move(x, BOARD_HEIGHT - 1, DOWN)
    for y in range(BOARD_HEIGHT):
        assert not board.can_move(0, y, LEFT)
        assert not board.can_move(BOARD_WIDTH - 1, y, RIGHT)


def test_only_level_04_has_bounce_pads():
    """Pins which level exercises the pad code, so a test asking for pads asks
    for the level that has them."""

    with_pads = {
        level for level in ALL_LEVELS
        if BouncePadManager(level_path(level)).bounce_pads
    }

    assert with_pads == {"level_04.json"}


def test_bounce_pad_redirects_only_the_matching_colour():
    """A pad rewrites the direction of a robot of its own colour and no other."""

    manager = BouncePadManager(level_path("level_04.json"))
    (x, y), pad = next(iter(manager.bounce_pads.items()))

    incoming = next(d for d, out in pad["redirect"].items() if out)

    assert manager.handle_bounce_pad(x, y, incoming, pad["color"]) == pad["redirect"][incoming]

    other = next(c for c in ("red", "blue", "green", "yellow") if c != pad["color"])
    assert manager.handle_bounce_pad(x, y, incoming, other) is None


def test_bounce_pad_is_absent_on_empty_squares():
    manager = BouncePadManager(level_path("level_04.json"))
    empty = next((x, y) for x in range(BOARD_WIDTH) for y in range(BOARD_HEIGHT)
                 if (x, y) not in manager.bounce_pads)

    assert manager.handle_bounce_pad(*empty, UP, "red") is None


def test_missing_level_file_does_not_raise():
    """A bad path is logged, not raised -- pinned because the error path is
    silent and would otherwise be easy to turn into a crash."""

    assert Board(wall_file=level_path("does_not_exist.json")).walls
    assert BouncePadManager(level_path("does_not_exist.json")).bounce_pads == {}


def test_default_level_file_is_resolved_per_call():
    """It was a default argument, evaluated once at import, which froze
    `LEVEL_FILE` for the life of the process."""

    import src.game.board as board_module

    original = board_module.LEVEL_FILE
    try:
        board_module.LEVEL_FILE = "level_04.json"
        assert default_level_file().endswith("level_04.json")
    finally:
        board_module.LEVEL_FILE = original

    assert default_level_file().endswith(original)


def test_robots_slide_until_blocked_and_stop_inside_the_board():
    board = Board()
    robots = [Robot("red", 5, 5)]
    for robot in robots:
        robot.prev_x, robot.prev_y = robot.x, robot.y

    robots[0].move_until_blocked(simulated=False, direction=UP, board=board, other_robots=robots)

    assert 0 <= robots[0].x < BOARD_WIDTH
    assert 0 <= robots[0].y < BOARD_HEIGHT
    assert not board.can_move(robots[0].x, robots[0].y, UP), "must stop against a blocker"


def test_simulated_move_does_not_mutate_the_robot():
    board = Board()
    robots = [Robot("red", 5, 5)]
    before = robots[0].get_position()

    robots[0].move_until_blocked(simulated=True, direction=UP, board=board, other_robots=robots)

    assert robots[0].get_position() == before


def test_a_robot_blocks_another():
    board = Board()
    blocker = Robot("blue", 5, 2)
    mover = Robot("red", 5, 5)
    robots = [mover, blocker]
    for robot in robots:
        robot.prev_x, robot.prev_y = robot.x, robot.y

    mover.move_until_blocked(simulated=False, direction=UP, board=board, other_robots=robots)

    assert mover.get_position() == (5, 3), "should stop directly below the blocker"


def test_robots_are_never_placed_on_the_centre_squares():
    board = Board()
    blocked = center_squares()

    for _ in range(20):
        robots = [Robot(c) for c in ("red", "blue", "green", "yellow")]
        RobotManager(board, robots)
        assert not ({r.get_position() for r in robots} & blocked)


def test_centre_squares_track_the_board_dimensions():
    """Was the literal {(7,7),(7,8),(8,7),(8,8)} -- correct only for 16x16."""

    squares = center_squares()

    assert len(squares) == 4
    assert squares == {(7, 7), (7, 8), (8, 7), (8, 8)}, "expected centre of a 16x16 board"


@pytest.mark.parametrize("level", ALL_LEVELS)
def test_target_deck_loads_and_cycles(level):
    deck = TargetDeck(level_data=level_path(level))

    assert deck.deck, f"{level} defined no targets"

    # `_load_target` copies values into a single long-lived `current_target`
    # rather than pointing at a deck entry, and normalises colour to lower case,
    # so this compares by value and not by identity.
    squares = {(t.x, t.y) for t in deck.deck}
    assert (deck.current_target.x, deck.current_target.y) in squares

    seen = set()
    for _ in range(len(deck.deck)):
        seen.add((deck.current_target.x, deck.current_target.y, deck.current_target.color))
        assert (deck.current_target.x, deck.current_target.y) in squares
        deck.set_new_target()

    assert len(seen) > 1, "the deck never advanced"
