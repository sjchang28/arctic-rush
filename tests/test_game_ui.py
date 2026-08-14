"""Regression tests for the pygame-facing layer.

`src/game/game.py`, `board.py` and `target.py` had no direct tests at all, and
all three bugs pinned here live in exactly that gap. Every test in this file
fails against the pre-fix code.

pygame runs under the dummy video driver (set in `conftest.py`), so these need
no window server.
"""

import os

import pygame
import pytest

from src.game.board import Board, BouncePadManager
from src.game.config import LEVEL_FILE
from src.game.game import Game
from src.game.robots import Robot

_LEVELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "game", "levels")


def _robots():
    return [Robot("red"), Robot("blue"), Robot("green"), Robot("yellow")]


@pytest.fixture(autouse=True)
def _quit_pygame():
    yield
    pygame.quit()


def test_keydown_does_not_depend_on_a_module_level_robots_global():
    """`Game` must work when imported, not only when run as `__main__`.

    `render_player_environment` referenced a bare `robots`, which resolved only
    because the `if __name__ == "__main__"` block happened to bind that name at
    module scope. Imported from anywhere else -- which is every real caller --
    the first arrow key raised `NameError`.

    Driven by posting a real KEYDOWN followed by a QUIT: both are read in the
    same `pygame.event.get()` batch, so the move is processed and then the loop
    exits.
    """

    game = Game(_robots(), render_pygame=True)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    game.render_player_environment()


def test_tab_cycles_the_selected_robot():
    """The TAB branch reads the same undefined `robots` name as the move branch."""

    game = Game(_robots(), render_pygame=True)
    assert game.robot_idx == 0

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    game.render_player_environment()

    assert game.robot_idx == 1


def test_board_loads_bounce_pads_from_the_file_it_was_given():
    """`Board(wall_file=X)` must not read its pads from a different level.

    `Board.__init__` built `BouncePadManager()` with no argument, so it silently
    fell back to the module-level default. A board asked for level_04 got
    level_04's walls and level_01's (nonexistent) bounce pads.

    level_04 is the only level carrying pads, which is what makes this visible.
    """

    level_04 = os.path.join(_LEVELS, "level_04.json")
    assert BouncePadManager(level_04).bounce_pads, "fixture assumption: level_04 has pads"

    board = Board(wall_file=level_04)

    assert board.bounce_pad_manager.bounce_pads, (
        "Board ignored wall_file when loading bounce pads"
    )
    assert len(board.bounce_pad_manager.bounce_pads) == 8


def test_board_default_still_matches_the_configured_level():
    """The fix must not change what a default-constructed Board loads."""

    default_level = os.path.join(_LEVELS, LEVEL_FILE)
    assert Board().bounce_pad_manager.bounce_pads == BouncePadManager(default_level).bounce_pads


def test_close_is_idempotent():
    """`close()` assigned `self.window`, an attribute that does not exist.

    `self.screen` was left dangling, so the `if self.screen` guard stayed true
    and a second close called `pygame.quit()` again.
    """

    game = Game(_robots(), render_pygame=True)

    game.close()
    assert game.screen is None

    game.close()  # must be a no-op, not a second pygame.quit()
