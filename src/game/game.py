# Initialization
# Runs game loop
# Event/screen handling

# Renders grid
# Stores wall data

import pygame

from src.core.logger import logger
from src.game.board import Board
from src.game.config import (
    BOARD_WIDTH,
    DARK_GREY,
    DOWN,
    FPS,
    LEFT,
    RIGHT,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_SIZE,
    UP,
    WHITE,
)
from src.game.robots import Robot, RobotManager
from src.game.target import TargetDeck


class BaseGame:
    """Board, robots, target deck and the pygame surface they are drawn on.

    `Game` (keyboard) and `AI_Game` (agent) previously duplicated all of this:
    two near-identical `__init__`s, two byte-identical `_initialize`s, two
    identical `close`s and two copies of the side-panel drawing. The only real
    differences are when the display is created and what drives the moves, so
    those are all that the subclasses now hold.
    """

    def __init__(self, robots, render_pygame=True):

        self.screen = None
        self.clock = None
        self.font = None

        # Init target deck and manager
        self.board = Board()
        self.robot_manager = RobotManager(self.board, robots)
        self.target_deck = TargetDeck()

        self.robot_idx = 0
        self.move_counter = 0

        self.initialized_game = False
        self.render_pygame = render_pygame


    def _initialize(self):

        if self.initialized_game:
            return

        if self.render_pygame:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 24)
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Arctic Rush")
            self.clock = pygame.time.Clock()

            self.initialized_game = True


    def terminal(self):

        return self.robot_manager.robots[self.robot_idx].is_target_reached(self.target_deck.current_target)


    def _draw_side_panel(self, move_counter):
        """Move count and current target, drawn in the panel beside the board."""

        # Define a "div" as a pygame.Rect
        ui_div_game_information = pygame.Rect((TILE_SIZE * BOARD_WIDTH), 0, (SCREEN_WIDTH - (TILE_SIZE * BOARD_WIDTH)), SCREEN_HEIGHT)  # x, y, width, height
        pygame.draw.rect(self.screen, DARK_GREY, ui_div_game_information)  # dark gray background

        # Display move count
        move_text = f"Total Moves: {move_counter}"
        text_surface = self.font.render(move_text, True, WHITE)
        self.screen.blit(text_surface, (ui_div_game_information.x + 10, 10))

        # Display current target
        target_info = self.target_deck.current_target._format_target()
        text_surface = self.font.render(target_info, True, WHITE)
        self.screen.blit(text_surface, (ui_div_game_information.x + 10, 40))


    def close(self):

        if self.screen:
            pygame.quit()
            # This cleared `self.window`, an attribute that never existed, so
            # `self.screen` stayed truthy and a second close() called
            # pygame.quit() again.
            self.screen = None
            self.clock = None


class Game(BaseGame):
    """Keyboard-driven play."""

    def __init__(self, robots, render_pygame=True):

        super().__init__(robots, render_pygame=render_pygame)

        if self.render_pygame:
            self._initialize()


    def render_player_environment(self):

        running = True
        while running:
            self.clock.tick(FPS)
            self.screen.fill(WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    keys = {
                        pygame.K_UP: UP,
                        pygame.K_DOWN: DOWN,
                        pygame.K_LEFT: LEFT,
                        pygame.K_RIGHT: RIGHT
                    }

                    robots = self.robot_manager.robots

                    # Move selected Robot
                    if event.key in keys:
                        if robots[self.robot_idx].move_until_blocked(simulated=False, direction=keys[event.key], board=self.board, other_robots=robots):
                            self.move_counter += 1
                            if robots[self.robot_idx].is_target_reached(self.target_deck.current_target):
                                logger.info(f"You Won in {self.move_counter} moves!")
                                self.target_deck.set_new_target()
                                self.robot_manager.reset_robots()
                                self.move_counter = 0

                    # Change selected Robot
                    elif event.key == pygame.K_TAB:
                        self.robot_idx = (self.robot_idx + 1) % len(robots)

            # Draw everything (board, targets, robots)
            self.board.draw_board(self.screen)
            self.board.bounce_pad_manager.draw_bounce_pads(self.screen)
            self.target_deck.draw_targets(self.screen, self.font)
            self.robot_manager.draw_robots(self.screen, self.robot_idx)

            self._draw_side_panel(self.move_counter)

            pygame.display.flip()

        self.close()


class AI_Game(BaseGame):
    """Agent-driven play.

    Defaults to no display and does not initialise pygame in the constructor:
    training runs thousands of episodes headless, and `render_ai_environment`
    brings the window up on demand.
    """

    def __init__(self, robots, render_pygame=False):

        super().__init__(robots, render_pygame=render_pygame)


    def render_ai_environment(self, robot_idx=0, move_counter=0):

        if self.render_pygame and not self.initialized_game:
            self._initialize()

        self.screen.fill(WHITE)

        # Draw everything (board, targets, robots)
        self.board.draw_board(self.screen)
        self.target_deck.draw_targets(self.screen, self.font)
        self.robot_manager.draw_robots(self.screen, robot_idx)

        self._draw_side_panel(move_counter)

        pygame.display.flip()
        self.clock.tick(FPS)


def main():
    """Play the game with a keyboard.

    Deliberately a function rather than bare module-level statements: binding
    `robots` at module scope is what let `render_player_environment` reference an
    undefined global and still work when this file was run directly. Keeping the
    names local means pyflakes (`ruff check`) catches any recurrence.
    """

    robots = [
        Robot("red"),
        Robot("blue"),
        Robot("green"),
        Robot("yellow")
    ]

    Game(robots).render_player_environment()


if __name__ == "__main__":
    main()
