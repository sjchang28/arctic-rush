# Initialization
# Runs game loop
# Event/screen handling

# Renders grid
# Stores wall data

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, BOARD_WIDTH, FPS, WHITE, DARK_GREY, UP, DOWN, LEFT, RIGHT
from game.board import Board
from game.target import TargetDeck
from game.penguins import Penguin, PenguinManager

class Game():
    
    def __init__(self, penguins):
        
        self.screen = None
        self.clock = None
        self.font = None

        # Init target deck and manager
        self.board = Board()
        self.penguin_manager = PenguinManager(self.board, penguins)
        self.target_deck = TargetDeck()

        self.penguin_idx = 0
        self.move_counter = 0
        
        #self._initialize()


    def _initialize(self):
        
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 24)
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Arctic Rush")
            self.clock = pygame.time.Clock()
    
    
    def terminal(self):
        
        return self.penguin_manager.penguins[self.penguin_idx].is_target_reached(self.target_deck.current_target)


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

                    # Move selected penguin
                    if event.key in keys:
                        if penguins[self.penguin_idx].move_until_blocked(simulated=False, direction=keys[event.key], board=self.board, other_penguins=penguins):
                            self.move_counter += 1
                            if penguins[self.penguin_idx].is_target_reached(self.target_deck.current_target):
                                print("You Won in " + str(self.move_counter) + " moves!")
                                self.target_deck.set_new_target()
                                self.penguin_manager.reset_penguins()
                                self.move_counter = 0

                    # Change selected penguin
                    elif event.key == pygame.K_TAB:
                        self.penguin_idx = (self.penguin_idx + 1) % len(penguins)

            # Draw everything (board, targets, penguins)
            self.board.draw_board(self.screen)
            self.target_deck.draw_targets(self.screen, self.font)
            self.penguin_manager.draw_penguins(self.screen, self.penguin_idx)
            
            # Define a "div" as a pygame.Rect
            ui_div_game_information = pygame.Rect((TILE_SIZE * BOARD_WIDTH), 0, (SCREEN_WIDTH - (TILE_SIZE * BOARD_WIDTH)), SCREEN_HEIGHT)  # x, y, width, height
            pygame.draw.rect(self.screen, DARK_GREY, ui_div_game_information)  # dark gray background

            # Display move count
            move_text = f"Total Moves: {self.move_counter}"
            text_surface = self.font.render(move_text, True, WHITE)
            self.screen.blit(text_surface, (ui_div_game_information.x + 10, 10))

            # Display current target
            target_info = self.target_deck.current_target._format_target()
            text_surface = self.font.render(target_info, True, WHITE)
            self.screen.blit(text_surface, (ui_div_game_information.x + 10, 40))

            pygame.display.flip()
        
        self.close()


    def render_ai_environment(self, penguin_idx=0, move_counter=0):
        
        if not pygame.get_init():
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            
        self.screen.fill(WHITE)

        # Draw everything (board, targets, penguins)
        self.board.draw_board(self.screen)
        self.target_deck.draw_targets(self.screen, self.font)
        self.penguin_manager.draw_penguins(self.screen, penguin_idx)
        
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

        pygame.display.flip()
        self.clock.tick(FPS)
    
    
    def close(self):
        
        if self.screen:
            pygame.quit()
            self.window=None
            self.clock=None


if __name__ == "__main__":
    
    # Init board and penguins
    penguins = [
        Penguin("red"),
        Penguin("blue"),
        Penguin("green"),
        Penguin("yellow")
    ]

    arcticRush = Game(penguins)
    arcticRush.render_player_environment()