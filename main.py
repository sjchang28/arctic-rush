# Initialization
# Runs game loop
# Event/screen handling

# Renders grid
# Stores wall data

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, GRID_WIDTH, FPS, WHITE, PINK, UP, DOWN, LEFT, RIGHT
from game.board import Board
from game.target import TargetDeck, TargetManager
from game.penguins import Penguin

def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Arctic Rush")
    clock = pygame.time.Clock()

    # Init board and penguins
    board = Board()
    penguins = [
        Penguin("red", 1, 1),
        Penguin("blue", 14, 1),
        Penguin("green", 1, 14),
        Penguin("yellow", 14, 14)
    ]
    target_deck = TargetDeck()
    target_deck.set_new_target()
    target_manager = TargetManager(board, target_deck)
    selected_index = 0
    move_counter = 0

    running = True
    while running:
        clock.tick(FPS)
        screen.fill(WHITE)

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
                    if penguins[selected_index].move_until_blocked(keys[event.key], board, penguins):
                        move_counter += 1
                        if target_manager.check_target_reached(penguins, target_deck.target_position, target_deck.target_color):
                            print("You Won in " + str(move_counter) + " moves!")
                            target_deck.set_new_target()
                            move_counter = 0

                # Change selected penguin
                elif event.key == pygame.K_TAB:
                    selected_index = (selected_index + 1) % len(penguins)

        # Draw everything
        board.draw_board(screen)
        for i, target in enumerate(target_deck.deck):
            target.draw(screen, font)
        for i, penguin in enumerate(penguins):
            if i == selected_index:
                # Full cell position in pixels
                px = penguin.x * TILE_SIZE
                py = penguin.y * TILE_SIZE

                # Target rectangle size (smaller than full cell)
                padding = 2
                targeted_penguin_size = TILE_SIZE - 2 * padding
                targeted_penguin_rect = pygame.Rect(px + padding, py + padding, targeted_penguin_size, targeted_penguin_size)

                # Draw the inner rectangle for the target
                pygame.draw.rect(screen, PINK, targeted_penguin_rect, 3)

            penguin.draw(screen)

        # Define a "div" as a pygame.Rect
        ui_div_game_information = pygame.Rect((TILE_SIZE * GRID_WIDTH), 0, (SCREEN_WIDTH - (TILE_SIZE * GRID_WIDTH)), SCREEN_HEIGHT)  # x, y, width, height
        pygame.draw.rect(screen, (33, 33, 33), ui_div_game_information)  # dark gray background

        # Display move count
        move_text = f"Total Moves: {move_counter}"
        text_surface = font.render(move_text, True, WHITE)
        screen.blit(text_surface, (ui_div_game_information.x + 10, 10))

        # Display current target
        target_info = f"Target: {target_deck.target_color.upper()}-{target_deck.target_type}"
        text_surface = font.render(target_info, True, WHITE)
        screen.blit(text_surface, (ui_div_game_information.x + 10, 40))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()