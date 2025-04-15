# Initialization
# Runs game loop
# Event/screen handling

# Renders grid
# Stores wall data

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, FPS, WHITE
from game.board import Board
from game.penguins import Penguin

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Arctic Rush")
    clock = pygame.time.Clock()

    # Init board and robots
    board = Board()
    penguins = [
        Penguin("red", 1, 1),
        Penguin("blue", 14, 1),
        Penguin("green", 1, 14),
        Penguin("yellow", 14, 14)
    ]
    selected_index = 0

    running = True
    while running:
        clock.tick(FPS)
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                keys = {
                    pygame.K_UP: 'up',
                    pygame.K_DOWN: 'down',
                    pygame.K_LEFT: 'left',
                    pygame.K_RIGHT: 'right'
                }

                # Move selected robot
                if event.key in keys:
                    penguins[selected_index].move_until_blocked(keys[event.key], board, penguins)

                # Change selected robot
                elif event.key == pygame.K_TAB:
                    selected_index = (selected_index + 1) % len(penguins)

        # Draw everything
        board.draw(screen)
        for i, penguin in enumerate(penguins):
            if i == selected_index:
                # Draw selection ring
                rect = pygame.Rect(penguin.x * TILE_SIZE, penguin.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), rect, 3)
            penguin.draw(screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()