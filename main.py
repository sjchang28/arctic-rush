# Initialization
# Runs game loop
# Event/screen handling

# Renders grid
# Stores wall data

import pygame
from config import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE, FPS, WHITE
from game.board import Board
from game.penguins import Robot

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ricochet Robots")
    clock = pygame.time.Clock()

    # Init board and robots
    board = Board()
    robots = [
        Robot("red", 1, 1),
        Robot("blue", 14, 1),
        Robot("green", 1, 14),
        Robot("yellow", 14, 14)
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
                    robots[selected_index].move_until_blocked(keys[event.key], board, robots)

                # Change selected robot
                elif event.key == pygame.K_TAB:
                    selected_index = (selected_index + 1) % len(robots)

        # Draw everything
        board.draw(screen)
        for i, robot in enumerate(robots):
            if i == selected_index:
                # Draw selection ring
                rect = pygame.Rect(robot.x * TILE_SIZE, robot.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), rect, 3)
            robot.draw(screen)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()