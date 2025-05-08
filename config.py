LEVEL_FILE = "level_01.json"

TILE_SIZE = 40
TARGET_SIZE = 30
BOARD_WIDTH = 16
BOARD_HEIGHT = 16
SCREEN_WIDTH = TILE_SIZE * BOARD_WIDTH * 1.5
SCREEN_HEIGHT = TILE_SIZE * BOARD_HEIGHT
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
DARK_GREY = (33, 33, 33)
PINK = (231, 50, 189)
ROBOT_COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0)
}
NUMBER_OF_ROBOTS = len(ROBOT_COLORS)

COLOR_MAP = {
    "red": 0, 
    "blue": 1, 
    "green": 2, 
    "yellow": 3,
    "any": 4
}
INDEX_COLOR_MAP = {
    0: "red", 
    1: "blue", 
    2: "green", 
    3: "yellow",
    4: "any"
}

# Directions for readability
UP, RIGHT, DOWN, LEFT, SWITCH = 'up', 'right', 'down', 'left', 'switch'
ALL_MOVES = [UP, RIGHT, DOWN, LEFT, SWITCH]
NUMBER_OF_POSSIBLE_MOVES = len(ALL_MOVES)
ALL_DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
INT2DIRECTION = {
    0: UP,
    1: RIGHT,
    2: DOWN,
    3: LEFT,
    4: SWITCH
}
DIRECTION2INT = {
    UP: 0,
    RIGHT: 1,
    DOWN: 2,
    LEFT: 3,
    SWITCH: 4
}

# Train LEELA
WEIGHTS_FILE_PATH = "leela.pth"
TOTAL_TIMESTEPS_FOR_TRAINING = 500 * 1000  # 500k timesteps
TOTAL_MCTS_EPISODES = 100
MAX_TOTAL_MOVES_PER_GAME = 100 # ? Includes tabbing/switching