# Tracks current target position and color
# Handles win condition

import json, os
import pygame
import random

from config import TILE_SIZE, PENGUIN_COLORS, BLACK, GREY

# Get absolute path to the level.json file
current_dir = os.path.dirname(__file__)
level_file = os.path.join(current_dir, "levels", "level_01.json")

class Target:
    def __init__(self, x, y, color, target_type):
        self.x = x
        self.y = y
        self.color = color
        self.target_type = target_type
        self.cell_size = TILE_SIZE

    def __repr__(self):
        return f"Target(position=({self.x}, {self.y}), color={self.color}, type={self.type})"

    def draw(self, screen, font):
        # Full cell position in pixels
        px = self.x * self.cell_size
        py = self.y * self.cell_size

        # Target rectangle size (smaller than full cell)
        padding = 3
        target_size = self.cell_size - 2 * padding
        target_rect = pygame.Rect(px + padding, py + padding, target_size, target_size)

        # Draw the inner rectangle for the target
        pygame.draw.rect(screen, PENGUIN_COLORS.get(self.color, (200, 200, 200)), target_rect)
        pygame.draw.rect(screen, BLACK, target_rect, 2)  # Border

        # Draw the target letter in the center
        text_surf = font.render(self.target_type, True, GREY)
        text_rect = text_surf.get_rect(center=target_rect.center)
        screen.blit(text_surf, text_rect)

class TargetDeck:
    def __init__(self, level_data=level_file):
        self.deck = []
        self.deck_index = 0
        self.target_position = None
        self.target_color = None
        self.target_type = None

        self._load_custom_deck(level_data)

        self.shuffle_deck()
        self._load_target(self.deck[0])

    def _load_custom_deck(self, deck_file):
        if not os.path.exists(deck_file):
            print(f"❌ Deck file '{deck_file}' not found.")
            return

        with open(deck_file, "r") as f:
            data = json.load(f)

        for target in data.get("deck", []):
            self.deck.append(Target(
                target["position"][0], target["position"][1], target["color"], target["type"]
            ))

    def _load_target(self, target_dict):
        self.target_position = tuple([target_dict.x, target_dict.y])
        self.target_color = target_dict.color.lower()
        self.target_type = target_dict.target_type.upper()

    def shuffle_deck(self):
        self.deck = self.deck.copy()
        random.shuffle(self.deck)

    def set_new_target(self):
        if not self.deck:
            print("Deck is empty!")
            return None
        
        previous_target = {
            "position": self.target_position,
            "color": self.target_color,
            "type": self.target_type
        }
        
        attempts = 0
        while attempts < 10:
            target = random.choice(self.deck)
            self.deck_index += 1

            if ((target.x, target.y) != list(previous_target["position"]) or
                (target.color) != previous_target["color"] or
                (target.target_type) != previous_target["type"]):
                self._load_target(target)
                return
            attempts += 1
        self._load_target(target)  # fallback
    
class TargetManager:
    def __init__(self, board, deck):
        self.board = board
        self.deck = deck

    def check_target_reached(self, penguins, target_position, target_color):
        for penguin in penguins:
            if tuple([penguin.x, penguin.y]) == target_position:
                # If target is ANY, any penguin can reach
                if target_color.upper() == "ANY":
                    return True
                # Match penguin color and type
                if penguin.color.lower() == target_color:
                    return True
        return False

