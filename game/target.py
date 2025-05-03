# Tracks current target position and color
# Handles win condition

import json, os, numpy as np
import pygame
import random

from config import TILE_SIZE, COLOR_MAP, PENGUIN_COLORS, BLACK, GREY

# Get absolute path to the level.json file
current_dir = os.path.dirname(__file__)
level_file = os.path.join(current_dir, "levels", "level_01.json")

class Target:
    
    def __init__(self, x=0, y=0, color=None, target_type=None):
        
        self.x = x
        self.y = y
        self.color = color
        self.target_type = target_type
        

    def _format_target(self):
        
        return f"Target: {self.color.upper()}-{self.target_type.upper()} ({self.x}, {self.y})"


    def draw(self, screen, font):
        
        # Full cell position in pixels
        px = self.x * TILE_SIZE
        py = self.y * TILE_SIZE

        # Target rectangle size (smaller than full cell)
        padding = 3
        target_size = TILE_SIZE - 2 * padding
        target_rect = pygame.Rect(px + padding, py + padding, target_size, target_size)

        # Draw the inner rectangle for the target
        pygame.draw.rect(screen, PENGUIN_COLORS.get(self.color, GREY), target_rect)
        pygame.draw.rect(screen, BLACK, target_rect, 2)  # Border

        # Draw the target letter in the center
        text_surf = font.render(self.target_type, True, GREY)
        text_rect = text_surf.get_rect(center=target_rect.center)
        screen.blit(text_surf, text_rect)

class TargetDeck:
    
    def __init__(self, level_data=level_file):
        
        self.deck = []
        self.current_target = Target()
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


    def _load_target(self, new_target):
        
        self.current_target.x, self.current_target.y = new_target.x, new_target.y
        self.current_target.color = new_target.color.lower()
        self.current_target.target_type = new_target.target_type.upper()
    
    
    def flatten_current_target(self):
        
        position = [self.current_target.x, self.current_target.y]
        color_idx = COLOR_MAP[self.current_target.color.lower()]

        return np.array(position + [color_idx], dtype=np.int32)


    def shuffle_deck(self):
        
        self.deck = self.deck.copy()
        random.shuffle(self.deck)


    def set_new_target(self):
        
        if not self.deck:
            print("Deck is empty!")
            return None
        
        attempts = 0
        while attempts < 10:
            new_target = random.choice(self.deck)

            if new_target != self.current_target:
                self._load_target(new_target)
                return
            attempts += 1
        self._load_target(new_target)  # fallback
    
    def draw_targets(self, screen, font):
        
        for target in self.deck:
            target.draw(screen, font)