"""What to say about a training run.

Split from `src/core/boot.py`, which draws checklist rows but has no business
knowing what a curriculum or a replay window is. Reading the learner's
configuration from `core` made the bottom layer depend on the top one, so
importing the startup banner pulled in `src.model.config` and, through it, the
board geometry.

The rendering primitives still live in `core.boot`; only the content is here.
"""

import os

from src.config import settings
from src.core.boot import boot_blank, boot_heading, boot_step
from src.game.config import (
    CURRICULUM_MAX_MOVES,
    CURRICULUM_SCRAMBLE_ATTEMPTS,
    CURRICULUM_VERIFY_DEPTH,
)
from src.model.config import (
    MIN_REPLAY_GAMES,
    NUM_BLOCKS,
    NUM_CHANNELS,
    WEIGHTS_FILE_PATH,
)


def report_ready(config, network, device):
    """Checklist of what this run is actually configured to do.

    Called once the network exists so every line is a fact rather than a
    promise: whether checkpoint weights were found, how big the model is, and
    the knobs that decide how long the run takes.
    """

    weights_path = os.path.join(settings.MODEL_DIR, settings.RUN_ID, WEIGHTS_FILE_PATH)
    resumed = os.path.exists(weights_path)
    parameters = sum(p.numel() for p in network.parameters())

    boot_heading(f"  Run {settings.RUN_ID}")

    boot_step("Device", str(device).upper())
    boot_step("Weights", f"resumed from {weights_path}" if resumed else "fresh initialisation")
    boot_step("Checkpoint", "best only [by curriculum level, then solved rate]"
                            if settings.SAVE_BEST_ONLY else "every episode")
    boot_step("Network", f"{parameters/1e6:.2f}M parameters "
                         f"[{NUM_BLOCKS} blocks x {NUM_CHANNELS} channels]")
    boot_step("Search", f"{config.search_mode} [{config.num_simulations} simulations/move]")
    actors = f"{config.num_actors} actor" + ("s" if config.num_actors != 1 else "")
    boot_step("Self-play", f"{config.training_episodes} episodes x {actors} "
                           f"[{settings.TRAIN_STEPS_PER_EPISODE} gradient steps/episode]")
    verified = (f", BFS-verified x{CURRICULUM_SCRAMBLE_ATTEMPTS}"
                if CURRICULUM_VERIFY_DEPTH else ", unverified depth")
    boot_step("Curriculum", f"depth {config.curriculum_moves} -> {CURRICULUM_MAX_MOVES}"
                            f"{verified}"
              if config.curriculum_moves > 0 else "disabled")
    boot_step("Replay", f"window {config.window_size} games "
                        f"[training starts at {MIN_REPLAY_GAMES}]")

    boot_blank()
