?"""Startup banner and readiness checklist.

The banner used to be one `logger.info(ASCII_LEELA_BOT)` call: thirty lines
landing at once behind a single timestamp, with a checklist baked into the art
that claimed the value/policy/reward nets were loaded before anything had been
built. Here the art streams a line at a time and the checklist reports what the
run actually did -- device, weights, sizes -- after the network exists.

Pacing only happens on a TTY. Under `docker logs` without `-t` there is nobody
watching the frames, so the whole banner is written at full speed.
"""

import os
import sys
import time

from src.config import ASCII_LEELA_BOT, settings

ART_LINE_DELAY = 0.03
STEP_DELAY = 0.10

CYAN, GREEN, RED, DIM = "36", "32", "31", "90"


def _animated() -> bool:
    """Only pace output when a human is watching it live."""

    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _ascii_only() -> bool:
    """True when the attached stream cannot encode the check mark."""

    encoding = getattr(sys.stderr, "encoding", None) or "ascii"
    try:
        "✓".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return True
    return False


def _emit(text: str = "", colour: str = CYAN):
    # Written straight to stderr rather than through loguru: the banner is a
    # single visual block, and a timestamp/level prefix on every row of ASCII
    # art is what made it unreadable.
    sys.stderr.write(f"\033[{colour}m{text}\033[0m\n")
    sys.stderr.flush()


def _pause(delay: float):
    if _animated():
        time.sleep(delay)


def stream_banner(art: str = ASCII_LEELA_BOT, delay: float = ART_LINE_DELAY):
    """Draw the ASCII art one line at a time."""

    for line in art.strip("\n").splitlines():
        _emit(line)
        _pause(delay)

    _emit()


def boot_step(label: str, detail: str = "", ok: bool = True):
    """One checklist row: `[✓] label   detail`."""

    mark = ("OK" if ok else "--") if _ascii_only() else (" ✓" if ok else " ✗")
    _emit(f"  [{mark}] {label:<18} {detail}", GREEN if ok else RED)
    _pause(STEP_DELAY)


def report_ready(config, network, device):
    """Checklist of what this run is actually configured to do.

    Called once the network exists so every line is a fact rather than a
    promise: whether checkpoint weights were found, how big the model is, and
    the knobs that decide how long the run takes.
    """

    weights_path = os.path.join(settings.MODEL_DIR, settings.RUN_ID, settings.WEIGHTS_FILE_PATH)
    resumed = os.path.exists(weights_path)
    parameters = sum(p.numel() for p in network.parameters())

    _emit(f"  Run {settings.RUN_ID}", DIM)

    boot_step("Device", str(device).upper())
    boot_step("Weights", f"resumed from {weights_path}" if resumed else "fresh initialisation")
    boot_step("Checkpoint", "best only [by curriculum level, then solved rate]"
                            if settings.SAVE_BEST_ONLY else "every episode")
    boot_step("Network", f"{parameters/1e6:.2f}M parameters "
                         f"[{settings.NUM_BLOCKS} blocks x {settings.NUM_CHANNELS} channels]")
    boot_step("Search", f"{config.search_mode} [{config.num_simulations} simulations/move]")
    actors = f"{config.num_actors} actor" + ("s" if config.num_actors != 1 else "")
    boot_step("Self-play", f"{config.training_episodes} episodes x {actors} "
                           f"[{settings.TRAIN_STEPS_PER_EPISODE} gradient steps/episode]")
    verified = (f", BFS-verified x{settings.CURRICULUM_SCRAMBLE_ATTEMPTS}"
                if settings.CURRICULUM_VERIFY_DEPTH else ", unverified depth")
    boot_step("Curriculum", f"depth {config.curriculum_moves} -> {settings.CURRICULUM_MAX_MOVES}"
                            f"{verified}"
              if config.curriculum_moves > 0 else "disabled")
    boot_step("Replay", f"window {config.window_size} games "
                        f"[training starts at {settings.MIN_REPLAY_GAMES}]")

    _emit()
