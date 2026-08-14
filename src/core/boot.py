"""Terminal primitives for the startup banner and checklist.

The banner used to be one `logger.info(ASCII_LEELA_BOT)` call: thirty lines
landing at once behind a single timestamp, with a checklist baked into the art
that claimed the value/policy/reward nets were loaded before anything had been
built. Here the art streams a line at a time and the checklist reports what the
run actually did -- device, weights, sizes -- after the network exists.

Pacing only happens on a TTY. Under `docker logs` without `-t` there is nobody
watching the frames, so the whole banner is written at full speed.

This module draws rows; it does not know what a training run is. The checklist
*content* lives in `src/model/reporting.py`, because describing a run means
reading the learner's configuration -- and importing that here made `core`, the
bottom layer, depend on `model`, the top one.
"""

import sys
import time

from src.core.config import ASCII_LEELA_BOT

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


def boot_heading(text: str):
    """A dim, unmarked line above a group of checklist rows."""

    _emit(text, DIM)


def boot_blank():
    _emit()
