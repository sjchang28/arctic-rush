"""Episode and moving-average log lines.

Split out of `train.py`. Formatting a log line and running a gradient step are
unrelated concerns; these were interleaved with the optimiser code.

`reporting.py` holds the one-off startup checklist; this holds the per-episode
output.
"""

import numpy as np

from src.core.config import LOG_LONG_AVG_EVERY, LOG_SHORT_AVG_EVERY
from src.core.logger import logger


def _fmt(value, spec=".3f"):
    """Format a float, rendering NaN as a dash so columns stay aligned."""
    return "--".rjust(len(format(0.0, spec))) if value != value else format(value, spec)


def excess_moves(game):
    """Moves spent above the optimum, or None when either number is unknown.

    This is the metric that says whether the agent is playing *well* rather than
    merely finishing: zero means it found a shortest solution.
    """

    if game.optimal_depth is None or not game.is_terminal():
        return None

    return game.total_moves() - game.optimal_depth


def log_episode(episode, total_episodes, game, reward, loss):
    """One compact line per episode: index, outcome, reward, loss.

    The per-episode line is the log's backbone, so it stays a single row of
    fixed-width columns -- three wrapped lines per episode made consecutive
    episodes impossible to tell apart in `docker logs`.
    """

    width = len(str(total_episodes))
    solved = game.is_terminal()

    # Only the verdict is coloured; padding sits outside the tag so the columns
    # after it still line up.
    verdict = "solved" if solved else "failed"
    colour = "green" if solved else "red"
    tail = (f" in {game.total_moves():>3} moves" if solved
            else f" after {game.total_moves():>3} moves").ljust(22 - len(verdict))

    # "solved in 1" reads as success until you know the position needed 1 move.
    optimal = game.optimal_depth
    best = f"| best {optimal:>2} " if optimal is not None else ""

    logger.opt(colors=True).info(
        f"Ep {episode:>{width}}/{total_episodes} | <{colour}>{verdict}</{colour}>{tail} "
        f"{best}| reward {_fmt(reward, '+.2f')} | loss {_fmt(loss, '.4f')}"
    )


def log_moving_averages(episode, rewards, losses, solved, depths=()):
    """Emit the rolling averages on their own cadence, highlighted in blue.

    nanmean: episodes before MIN_REPLAY_GAMES record no loss.
    """

    for window in (LOG_SHORT_AVG_EVERY, LOG_LONG_AVG_EVERY):

        if window <= 0 or episode % window != 0:
            continue

        rate = float(np.mean(solved[-window:])) if solved else float('nan')

        # The solved rate is only interpretable next to the difficulty it was
        # earned on, so the two are reported together or not at all.
        measured = [d for d in list(depths)[-window:] if d is not None]
        depth_note = f"| best {float(np.mean(measured)):.1f} " if measured else ""

        logger.opt(colors=True).info(
            f"<blue>---- last {window:>3} episodes @ Ep {episode} "
            f"| solved {rate:.0%} "
            f"{depth_note}"
            f"| reward {_fmt(float(np.mean(rewards[-window:])), '+.2f')} "
            f"| loss {_fmt(float(np.nanmean(losses[-window:])), '.4f')} ----</blue>"
        )
