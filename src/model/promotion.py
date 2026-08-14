"""Curriculum promotion policy: *when* to make the puzzles harder.

Split out of `train.py`, which mixed the gradient step, the self-play
orchestration, this policy decision and the log formatting in one 707-line
module of 22 procedural functions.

Not to be confused with `src/game/curriculum.py`, which generates start
positions of a given difficulty. That one decides *what* a level looks like;
this one decides when the agent has earned the next level. Hence the separate
name -- two modules called `curriculum` in the same project would be worse than
either problem they solve.

The gate is deliberately conservative, and the reason is recorded in
`src/game/config.py`: an earlier version read only the solved rate and promoted
on every window it saw, ramping depth 2 -> 12 in ten consecutive windows on
positions that were mostly one move from solved.
"""

import numpy as np

from src.core.logger import logger
from src.game.config import (
    CURRICULUM_MAX_MOVES,
    CURRICULUM_MIN_DEPTH_RATIO,
    CURRICULUM_PROMOTE_THRESHOLD,
    CURRICULUM_PROMOTE_WINDOW,
    CURRICULUM_VERIFY_DEPTH,
)


def maybe_promote_curriculum(config, solved, depths, episode) -> bool:
    """Deepen the reverse curriculum once the agent is reliably solving.

    The curriculum only helps while it is the *edge* of what the agent can do.
    Held at a fixed depth it stops teaching anything the moment the agent
    saturates it; ramped, each level is learned from a position where the
    previous level already supplies most of the answer.

    Promotion needs a full window of recent history so a lucky streak cannot
    push the agent to a depth it cannot handle.

    It also needs the positions to be as hard as the level claims. Scramble depth
    is only an upper bound on true difficulty, so a solved rate on its own can
    promote every single window on positions that never got deeper than one move
    -- the failure this gate exists to catch. `depths` carries the exact optimal
    lengths measured by `game.solver`, and a level whose generated positions are
    too shallow refuses to promote and says so.
    """

    if config.curriculum_moves <= 0 or config.curriculum_moves >= CURRICULUM_MAX_MOVES:
        return False

    window = CURRICULUM_PROMOTE_WINDOW
    if len(solved) < window:
        return False

    recent = list(solved)[-window:]
    if float(np.mean(recent)) < CURRICULUM_PROMOTE_THRESHOLD:
        return False

    measured = [d for d in list(depths)[-window:] if d is not None]

    # Verification off entirely: nothing to check against, so the solved rate is
    # all there is. Deliberately permissive -- opting out is opting out.
    if CURRICULUM_VERIFY_DEPTH:

        # Every episode in the window would repeat these, so report on the
        # window's own cadence.
        loud = episode % window == 0

        if len(measured) < window // 2:
            if loud:
                logger.warning(
                    f"[Curriculum] Episode {episode}: only {len(measured)}/{window} recent "
                    f"episodes had a verifiable optimal depth at level "
                    f"{config.curriculum_moves}. Holding -- promoting here would be "
                    f"promoting on unmeasured difficulty. Raise SOLVER_NODE_BUDGET to "
                    f"push further."
                )
            return False

        mean_depth = float(np.mean(measured))
        required = CURRICULUM_MIN_DEPTH_RATIO * config.curriculum_moves
        if mean_depth < required:
            if loud:
                logger.warning(
                    f"[Curriculum] Episode {episode}: solved {np.mean(recent):.0%} but the "
                    f"generated positions average only {mean_depth:.1f} optimal moves at "
                    f"depth {config.curriculum_moves} (need {required:.1f}). Holding -- the "
                    f"solved rate is measuring easier puzzles than the level claims."
                )
            return False

    config.set_curriculum_moves(config.curriculum_moves + 1)

    depth_note = f", mean optimal depth {np.mean(measured):.1f}" if measured else ""
    logger.info(
        f"[Curriculum] Episode {episode}: solved {np.mean(recent):.0%} of the last "
        f"{window}{depth_note} -- promoting to depth {config.curriculum_moves}."
    )

    # Start the next window fresh; the old history describes the easier level.
    solved.clear()
    depths.clear()
    return True
