"""Start-position generation for the reverse curriculum.

Extracted from `RicochetRobotsEnv`, which was a 477-line class doing three jobs:
implementing the Gymnasium step/observation contract, snapshotting state for the
search, and generating start positions of a requested difficulty. The third is
the only one with its own tuning constants and its own failure modes, and it is
what this file holds.

The generator reads and mutates the environment's robots and target rather than
owning them: a start position *is* a mutation of the live board, and copying it
out and back would be both slower and a second source of truth.

Note on structure: `generate` is long, and deliberately so. Its docstring
records measured generator statistics that justify every branch, and the
attempt loop is one algorithm -- draw a candidate, solve it exactly, keep the
closest -- whose parts are not independently meaningful. Only the pool lookup,
which genuinely is a separate concern, has been split off into
`_take_from_pool`.
"""

import collections
import random

from src.game.config import (
    CURRICULUM_POOL_MIN,
    CURRICULUM_POOL_REFRESH,
    CURRICULUM_POOL_SIZE,
    CURRICULUM_SCRAMBLE_ATTEMPTS,
    CURRICULUM_SOLVER_BIAS,
    CURRICULUM_VERIFY_DEPTH,
    CURRICULUM_WALK_MAX,
    CURRICULUM_WALK_MIN,
    CURRICULUM_WALK_PER_DEPTH,
)
from src.game.solver import SEARCH_EXHAUSTED


class CurriculumGenerator:
    """Produces start positions of a measured optimal depth."""

    def __init__(self, env):

        self.env = env

        # Verified start positions, keyed by optimal depth. Solving a candidate
        # exactly costs milliseconds at depth 3 but seconds at depth 6, which is
        # more than a whole episode; without reuse, verification would dominate
        # the run. Positions are kept with the target they were verified
        # against, since difficulty is a property of the pair.
        self.depth_pool = {}

    @property
    def robots(self):
        return self.env.robots

    @property
    def curriculum_moves(self):
        return self.env.curriculum_moves

    def generate(self, target, depth=None):
        """Start a position whose *measured* optimal depth is `depth`.

        `depth` defaults to the current curriculum level. It is passed explicitly
        only to rehearse a base case (see `env.reset`), which needs a shallow
        position without moving the level the agent is being assessed at.

        Returns the depth actually produced, or None when it could not be
        measured. That number, not the requested one, is what the promotion gate
        and TensorBoard read.

        Three generators, because none covers the range alone. Measured on
        level_01 against BFS, 60-80 samples per cell:

          * Backward scramble from solved (sampling predecessor states): ~85% of
            its output is one move from the goal, and that does not change with
            scramble length (tested 2 -> 32) or with how strongly the goal robot
            is favoured. Sliding is long-range, so a robot anywhere on the
            target's row or column is still one move out, and a backward random
            walk essentially never escapes that set. Kept only as a reliable
            depth-1-to-2 source.

          * Forward walk from solved -- start on the goal and play `n` random
            *legal* moves. Unlike the backward scramble this actually responds to
            `n`: mean optimal depth 4.5 at n=16, 6.4 at n=64. Moves are not
            invertible here, so `n` is not the resulting depth and the walk can
            wander back onto the goal; the solver settles it either way.

          * Uniformly random placement: mean optimal depth 5.8, but with a fixed
            distribution that cannot be asked for more.

        The forward walk is what reaches the deep tail: at n=64-128 it produces a
        position needing 8+ moves 31-35% of the time against random placement's
        21%, and 10+ moves 6-13% against 3%. That difference is the curriculum's
        ceiling, so deep levels are walked and shallow ones are scrambled, with
        random placement mixed in throughout for variety.
        """

        if depth is None:
            depth = self.curriculum_moves

        matching = [
            index for index, robot in enumerate(self.robots)
            if target.color.upper() == "ANY" or robot.color.lower() == target.color.lower()
        ]
        solver_index = matching[0] if matching else None

        if not CURRICULUM_VERIFY_DEPTH or solver_index is None:
            self.random_once(target)
            return None

        pool = self.depth_pool.setdefault(
            depth,
            collections.deque(maxlen=max(1, CURRICULUM_POOL_SIZE)),
        )

        if self._take_from_pool(pool):
            return depth

        best_positions, best_error, best_depth = None, None, None

        # Shallow levels are a thin slice -- the generators mostly land on depth 1
        # or on 3 and up, so depth 2 in particular needs more draws to hit. It can
        # afford them: verifying a depth-2 candidate costs single-digit
        # milliseconds against most of a second at depth 6.
        attempts = max(1, CURRICULUM_SCRAMBLE_ATTEMPTS)
        if depth <= 2:
            attempts *= 4

        for attempt in range(attempts):

            # Alternate the depth-seeking generator with plain random placement:
            # the first supplies the difficulty, the second keeps the positions
            # from all sharing a common ancestor in the solved state.
            if attempt % 2 == 1:
                self.random_once(target)
            elif depth <= 2:
                self.scramble_once(target, solver_index, depth)
            else:
                self.forward_walk_once(target, solver_index, depth)

            # Only the verdict "is this position exactly `curriculum_moves`
            # deep?" is needed, so the search stops one level past it. Solving to
            # the true depth of a deep position costs seconds; this costs
            # milliseconds, and every episode pays it.
            measured = self.env.measure_optimal_depth(max_depth=depth + 1)

            # Gave up rather than concluded. Accepting these would quietly
            # reinstate the original bug -- an unverified position counted as a
            # deep one -- so they are not eligible at all.
            if measured == SEARCH_EXHAUSTED:
                continue

            # Proven deeper than the bound. Recorded as one past the level rather
            # than as its unknown true depth: the promotion gate reads these as a
            # floor, so understating is the safe direction.
            reached = depth + 1 if measured is None else measured

            # Degenerate: a robot already sits on the goal, so there is no puzzle.
            if reached == 0:
                continue

            error = abs(reached - depth)
            if best_error is None or error < best_error:
                best_error, best_depth = error, reached
                best_positions = [(r.x, r.y) for r in self.robots]

            if error == 0:
                break

        if best_positions is None:
            # Nothing could be verified within the budget -- the usual cause is a
            # level deep enough that the search gives up. Play a random position
            # and report the depth as unknown rather than inventing one; the
            # promotion gate refuses to advance on unlabelled episodes.
            self.random_once(target)
            return None

        # The last attempt is not necessarily the best one, so restore explicitly.
        for robot, position in zip(self.robots, best_positions):
            robot.x, robot.y = position
            robot.prev_x, robot.prev_y = position
            robot.robotLeftTarget = False

        # Only exact matches are worth keeping; a near miss would drift the
        # pool's difficulty away from the level it is filed under.
        if best_depth == depth:
            pool.append(((target.x, target.y, target.color), list(best_positions)))

        return best_depth

    def _take_from_pool(self, pool) -> bool:
        """Install a pooled position if the pool is ready to be drawn from.

        Draw once the pool holds enough positions to be varied, but keep
        generating on a fraction of resets so it goes on growing and the agent is
        not shown the same handful of boards for a whole level.
        """

        if len(pool) < CURRICULUM_POOL_MIN or random.random() <= CURRICULUM_POOL_REFRESH:
            return False

        self.apply_pooled(random.choice(pool))
        return True

    def apply_pooled(self, entry):
        """Install a previously verified (target, positions) pair.

        Difficulty belongs to the pair, not to the positions alone, so the target
        is restored alongside the robots -- the same mutation `restore` performs.
        """

        (target_x, target_y, target_color), positions = entry

        target = self.env.current_target
        target.x, target.y, target.color = target_x, target_y, target_color

        for robot, position in zip(self.robots, positions):
            robot.x, robot.y = position
            robot.prev_x, robot.prev_y = position
            robot.robotLeftTarget = False

    def random_once(self, target):
        """Place every robot on a random free square, keeping the goal clear."""

        self.env.place_robots_randomly(forbidden_extra={(target.x, target.y)})

    def forward_walk_once(self, target, solver_index, depth):
        """Start solved and play random legal moves until the position is messy.

        The walk length is scaled to the requested depth but deliberately much
        longer than it. Moves here are not invertible -- a slide runs until it is
        blocked -- so `n` steps away from the goal is not `n` moves back, and the
        walk can even wander onto the goal again. It is a mixing process, not a
        distance: the length only has to be enough to reach the depth being asked
        for, and the solver decides whether it did.

        The goal robot is moved first so the position does not start out solved,
        which is the walk's one systematic failure mode at short lengths.
        """

        self.random_once(target)

        solver = self.robots[solver_index]
        solver.x, solver.y = target.x, target.y
        for robot in self.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False

        manager = self.env.game.robot_manager
        steps = min(
            CURRICULUM_WALK_MAX,
            max(CURRICULUM_WALK_MIN,
                CURRICULUM_WALK_PER_DEPTH * depth),
        )

        for step in range(steps):

            moves = manager.get_all_legal_moves(selected_idx=None)

            if step == 0:
                moves = [move for move in moves if move[0] == solver_index] or moves

            if not moves:
                break

            robot_index, direction = moves[random.randrange(len(moves))]
            self.robots[robot_index].move_until_blocked(
                simulated=False, direction=direction,
                board=self.env.game.board, other_robots=self.robots,
            )

        for robot in self.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False

    def scramble_once(self, target, solver_index, depth):
        """Re-roll the robots and walk one fresh scramble backwards from solved.

        Re-rolling every attempt matters: repeating the scramble from the same
        blocker layout produces correlated candidates.
        """

        # Robots are placed off the target, so moving the solver onto it cannot
        # collide with anyone.
        self.random_once(target)

        solver = self.robots[solver_index]
        solver.x, solver.y = target.x, target.y

        for robot in self.robots:
            robot.prev_x, robot.prev_y = robot.x, robot.y
            robot.robotLeftTarget = False

        self.env.game.robot_manager.reverse_scramble(
            depth,
            required_first=solver_index,
            solver_index=solver_index,
            solver_bias=CURRICULUM_SOLVER_BIAS,
            avoid_square=(target.x, target.y),
        )
