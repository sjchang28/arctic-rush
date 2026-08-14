"""Exact shortest-solution search over the real simulator.

Ricochet Robots is deterministic and fully observable and a whole position is
four robot coordinates, so breadth-first search returns the true optimal move
count rather than an estimate. Two parts of this project need that number:

  * The reverse curriculum. Scrambling `N` moves backwards from a solved
    position only bounds the difficulty from above -- a backward move is not the
    inverse of a forward one here, moves slide until blocked, and most backward
    moves shuffle robots that are not the one that has to reach the goal. Left
    unchecked the generator happily labels a one-move position "depth 12".

  * Evaluation. "Solved in 3 moves" carries no information without the optimum
    to measure it against; the gap between the two is the real score.

The search is bounded on both depth and expanded nodes. Shallow positions --
the common case, and the ones the curriculum is trying to detect -- terminate
almost immediately, while genuinely deep ones exhaust the budget and report
`None`, which callers read as "at least this deep".
"""

from collections import deque

from src.game.config import ALL_DIRECTIONS
from src.game.robots import Robot

#: Returned when the node budget ran out before the search finished. It is *not*
#: the same as "no solution": the search simply stopped looking, so the position
#: could be shallow or deep and nothing here can say which. Callers must treat it
#: as unknown. Folding it into "deep" is how a difficulty check quietly turns
#: back into the bug it was written to catch.
SEARCH_EXHAUSTED = -1


def shortest_solution_length(board, robots, target, max_depth=12, node_budget=50_000):

    """Optimal number of moves to satisfy `target` from the robots' positions.

    Returns:
        0 -- a matching robot already stands on the goal (degenerate, not a puzzle)
        n -- proven optimal, n <= max_depth
        None -- proven to need more than `max_depth` moves (search completed)
        SEARCH_EXHAUSTED -- gave up after `node_budget` expansions; unknown

    `robots` is read, never mutated: the search runs on throwaway copies.
    """

    goal_square = (target.x, target.y)
    any_color = target.color.upper() == "ANY"

    goal_indices = [
        index for index, robot in enumerate(robots)
        if any_color or robot.color.lower() == target.color.lower()
    ]
    if not goal_indices:
        return None

    start = tuple((robot.x, robot.y) for robot in robots)

    if any(start[index] == goal_square for index in goal_indices):
        return 0

    # move_until_blocked mutates the robot it moves, so search on stand-ins.
    scratch = [Robot(robot.color) for robot in robots]

    seen = {start}
    frontier = deque([(start, 0)])
    expanded = 0

    while frontier:

        positions, depth = frontier.popleft()

        # FIFO order makes depth non-decreasing, so the first node past the bound
        # means every node left is too. Draining that last layer one `continue` at
        # a time is the single most expensive thing this search can do, because
        # the final layer is most of the frontier.
        if depth >= max_depth:
            break

        expanded += 1
        if expanded > node_budget:
            return SEARCH_EXHAUSTED

        # Load the node once; a move displaces exactly one robot, so only that
        # one needs rewinding between trials. The board is small and the search
        # is the per-episode cost, so this inner loop is worth the fuss.
        for stand_in, position in zip(scratch, positions):
            stand_in.x, stand_in.y = position

        for index, robot in enumerate(scratch):

            home = positions[index]

            for direction in ALL_DIRECTIONS:

                robot.x, robot.y = home

                if not robot.move_until_blocked(
                    simulated=False, direction=direction,
                    board=board, other_robots=scratch
                ):
                    continue

                successor = tuple((stand_in.x, stand_in.y) for stand_in in scratch)

                # The goal is *arriving* on the target square, which is exactly
                # what a successor state records -- a robot that merely slides
                # across the square does not stop there.
                if index in goal_indices and successor[index] == goal_square:
                    return depth + 1

                if successor in seen:
                    continue

                seen.add(successor)
                frontier.append((successor, depth + 1))

            # Leave the node as it was found for the next robot's trials.
            robot.x, robot.y = home

    return None
