"""Small containers shared across the model package.

This module exists to break an import cycle. `ActionHistory` lived in `mcts.py`,
but `state.py` needs it too, and:

    state -> mcts -> muzero -> state

closes a loop. It was evaded by importing `ActionHistory` *inside*
`RicochetRobotsGame.action_history()` at call time, with nothing marking that
line as load-bearing -- moving that import to the top of the file, the obvious
tidy-up, turned it into an `ImportError` at collection.

`ActionHistory` is a plain container with no dependency on anything in the
package, so hoisting it into a leaf module lets both importers reach it at
module scope and the cycle disappears rather than being worked around.

Anything added here must stay dependency-free, or the cycle comes back.
"""

from typing import List


class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[int], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: int):
        self.history.append(action)

    def last_action(self) -> int:
        return self.history[-1]

    def action_space(self) -> List[int]:
        return [int(i) for i in range(self.action_space_size)]
