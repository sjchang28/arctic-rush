"""Structural tests over the import graph.

These do not exercise behaviour; they pin the shape of the codebase. Both
properties they check were violated before, silently, and neither is the kind of
thing code review reliably catches:

  * `state -> mcts -> muzero -> state` was a real cycle, hidden by a
    function-local import. Nothing failed until someone moved that import to the
    top of the file, where it belongs.

  * `core` imported from `model`, and `game` imported 20 names from `model`,
    which left no ordering in which the packages could be layered.

The graph is built from the AST rather than by importing anything, so a
violation is reported as a readable list instead of an `ImportError` traceback.
"""

import ast
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")


def _module_name(path):
    rel = os.path.relpath(path, _REPO_ROOT)
    parts = os.path.splitext(rel)[0].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _source_files():
    for dirpath, dirnames, filenames in os.walk(_SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _imports(path, module_scope_only=True):
    """Map of `src.*` module -> set of names taken from it.

    `module_scope_only` restricts the walk to top-level statements, which is what
    determines whether a cycle actually bites at import time. A function-local
    import does not participate in the cycle -- that is exactly why one was used
    to hide this one.

    Names are tracked, not just modules, so an allowlist can be written per
    constant. Allowing a whole module would let the inversion return one name at
    a time without failing anything.
    """

    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    nodes = tree.body if module_scope_only else ast.walk(tree)

    found = {}
    for node in nodes:
        # `if TYPE_CHECKING:` and `if __name__ == ...` blocks still count as
        # module scope for cycle purposes, so descend into them.
        candidates = [node]
        if isinstance(node, ast.If):
            candidates = list(ast.walk(node))

        for item in candidates:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    if alias.name.startswith("src."):
                        found.setdefault(alias.name, set())
            elif isinstance(item, ast.ImportFrom):
                if item.module and item.module.startswith("src.") and item.level == 0:
                    names = found.setdefault(item.module, set())
                    names.update(alias.name for alias in item.names)

    return found


def _graph():
    """module -> {imported module: {names}}."""

    return {_module_name(path): _imports(path) for path in _source_files()}


def _find_cycle(graph):
    """Return one import cycle as a list of module names, or None."""

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    stack = []

    def visit(node):
        colour[node] = GREY
        stack.append(node)

        for neighbour in sorted(graph.get(node, ())):
            if neighbour not in graph:
                continue
            if colour[neighbour] == GREY:
                return stack[stack.index(neighbour):] + [neighbour]
            if colour[neighbour] == WHITE:
                cycle = visit(neighbour)
                if cycle:
                    return cycle

        stack.pop()
        colour[node] = BLACK
        return None

    for node in sorted(graph):
        if colour[node] == WHITE:
            cycle = visit(node)
            if cycle:
                return cycle

    return None


def test_no_import_cycles():
    """A cycle here is only survivable by deferring an import inside a function.

    That workaround is invisible at the call site and turns a routine tidy-up
    into an ImportError, so the cycle is banned rather than documented.
    """

    cycle = _find_cycle(_graph())

    assert cycle is None, "import cycle: " + " -> ".join(cycle)


def test_core_does_not_depend_on_game_or_model():
    """`core` is the bottom layer: logging, boot banner, log constants.

    `boot.py` imported seven names from `src.model.config`, which made the
    lowest layer depend on the highest and meant importing the banner pulled in
    the learner's configuration.
    """

    offenders = {
        module: sorted(bad)
        for module, deps in _graph().items()
        if module.startswith("src.core.")
        for bad in [{d for d in deps if d.startswith(("src.game", "src.model"))}]
        if bad
    }

    assert not offenders, f"core must not import game/model: {offenders}"


def test_game_does_not_depend_on_model():
    """`game` is the environment; `model` is the learner that consumes it.

    The dependency runs learner -> environment. `env.py` pointed it the other
    way for 20 names, so the environment could not be used, or tested, without
    the learner's configuration.

    `src.model.config` remains permitted for the reward and observation-shape
    constants: those genuinely describe the learner's contract with the
    environment, and `src/model/config.py` documents that choice. The curriculum
    and solver constants, which describe the environment itself, were moved to
    `src/game/config.py`.
    """

    # Allowed per *name*, not per module: permitting `src.model.config` wholesale
    # is how twenty names accumulated there in the first place.
    allowed = {
        "src.model.config": {
            "AI_ACTION_SPACE_SIZE",
            "AI_OBSERVATION_SHAPE",
            "MAX_TOTAL_MOVES_PER_GAME",
            "REWARD_PER_MOVE",
            "REWARD_REPEAT_STATE",
            "REWARD_SOLVE",
        },
    }

    offenders = {}
    for module, deps in _graph().items():
        if not module.startswith("src.game."):
            continue
        for dep, names in deps.items():
            if not dep.startswith("src.model"):
                continue
            extra = names - allowed.get(dep, set())
            if dep not in allowed or extra:
                offenders.setdefault(module, []).extend(
                    f"{dep}.{name}" for name in sorted(extra or names)
                )

    assert not offenders, f"game must not import these from model: {offenders}"


@pytest.mark.parametrize("leaf", ["src.model.types", "src.model.support", "src.game.config",
                                  "src.core.config"])
def test_leaf_modules_stay_leaves(leaf):
    """These four are what everything else is allowed to depend on.

    `types.py` in particular only breaks the `state`/`mcts` cycle for as long as
    it imports nothing from the package.
    """

    deps = _graph()[leaf]

    assert not deps, f"{leaf} must not import from src: {sorted(deps)}"
