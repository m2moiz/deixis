# The fast inner loop. Same thing plain `uv run pytest` gives you -- the
# deselection lives in pyproject addopts, not here, so the two cannot drift.
test:
    uv run pytest

# Two pyright invocations, not one: pyproject holds exactly one [tool.pyright]
# table, and the package and the tests need separate configs. Per-directory
# typeCheckingMode via executionEnvironments was measured and does not work --
# the total went UP rather than splitting.
typecheck:
    uv run pyright --project pyrightconfig.json
    uv run pyright --project pyrightconfig.tests.json

# THE GATE. Lint, types and the fast suite, in the order that fails cheapest
# first. One command for a human, for a coding agent, and for CI -- so a green
# here means the same thing to all three, and local and CI cannot drift.
#
# NOT pre-commit. `pre-commit install` refuses outright in this repo:
# core.hooksPath is set to .beads/hooks by `bd init`, and globally to
# ~/.githooks. Its own remedy -- `git config --unset-all core.hooksPath` --
# would disable the beads hooks AND every global hook on this machine, a blast
# radius outside this project. Worse, a pre-commit config would have LOOKED
# installed: the beads hook still runs and still exits 0, so the phase would
# have been recorded as landed while doing nothing. That is exactly the
# covered-but-unasserted failure this whole effort exists to remove.
#
# No formatter, deliberately. This codebase has a hand-set style -- aligned
# argv lists, comment-dense blocks -- and running one is a whole-tree diff
# nobody asked for.
check: typecheck
    uv run ruff check .
    uv run pytest

# Everything, with coverage. The session-close gate, and the named observer
# that makes deselecting the slow tests honest rather than a quiet loss.
# `-m "slow or not slow"` rather than `-m ""`: it unambiguously overrides the
# deselection in addopts, and empty-marker behaviour is not worth relying on.
verify: typecheck
    uv run ruff check .
    uv run pytest -m "slow or not slow" --cov=deixis --cov-report=term-missing:skip-covered
