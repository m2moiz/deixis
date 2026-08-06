# The fast inner loop. Same thing plain `uv run pytest` gives you -- the
# deselection lives in pyproject addopts, not here, so the two cannot drift.
test:
    uv run pytest

# Everything, with coverage. THE session-close gate, and the named observer
# that makes deselecting the slow tests honest rather than a quiet loss.
# `-m "slow or not slow"` rather than `-m ""`: it unambiguously overrides the
# deselection in addopts, and empty-marker behaviour is not worth relying on.
verify:
    uv run pytest -m "slow or not slow" --cov=deixis --cov-report=term-missing:skip-covered
