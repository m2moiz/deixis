"""Import the end-to-end gates from scratch/ so the suite can drive them.

The gates are NOT reimplemented here. `scratch/resume_gate.py` and
`scratch/diarize_gate.py` are tracked code, each proven on the real reference
recording, and each carries a `--self-test` that demonstrates it can fail.
Copying their process machinery into tests/ would create a second
implementation that drifts from the one an operator runs by hand -- and the
whole point of these gates is that the thing running is the thing that was
proven.

So: one implementation, two front ends. The scripts stay runnable standalone;
the tests import `gate()` and call it directly.

`scratch/` is not a package and has no `__init__.py`, so the import goes
through the path rather than through a name. That is the cost of not
duplicating 30 KB of subprocess machinery, and it is the cheaper side of the
trade.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

REPO = Path(__file__).resolve().parent.parent
SCRATCH = REPO / "scratch"


class GateModule(Protocol):
    """The surface a gate script exposes to the suite.

    Two names, which is the whole contract: the callable that runs the gate and
    the exception it raises. Writing it down is the same discipline as the
    parakeet Protocol in deixis/chunking.py -- a dynamically loaded module is
    otherwise Unknown to the type checker, and `Any` would hide a renamed
    `gate()` until the slow lane ran.
    """

    GateFailure: type[AssertionError]
    gate: Callable[..., None]


def load_gate(name: str) -> GateModule:
    """Import `scratch/<name>.py` as a module.

    Args:
        name: file stem under scratch/, e.g. "resume_gate".

    Returns:
        The imported module, exposing at least `gate` and `GateFailure`.

    Raises:
        ImportError: if the file is absent or cannot be loaded. Deliberately
            loud: a missing gate script is not a reason to skip quietly, it is
            a reason to fail. The fixtures the gates need may legitimately be
            absent on another machine -- the gates themselves may not.
    """
    path = SCRATCH / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the gate at {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the module can be pickled by anything that
    # needs to, and so a second load returns the same object.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return cast("GateModule", module)
