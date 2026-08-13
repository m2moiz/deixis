"""No core module may import a backend. The fact, checked, not claimed.

The Android port rests on one line: everything outside the engine leaf
modules runs on any machine with Python, numpy and ffmpeg. Nothing else in
the suite can observe that on a Mac, where every backend happens to import --
the failure only exists on the machine that does not have them, which is
exactly the machine that runs no tests. So the line is enforced statically.

AST, not grep: suno.py and friends legitimately MENTION backends in prose --
docstrings argue why parakeet is the default -- and a comment must never fail
this. Only an actual import statement counts, including one inside a function
or under TYPE_CHECKING: function-local backend imports belong in engine
modules, and even a type-only core import would force the backend onto any
environment that typechecks core.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "dsj"

# The leaf modules allowed to import their one backend. diarize.py is an
# engine in this sense -- senko is its backend -- even though it is not an ASR
# engine behind dsj.asr.
ENGINE_MODULES = {"parakeet.py", "whisper.py", "diarize.py"}

# Backend roots, present and future. Listing sherpa_onnx before sherpa.py
# exists is deliberate: when that engine lands, this test already knows the
# import belongs only there.
BACKENDS = {
    "parakeet_mlx",
    "mlx_whisper",
    "mlx",
    "senko",
    "coremltools",
    "sherpa_onnx",
}


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_modules_import_no_backend() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name in ENGINE_MODULES:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        hit = _imported_roots(tree) & BACKENDS
        if hit:
            offenders.append(f"{path.name} imports {sorted(hit)}")
    assert not offenders, (
        "core modules must stay portable; move these imports into an engine "
        f"module: {offenders}"
    )


def test_engine_modules_import_only_their_own_backend() -> None:
    """parakeet.py must not grow an mlx-whisper dependency, and so on.

    One backend per leaf is what keeps 'install the parakeet extra' meaning
    the parakeet engine works, rather than an undocumented subset of it.
    """
    own = {
        "parakeet.py": {"parakeet_mlx", "mlx"},
        "whisper.py": {"mlx_whisper", "mlx"},
        "diarize.py": {"senko", "coremltools"},
    }
    for name, allowed in own.items():
        tree = ast.parse((PACKAGE / name).read_text(), filename=name)
        foreign = (_imported_roots(tree) & BACKENDS) - allowed
        assert not foreign, f"{name} imports another engine's backend: {sorted(foreign)}"
