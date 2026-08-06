"""The diarization gate, run by the suite instead of by whoever remembers.

A run that skipped diarization still finishes, still exits 0, and still writes
a perfectly good transcript. The pass has to be caught WHILE IT RUNS -- the
status heartbeat reporting "diarizing" is the only witness. That is what
`scratch/diarize_gate.py` asserts.

Slow, and gated on the optional extra: it loads senko (~12s of CoreML model
load) in a subprocess and diarizes the full reference recording.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

# tests/ has no __init__.py, so pytest puts this directory on sys.path and the
# sibling import resolves without a package.
from gate_helpers import REPO, GateModule, load_gate

SOURCE = REPO / "scratch" / "meeting.wav"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"),
    # find_spec, not import: this process must not pay senko's CoreML load. The
    # SUBPROCESS the gate launches is the thing that diarizes.
    pytest.mark.skipif(
        importlib.util.find_spec("senko") is None,
        reason="diarize extra not installed (uv sync --extra diarize)",
    ),
    pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE} not present"),
]


@pytest.fixture(scope="module")
def diarize_gate() -> GateModule:
    """The gate module itself, imported from scratch/."""
    return load_gate("diarize_gate")


def test_the_diarization_pass_is_observed_running(
    diarize_gate: GateModule, tmp_path: Path
) -> None:
    """The real gate: watch the heartbeat, then compare against a plain run.

    probe=False drops leg 4, the straddle report. It prints numbers for a human
    to read and asserts nothing, so it is a diagnostic rather than a gate and
    has no business costing another full pass inside the suite.
    """
    diarize_gate.gate(
        SOURCE, tmp_path / "work", self_test=False, probe=False
    )


def test_the_diarize_gate_can_actually_fail(
    diarize_gate: GateModule, tmp_path: Path
) -> None:
    """Re-prove, every run, that the gate above is capable of going red.

    `--no-diarize` makes leg 1 skip the pass entirely. The gate MUST notice.

    `match=` matters more here than anywhere else in the suite. With the pass
    disabled, TWO assertions fire: the heartbeat check and, later, the
    "carries no provenance key" check. A bare pytest.raises would be satisfied
    by the second one -- so the heartbeat assertion, which the gate's own
    comment calls "THE assertion", could be deleted and this test would stay
    green. Pinned to the heartbeat message so it cannot.
    """
    with pytest.raises(
        diarize_gate.GateFailure,
        match="the heartbeat never reported 'diarizing'",
    ):
        diarize_gate.gate(
            SOURCE, tmp_path / "selftest", self_test=True, probe=False
        )
