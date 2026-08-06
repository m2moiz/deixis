"""The resume gate, run by the suite instead of by whoever remembers.

A resume that silently restarts produces a byte-identical transcript. No
output comparison can see it; the only witnesses are the decoded chunk count
and the status file's `resumed_from_s`. That is what `scratch/resume_gate.py`
asserts, and until now it ran only when someone thought to run it -- which is
the same as not running.

Slow: several real ASR passes over a 360s clip. It belongs to `just verify`,
not to the inner loop.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# tests/ has no __init__.py, so pytest puts this directory on sys.path
# and the sibling import resolves without a package.
from gate_helpers import GateModule, load_gate

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"),
]


@pytest.fixture(scope="module")
def resume_gate() -> GateModule:
    """The gate module itself, imported from scratch/."""
    return load_gate("resume_gate")


def test_a_resume_consumes_its_checkpoint(
    resume_gate: GateModule, chunked_audio_path: Path, tmp_path: Path
) -> None:
    """The real gate, end to end: interrupt a run, resume it, prove it resumed.

    Five legs -- baseline, interrupted, resume, equivalence, stale rejection --
    kept as one test because they are one sequential story sharing a work dir,
    and splitting them would re-run the expensive baseline for each.
    """
    resume_gate.gate(
        chunked_audio_path, tmp_path / "work", False, self_test=False
    )


def test_the_resume_gate_can_actually_fail(
    resume_gate: GateModule, chunked_audio_path: Path, tmp_path: Path
) -> None:
    """Re-prove, every run, that the gate above is capable of going red.

    `--no-resume` on leg 3 makes the run start over. The gate MUST notice.

    `match=` is load-bearing and is the whole difference between this test and
    a decorative one. A bare `pytest.raises(GateFailure)` passes on ANY failure
    from anywhere in gate() -- a dead subprocess, a wait ceiling, a missing
    fixture, or the sibling assertion two lines below the one that matters. It
    would still be green with the chunk-count check deleted, which is precisely
    the check that catches a silently-restarting resume.

    Pinned to the message that check raises, so this test fails if the
    assertion it exists to protect is removed or reworded.
    """
    with pytest.raises(
        resume_gate.GateFailure,
        match="It started over",
    ):
        resume_gate.gate(
            chunked_audio_path,
            tmp_path / "selftest",
            False,
            self_test=True,
            skip_stale=True,
        )
