"""A checkpoint written before the extraction must resume after it.

The Android port replaces the imported token type with dsj's own
(docs/android-port-design.md). That swap is only safe if it is byte-invisible
to a checkpoint already on disk: a mismatch is not an error, it is a silent
re-transcription of an hour of audio, so no test failure would ever point at
it. This file is the pointing.

`tests/golden/checkpoint-v1.json.ckpt` was written by `write_checkpoint` at
commit eb9ea14, while `AlignedToken` was still parakeet's class -- the real
pre-extraction bytes, committed as a literal file. It must never be
regenerated: a fixture rebuilt by new code only proves the new code agrees
with itself, which is exactly the non-proof this test exists to rule out.

The expected values below are duplicated from the fixture BY HAND, on
purpose. Deriving them from the file at test time would let one bug in the
reader corrupt both sides of the comparison.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from dsj.checkpoint import Fingerprint, read_checkpoint

GOLDEN = Path(__file__).parent / "golden" / "checkpoint-v1.json.ckpt"

# The fingerprint the writer recorded, spelled out field by field. If
# Fingerprint's shape changes such that this no longer constructs, that is the
# test working: the change has to come here and answer for old checkpoints.
GOLDEN_FP = Fingerprint(
    schema=1,
    media="/recordings/golden.m4a",
    media_size=48213977,
    media_mtime_ns=1755087412000000000,
    total_samples=71424000,
    model_id="mlx-community/parakeet-tdt-0.6b-v3",
    parakeet_version="0.5.2",
    chunk_s=120.0,
    overlap_s=15.0,
)


def test_golden_checkpoint_resumes() -> None:
    resumed = read_checkpoint(GOLDEN, GOLDEN_FP)
    assert resumed is not None, "the pre-extraction checkpoint stopped matching"
    next_start, tokens = resumed

    assert next_start == 1680000
    assert len(tokens) == 4

    first, last = tokens[0], tokens[-1]
    assert (first.id, first.text, first.start, first.duration, first.confidence) == (
        0,
        " the",
        0.08,
        0.24,
        1.0,
    )
    assert (last.id, last.text, last.start, last.duration, last.confidence) == (
        1204,
        " fox",
        119.96,
        0.301,
        1.0,
    )
    # `end` is not in the JSON; __post_init__ must recompute it. 119.96 + 0.301
    # is 120.261 only in floating point -- the literal pins the recompute to
    # the same arithmetic, not merely the same idea.
    assert last.end == 120.261
    assert first.end == first.start + first.duration

    # The third token carries the non-default confidence.
    assert tokens[2].confidence == 0.5


def test_golden_checkpoint_bytes_are_what_the_writer_wrote() -> None:
    """Guard the fixture itself against a well-meaning regeneration.

    If someone re-runs the writer and commits the result, values drift
    invisibly (a new parakeet version, a reordered dict) and the golden test
    starts proving self-agreement. Pinning the raw JSON keys and the exact
    fingerprint dict makes that a loud diff instead.
    """
    payload = json.loads(GOLDEN.read_text())
    assert set(payload) == {"fingerprint", "next_start", "tokens"}
    assert payload["fingerprint"] == dataclasses.asdict(GOLDEN_FP)
    # `end` must be absent from every stored token -- it is derived state, and
    # persisting it would add a way for the file to disagree with the class.
    for doc in payload["tokens"]:
        assert set(doc) == {"id", "text", "start", "duration", "confidence"}


def test_golden_checkpoint_rejects_a_different_run() -> None:
    """The same file must NOT resume under any other fingerprint.

    Guards against the reader ever weakening to a partial match -- the
    docstring on Fingerprint promises exact equality of the whole record.
    """
    other = dataclasses.replace(GOLDEN_FP, model_id="mlx-community/other-model")
    assert read_checkpoint(GOLDEN, other) is None
