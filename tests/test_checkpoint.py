"""A checkpoint that survives a change it should not survive is worse than none.

Reusing tokens decoded from different audio, a different model, or a different
chunk geometry produces a transcript that is silently wrong -- and silently
wrong is the failure mode this whole feature exists to avoid.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path

import pytest
from parakeet_mlx.alignment import AlignedToken

from deixis.checkpoint import (
    SCHEMA,
    Fingerprint,
    checkpoint_path_for,
    fingerprint,
    read_checkpoint,
    write_checkpoint,
)

FP = Fingerprint(
    schema=1,
    media="/x/meeting.mov",
    media_size=141664974,
    media_mtime_ns=1_700_000_000_000_000_000,
    total_samples=70_832_448,
    model_id="mlx-community/parakeet-tdt-0.6b-v3",
    parakeet_version="0.5.2",
    chunk_s=120.0,
    overlap_s=15.0,
)

TOKENS = [
    AlignedToken(id=5, text=" hello", start=1.25, duration=0.32, confidence=0.91),
    AlignedToken(id=9, text=" world", start=1.57, duration=0.48, confidence=0.87),
]


def test_checkpoint_path_sits_beside_the_output(tmp_path: Path) -> None:
    assert checkpoint_path_for(tmp_path / "meeting.json") == tmp_path / "meeting.json.ckpt"


def test_round_trip_preserves_tokens_exactly(tmp_path: Path) -> None:
    p = tmp_path / "out.json.ckpt"
    write_checkpoint(p, FP, next_start=1_680_000, tokens=TOKENS)

    got = read_checkpoint(p, FP)
    assert got is not None
    next_start, tokens = got
    assert next_start == 1_680_000
    assert [dataclasses.asdict(t) for t in tokens] == [
        dataclasses.asdict(t) for t in TOKENS
    ]


def test_end_is_recomputed_not_stored(tmp_path: Path) -> None:
    # AlignedToken.__post_init__ derives end from start + duration. Storing it
    # would only create a second source of truth that could disagree.
    p = tmp_path / "out.json.ckpt"
    write_checkpoint(p, FP, next_start=0, tokens=TOKENS)
    raw = json.loads(p.read_text())
    assert "end" not in raw["tokens"][0]

    got = read_checkpoint(p, FP)
    assert got is not None
    assert got[1][0].end == TOKENS[0].end


def test_missing_checkpoint_is_not_an_error(tmp_path: Path) -> None:
    assert read_checkpoint(tmp_path / "absent.ckpt", FP) is None


def test_a_truncated_checkpoint_is_discarded(tmp_path: Path) -> None:
    p = tmp_path / "out.json.ckpt"
    p.write_text('{"fingerprint": {"schema": 1,')
    assert read_checkpoint(p, FP) is None


def test_well_formed_json_of_the_wrong_shape_is_discarded(tmp_path: Path) -> None:
    p = tmp_path / "out.json.ckpt"
    p.write_text(json.dumps({"fingerprint": dataclasses.asdict(FP), "tokens": "nope"}))
    assert read_checkpoint(p, FP) is None


def test_every_fingerprint_field_invalidates(tmp_path: Path) -> None:
    p = tmp_path / "out.json.ckpt"
    write_checkpoint(p, FP, next_start=1_680_000, tokens=TOKENS)

    changed = {
        "schema": 2,
        "media": "/x/other.mov",
        "media_size": 1,
        "media_mtime_ns": 1,
        "total_samples": 1,
        "model_id": "mlx-community/parakeet-tdt-0.6b-v2",
        "parakeet_version": "0.6.0",
        "chunk_s": 60.0,
        "overlap_s": 5.0,
    }
    assert set(changed) == {f.name for f in dataclasses.fields(Fingerprint)}, (
        "a field was added to Fingerprint without a case here"
    )

    for field, value in changed.items():
        other = dataclasses.replace(FP, **{field: value})
        assert read_checkpoint(p, other) is None, f"{field} did not invalidate"


def test_float_values_survive_the_round_trip_bit_for_bit(tmp_path: Path) -> None:
    p = tmp_path / "out.json.ckpt"
    awkward = [
        AlignedToken(
            id=1, text="a", start=0.1 + 0.2, duration=1 / 3,
            confidence=0.9999999999999999,
        ),
        AlignedToken(id=2, text="b", start=4426.987654321, duration=1e-10, confidence=1.0),
    ]
    write_checkpoint(p, FP, next_start=0, tokens=awkward)
    got = read_checkpoint(p, FP)
    assert got is not None
    for a, b in zip(awkward, got[1], strict=True):
        assert struct.pack("<d", a.start) == struct.pack("<d", b.start)
        assert struct.pack("<d", a.duration) == struct.pack("<d", b.duration)
        assert struct.pack("<d", a.confidence) == struct.pack("<d", b.confidence)


def test_the_fingerprint_describes_the_source_media_not_a_temp_wav(tmp_path: Path) -> None:
    """The trap this keying exists to avoid.

    A .mov is extracted to a fresh temp wav on every run, with a new path and a
    new mtime each time. A fingerprint taken from that wav could never match on
    a second run, so resume would silently never fire for exactly the input the
    tool is built for -- while every test on a wav input stayed green.
    """
    source = tmp_path / "recording.mov"
    source.write_bytes(b"pretend this is a screen recording")

    fp = fingerprint(source, total_samples=70_832_448, model_id="m",
                     chunk_s=120.0, overlap_s=15.0)

    assert fp.media == str(source.resolve())
    assert fp.media_size == source.stat().st_size
    assert fp.media_mtime_ns == source.stat().st_mtime_ns

    # Taken twice, with a different extraction in between, it is the same.
    assert fingerprint(source, 70_832_448, "m", 120.0, 15.0) == fp


def test_a_re_encoded_source_invalidates(tmp_path: Path) -> None:
    source = tmp_path / "recording.mov"
    source.write_bytes(b"first cut")
    before = fingerprint(source, 100, "m", 120.0, 15.0)

    p = tmp_path / "out.json.ckpt"
    write_checkpoint(p, before, next_start=0, tokens=TOKENS)

    source.write_bytes(b"a different, longer second cut")
    after = fingerprint(source, 100, "m", 120.0, 15.0)

    assert read_checkpoint(p, after) is None


# --- the validated boundary -------------------------------------------------
#
# read_checkpoint parses bytes a PREVIOUS PROCESS wrote. It is the one place in
# deixis that reads a document it did not produce in this run, and until the
# checkpoint was validated it returned whatever the JSON happened to contain.


def _write_raw(path: Path, **overrides: object) -> None:
    """Write a checkpoint straight to disk, bypassing write_checkpoint.

    write_checkpoint cannot produce the shapes below -- that is the point. They
    come from a different writer, an older format, or a partially hand-edited
    file, which is exactly what the validation is for.
    """
    payload: dict[str, object] = {
        "fingerprint": dataclasses.asdict(FP),
        "next_start": 44,
        "tokens": [
            {"id": 5, "text": " hello", "start": 1.25, "duration": 0.32, "confidence": 0.91}
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def test_a_string_next_start_is_refused(tmp_path: Path) -> None:
    """The bug this validation exists for.

    Before the model, `payload["next_start"]` was returned unconverted and the
    `except (KeyError, TypeError)` net never fired -- indexing a dict whose
    value is a string raises nothing. read_checkpoint returned ('44', []), and
    the string travelled on to transcribe.py, where `skip_before / rate` raised
    TypeError several frames from the file that produced it.
    """
    p = tmp_path / "out.json.ckpt"
    _write_raw(p, next_start="44")
    assert read_checkpoint(p, FP) is None


def test_a_float_next_start_is_refused(tmp_path: Path) -> None:
    """44.0 is not a sample index, however close it looks to one."""
    p = tmp_path / "out.json.ckpt"
    _write_raw(p, next_start=44.0)
    assert read_checkpoint(p, FP) is None


def test_a_token_missing_a_field_is_refused(tmp_path: Path) -> None:
    """A half-written token means the whole document is untrustworthy."""
    p = tmp_path / "out.json.ckpt"
    _write_raw(p, tokens=[{"id": 5, "text": " hi", "start": 1.25, "duration": 0.32}])
    assert read_checkpoint(p, FP) is None


def test_integer_valued_token_numbers_are_accepted(tmp_path: Path) -> None:
    """Deliberately lax where strictness would DESTROY a resumable run.

    A token whose start/duration/confidence happens to serialise as a bare JSON
    integer is valid data. Under a strict model those are rejected, and a
    rejected checkpoint is a silent full re-transcription presenting as "resume
    just stopped working" -- the exact failure class this validation exists to
    close, reintroduced by the validation itself.

    So the numerics coerce and only next_start is strict. This test is the
    reason that asymmetry is not an oversight.
    """
    p = tmp_path / "out.json.ckpt"
    _write_raw(p, tokens=[{"id": 5, "text": " hi", "start": 0, "duration": 1, "confidence": 1}])

    got = read_checkpoint(p, FP)
    assert got is not None
    next_start, tokens = got
    assert next_start == 44
    assert tokens[0].start == 0.0
    assert tokens[0].duration == 1.0


# --- gaps mutation testing found ---------------------------------------------


def test_the_fingerprint_records_every_field_it_claims_to(tmp_path: Path) -> None:
    """Each field is actually populated, not merely present.

    Six mutants -- schema, total_samples, model_id, parakeet_version, chunk_s,
    overlap_s each replaced by None -- survived. `test_every_fingerprint_field
    _invalidates` could not see them: it compares two fingerprints built by the
    SAME function, so a field nulled on both sides still differs wherever it
    differed before. Only an absolute assertion catches a field that stopped
    being read.
    """
    source = tmp_path / "recording.mov"
    source.write_bytes(b"x")

    fp = fingerprint(source, total_samples=70_832_448, model_id="m",
                     chunk_s=120.0, overlap_s=15.0)

    assert fp.schema == SCHEMA
    assert fp.total_samples == 70_832_448
    assert fp.model_id == "m"
    assert fp.chunk_s == 120.0
    assert fp.overlap_s == 15.0
    # Not pinned to a literal: it moves when the dependency does, and the point
    # is that SOMETHING was read, not which release is installed.
    assert fp.parakeet_version and fp.parakeet_version != "None"


def test_the_checkpoint_is_written_with_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fsync=True at the call site, not just available on the writer.

    Three mutants -- fsync=False, fsync=None, and the argument dropped entirely
    -- survived, and no test could have caught them: whether data reached the
    platter is invisible to a process that then reads its own page cache.
    Asserting on the CALL is the only cheap witness.

    It is worth having. Losing a checkpoint to a power cut costs minutes of GPU
    time, which is the entire reason the argument is there.
    """
    seen: list[bool] = []

    def spy(path: Path, text: str, *, fsync: bool = False) -> None:
        seen.append(fsync)
        path.write_text(text)

    monkeypatch.setattr("deixis.checkpoint.atomic_write_text", spy)
    write_checkpoint(tmp_path / "out.json.ckpt", FP, next_start=44, tokens=TOKENS)

    assert seen == [True]
