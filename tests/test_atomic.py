"""The heartbeat's whole value is that an outside observer can trust it.

A reader that catches Path.write_text between truncate and write gets an empty
or half-written file, which is worse than no file at all -- it looks like data.

The raw-write_text version of the concurrent-reader test below was run before
this module existed and produced 80 JSONDecodeErrors over 200 writes, so the
race is measured rather than assumed.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jaano.atomic import atomic_write_text

# Big enough that the write cannot complete in one syscall-sized gulp, which is
# what makes the torn-read window wide enough to hit reliably. With a 200-byte
# payload the race exists but takes thousands of iterations to observe.
PAYLOAD_KEYS = 20_000
WRITES = 200


def _payload(i: int) -> str:
    return json.dumps({"n": i, "pad": {str(k): k for k in range(PAYLOAD_KEYS)}})


def test_concurrent_reader_never_sees_a_partial_document(tmp_path: Path) -> None:
    target = tmp_path / "status.json"
    atomic_write_text(target, _payload(0))

    torn: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                json.loads(target.read_text())
            except json.JSONDecodeError as exc:
                torn.append(str(exc))
            except FileNotFoundError:
                torn.append("file did not exist")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for i in range(1, WRITES + 1):
            atomic_write_text(target, _payload(i))
    finally:
        stop.set()
        t.join(timeout=5)

    assert torn == []
    assert json.loads(target.read_text())["n"] == WRITES


def test_temp_file_is_a_sibling_so_the_rename_stays_on_one_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rename(2) requires both paths on the same file system. A /tmp temp file
    # with a target on the data volume is EXDEV on this machine.
    import os

    from jaano import atomic

    target = tmp_path / "status.json"
    seen: list[Path] = []
    real_replace = os.replace

    # Path, not the wider StrOrBytesPath os.replace accepts: the only caller
    # this stands in front of is atomic_write_text, which passes two Paths.
    def spy(src: Path, dst: Path) -> None:
        seen.append(Path(src))
        return real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", spy)
    atomic_write_text(target, "{}")

    assert seen, "os.replace was never called -- this is not an atomic write"
    assert seen[0].parent == target.parent


class Boom(Exception):
    pass


def _explode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail at the rename, the last possible moment -- so a temp file exists."""
    from jaano import atomic

    def boom(src: Path, dst: Path) -> None:
        raise Boom

    monkeypatch.setattr(atomic.os, "replace", boom)


def test_a_failed_write_leaves_no_temp_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "status.json"
    _explode(monkeypatch)

    with pytest.raises(Boom):
        atomic_write_text(target, "{}")

    assert list(tmp_path.iterdir()) == []


def test_the_previous_document_survives_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "status.json"
    atomic_write_text(target, '{"n": 1}')
    _explode(monkeypatch)

    with pytest.raises(Boom):
        atomic_write_text(target, '{"n": 2}')

    assert json.loads(target.read_text()) == {"n": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_fsync_is_off_by_default_and_reaches_the_file_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The asymmetry is deliberate, so it is asserted rather than assumed.

    The heartbeat is only protected against a concurrent reader and a dying
    process, both of which the rename alone covers; paying an fsync per ~3s
    heartbeat buys only power-loss protection for a file that costs nothing to
    regenerate. The checkpoint that jaano-fdy adds inverts that trade, so the
    same primitive has to be able to do both.
    """
    from jaano import atomic

    synced: list[int] = []
    # def, not lambda: strict flags every unannotated lambda parameter, and a
    # lambda has nowhere to put the annotation.
    def record_fsync(fd: int) -> None:
        synced.append(fd)

    monkeypatch.setattr(atomic.os, "fsync", record_fsync)

    atomic_write_text(tmp_path / "a.json", "{}")
    assert synced == []

    atomic_write_text(tmp_path / "b.json", "{}", fsync=True)
    assert len(synced) == 1


def test_the_file_is_written_as_utf8_whatever_the_locale_is(tmp_path: Path) -> None:
    """encoding="utf-8" explicitly, not the platform default.

    Two mutants survived -- `encoding=None` and the argument dropped -- because
    on this machine the locale default IS utf-8, so a round-trip through
    read_text() cannot tell the difference. Asserting on the raw BYTES can.

    A transcript carries whatever the speaker said; a checkpoint carries the
    tokens. Both are non-ASCII the moment a name or an accent appears, and a
    file written in a different codec is one another machine cannot read back.
    """
    p = tmp_path / "out.json"
    atomic_write_text(p, "café — ünïcode ✓")

    assert p.read_bytes() == "café — ünïcode ✓".encode()
