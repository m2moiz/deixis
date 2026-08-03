"""Replace a file's contents without ever exposing a partial one.

Path.write_text truncates and then writes, so a reader that arrives in that
window gets an empty or half-written file. For the transcription heartbeat that
is not a cosmetic problem: the file exists to be trusted by an observer who
cannot see the job, and garbage that parses as "no progress" is worse than a
missing file, which at least announces itself.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, fsync: bool = False) -> None:
    """Make `path` contain `text`, atomically from a reader's point of view.

    A concurrent reader always resolves either the previous complete document
    or the new one.

    `fsync` additionally forces the bytes out of the page cache before the
    rename, which only matters across a power loss -- a process that merely
    dies leaves the page cache intact. Pass it for files whose loss costs real
    work, not for a progress heartbeat.
    """
    # rename(2) requires both paths on the same file system, so the temp file
    # is a sibling rather than something under /tmp -- on this machine those
    # are different volumes and the rename would fail with EXDEV.
    #
    # The pid is in the name so two processes writing the same status path
    # cannot hand each other a half-built temp file to rename into place.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write is
        # exactly the case that would otherwise strand a temp file next to the
        # output, where the next run would find it and wonder.
        tmp.unlink(missing_ok=True)
        raise
