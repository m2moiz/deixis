"""Persist a partially merged transcript so an interrupted run can continue.

The file sits beside the output rather than inside the status heartbeat. The
heartbeat is advisory and optional (`--status`); this is load-bearing for
correctness and keyed to the required `--out`. They also want different
durability -- see atomic_write_text's fsync argument.
"""

from __future__ import annotations

__all__ = [
    "SCHEMA",
    "Fingerprint",
    "checkpoint_path_for",
    "fingerprint",
    "read_checkpoint",
    "write_checkpoint",
]

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from parakeet_mlx.alignment import AlignedToken

from deixis.atomic import atomic_write_text

SCHEMA = 1


@dataclass(frozen=True)
class Fingerprint:
    """Everything that, if it changed, makes the stored tokens wrong to reuse.

    Reusing a checkpoint across any of these produces a transcript that is
    wrong without looking wrong, so the check is exact equality of the whole
    record rather than a heuristic on a few fields.
    """

    schema: int
    media: str
    media_size: int
    media_mtime_ns: int
    total_samples: int
    model_id: str
    parakeet_version: str
    chunk_s: float
    overlap_s: float


def fingerprint(
    media: Path,
    total_samples: int,
    model_id: str,
    chunk_s: float,
    overlap_s: float,
) -> Fingerprint:
    """Describe the run precisely enough that a stale checkpoint cannot match.

    Keyed on the SOURCE media -- the file the user handed us -- and never on
    the audio actually fed to the model. A .mov is extracted to a fresh temp
    wav on every run, with a new path and a new mtime each time, so keying on
    that would mean resume never matches for exactly the normal input, and
    would do it silently: every test on an already-conforming wav would still
    pass.

    `total_samples` is included even though size and mtime already cover most
    edits: it is the only field that catches an ffmpeg upgrade decoding the
    same untouched file to a different length, which would silently shift every
    chunk boundary.
    """
    from importlib.metadata import version

    st = media.stat()
    return Fingerprint(
        schema=SCHEMA,
        media=str(media.resolve()),
        media_size=st.st_size,
        media_mtime_ns=st.st_mtime_ns,
        total_samples=total_samples,
        model_id=model_id,
        # The merge functions live upstream. If they change, tokens merged by
        # the old ones cannot be extended by the new ones.
        parakeet_version=version("parakeet-mlx"),
        chunk_s=chunk_s,
        overlap_s=overlap_s,
    )


def checkpoint_path_for(out: Path) -> Path:
    """Return the checkpoint path that sits beside `out`.

    A sibling of the output rather than a temp-dir entry: a resume has to find
    it on a later run, in a later process, with no shared state but the path.
    """
    return out.with_name(out.name + ".ckpt")


def _to_json(token: AlignedToken) -> dict:
    # `end` is omitted deliberately: AlignedToken.__post_init__ recomputes it
    # from start + duration, so persisting it would only add a way to disagree.
    return {
        "id": token.id,
        "text": token.text,
        "start": token.start,
        "duration": token.duration,
        "confidence": token.confidence,
    }


def _from_json(d: dict) -> AlignedToken:
    return AlignedToken(
        id=d["id"],
        text=d["text"],
        start=d["start"],
        duration=d["duration"],
        confidence=d["confidence"],
    )


def write_checkpoint(
    path: Path, fp: Fingerprint, next_start: int, tokens: list[AlignedToken]
) -> None:
    """Record the merged tokens and the sample index to resume from.

    The whole token list is rewritten each time rather than appended to. That
    is quadratic in chunk count, but a 74-minute file is ~43 chunks and a few
    megabytes; an append log would buy nothing and cost a recovery path.
    """
    payload = {
        "fingerprint": dataclasses.asdict(fp),
        "next_start": next_start,
        "tokens": [_to_json(t) for t in tokens],
    }
    # fsync here, unlike the heartbeat: losing this costs minutes of GPU time,
    # and one fsync per ~105 seconds of audio is free.
    atomic_write_text(path, json.dumps(payload), fsync=True)


def read_checkpoint(path: Path, fp: Fingerprint) -> tuple[int, list[AlignedToken]] | None:
    """Return `(next_start, tokens)` if the checkpoint matches, else None.

    A mismatch is not an error. A changed model or a re-encoded source just
    means the stored tokens describe something else; the caller starts over.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("fingerprint") != dataclasses.asdict(fp):
        return None

    try:
        return payload["next_start"], [_from_json(d) for d in payload["tokens"]]
    except (KeyError, TypeError):
        # Well-formed JSON with the right fingerprint but the wrong shape means
        # something wrote this file that was not us. Do not guess.
        return None
