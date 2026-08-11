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
from typing import Any, cast

from parakeet_mlx.alignment import AlignedToken
from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError

from jaano.atomic import atomic_write_text

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


def _to_json(token: AlignedToken) -> dict[str, Any]:
    # `end` is omitted deliberately: AlignedToken.__post_init__ recomputes it
    # from start + duration, so persisting it would only add a way to disagree.
    return {
        "id": token.id,
        "text": token.text,
        "start": token.start,
        "duration": token.duration,
        "confidence": token.confidence,
    }


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


class _TokenDoc(BaseModel):
    """One token as it appears on disk.

    NOT strict. Strict mode rejects a JSON int where a float is declared, and
    AlignedToken is a plain dataclass with no coercion -- so a token whose
    duration or confidence happened to serialise as a bare integer would make
    read_checkpoint return None and silently re-transcribe a resumable run.
    "Resume just stopped working" with no error is the exact failure class this
    validation exists to close, so the numerics stay lax and the strictness
    goes where the real bug is: next_start, below.
    """

    id: int
    text: str
    start: float
    duration: float
    confidence: float


class _CheckpointDoc(BaseModel):
    """The checkpoint document, validated because a previous PROCESS wrote it.

    This is the one trust boundary in jaano that reads bytes it did not
    produce in this run. Everything else here is internal and belongs to the
    type checker.

    `next_start` is strict: it is an index into an audio buffer, and the string
    "44" survived the old `except (KeyError, TypeError)` net all the way to
    transcribe.py, where `skip_before / rate` raised TypeError. Loud, but at
    the wrong layer and only on the CLI's progress path -- a library caller
    with a different path could carry it further.

    `fingerprint` is `dict[str, Any]`: it is compared for exact equality
    against a freshly built one before this model is ever constructed, so
    validating its fields would restate the check that already happened.
    """

    model_config = ConfigDict(strict=False)

    fingerprint: dict[str, Any]
    next_start: StrictInt
    tokens: list[_TokenDoc]


def read_checkpoint(path: Path, fp: Fingerprint) -> tuple[int, list[AlignedToken]] | None:
    """Return `(next_start, tokens)` if the checkpoint matches, else None.

    A mismatch is not an error. A changed model or a re-encoded source just
    means the stored tokens describe something else; the caller starts over.
    """
    try:
        raw: object = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(raw, dict):
        return None
    # The guard above proves it is a dict but says nothing about key/value
    # types; JSON object keys are always str, and the values stay Any because
    # the shape check happens below, in the try.
    payload = cast("dict[str, Any]", raw)

    if payload.get("fingerprint") != dataclasses.asdict(fp):
        return None

    try:
        doc = _CheckpointDoc.model_validate(payload)
    except ValidationError:
        # Well-formed JSON with the right fingerprint but the wrong shape means
        # something wrote this file that was not us. Do not guess.
        return None

    return doc.next_start, [
        AlignedToken(id=t.id, text=t.text, start=t.start, duration=t.duration,
                     confidence=t.confidence)
        for t in doc.tokens
    ]
