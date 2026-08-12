"""The one shape every ASR engine in jaano returns.

parakeet and whisper agree on almost nothing. parakeet-mlx hands back an
`AlignedResult` of `AlignedToken`s and is driven chunk by chunk from
jaano/chunking.py so a run can resume; whisper owns its own 30-second window
loop and returns segments carrying words. Naming what they must both produce
keeps that difference inside the two backends instead of spreading it through
suno.py.

The contract is the transcript's own on-disk shape rather than a parallel
object model, because json.dumps is the only consumer and a schema class here
would be flattened straight back into dicts one function later.

Per-token times are the load-bearing half. merge.py labels a sentence by voting
its TOKENS against the diarizer's turns, so an engine that can only give
sentence boundaries can be transcribed but not attributed.
"""

from __future__ import annotations

__all__ = ["ENGINES", "Transcription"]

from typing import Any, NamedTuple

# The order is not alphabetical: parakeet is first because it is the default,
# and it is the default because it is ~10x faster and does not hallucinate over
# silence. whisper is what you reach for when the language is not in
# parakeet's 25.
ENGINES = ("parakeet", "whisper")


class Transcription(NamedTuple):
    """What an engine hands back to `suno.transcribe`.

    `sentences` are already in the payload's shape -- `{start, end, text,
    tokens: [{t, w}]}` -- so the caller writes them out rather than converting
    them.
    """

    text: str
    sentences: list[dict[str, Any]]
