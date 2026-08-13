"""The one shape every ASR engine in dsj returns.

parakeet and whisper agree on almost nothing. parakeet-mlx hands back an
`AlignedResult` of `AlignedToken`s and is driven chunk by chunk from
dsj/chunking.py so a run can resume; whisper owns its own 30-second window
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

__all__ = [
    "ENGINES",
    "ChunkEngine",
    "EngineSpec",
    "EngineUnavailable",
    "Transcription",
    "get_engine",
]

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

if TYPE_CHECKING:
    from dsj.alignment import AlignedToken

# The order is not alphabetical: parakeet is first because it is the default,
# and it is the default because it is ~10x faster and does not hallucinate over
# silence. whisper is what you reach for when the language is not in
# parakeet's 25.
ENGINES = ("parakeet", "whisper")


class EngineUnavailable(RuntimeError):
    """The requested ASR engine cannot run here. Nothing has been transcribed.

    Deliberately NOT the sibling of DiarizationUnavailable in consequence:
    diarization degrades -- _label_speakers catches its error and returns the
    unlabelled transcript, because the ASR work was already paid for and
    written to disk. An unavailable engine means no transcript exists at all,
    so this propagates to the CLI and exits non-zero. There is no automatic
    fallback to another engine on purpose: silently transcribing with a
    different model, at different quality, under a different `model` key in
    the payload, is worse than failing in one second.
    """


class ChunkEngine(Protocol):
    """An engine dsj drives chunk by chunk, so a run can resume.

    parakeet decodes every chunk from a fresh decoder state -- it passes no
    hidden state between windows -- and that statelessness is what makes
    resume exact rather than approximate. An engine that carries state across
    windows cannot implement this protocol; it owns its whole loop instead,
    the way dsj/whisper.py does, and returns a finished Transcription.
    """

    @property
    def sample_rate(self) -> int:
        """The rate decode() expects its samples at, from the loaded model."""
        ...

    @property
    def min_chunk_samples(self) -> int:
        """Below this, a decode would see a zero-length feature window."""
        ...

    def decode(self, samples: Any) -> list[AlignedToken]:
        """Tokens for one chunk, timed from the chunk's own start (t=0)."""
        ...


class EngineSpec(NamedTuple):
    """One row of the registry: where an engine lives and how it is driven."""

    name: str
    module: str
    kind: str  # "chunk" | "file"


# Availability is a RUNTIME question answered by each module's available(),
# never by this table: the tuple above is identical on every platform so that
# --help and --engine validation do not depend on where they run, and a script
# written on one machine fails elsewhere with a remedy, not "unknown engine".
_REGISTRY = {
    "parakeet": EngineSpec("parakeet", "dsj.parakeet", "chunk"),
    "whisper": EngineSpec("whisper", "dsj.whisper", "file"),
}


def get_engine(name: str) -> tuple[EngineSpec, ModuleType]:
    """Resolve an engine name to its spec and imported module, or refuse.

    The import is safe on every platform -- engine modules keep their backend
    imports function-local, exactly so this resolution can run where the
    backend cannot. `available()` is the platform test: it probes whether the
    backend would import (find_spec, shutil.which) without importing it, and
    a non-None return is the reason it would not, which becomes the error.
    No `sys.platform` is consulted anywhere; "does the dependency resolve" is
    the question that actually matters, and it answers correctly for free on
    machines a platform string would misclassify.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown engine {name!r}, expected one of {', '.join(ENGINES)}")
    module = import_module(spec.module)
    reason = module.available()
    if reason is not None:
        raise EngineUnavailable(f"the {name} engine cannot run here: {reason}")
    return spec, module


class Transcription(NamedTuple):
    """What an engine hands back to `suno.transcribe`.

    `sentences` are already in the payload's shape -- `{start, end, text,
    tokens: [{t, w}]}` -- so the caller writes them out rather than converting
    them.
    """

    text: str
    sentences: list[dict[str, Any]]
