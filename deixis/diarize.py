"""senko, behind a boundary that degrades instead of failing.

Diarization is an optional pass over a transcript that is already correct
without it. That asymmetry is the whole design of this module: an hour of ASR
has already been paid for and written to disk by the time anything here runs, so
no way this pass can go wrong is allowed to cost that run. Every foreseeable
failure -- the extra not installed, a model download that cannot reach the net,
a file senko will not open, a diarizer that finds nobody -- arrives at the caller
as one exception type it can catch in one place.

Deliberately narrow, though. A `TypeError` from a bug in this code or in the
merge is NOT converted; it propagates and takes the run down. A missing
dependency should be quiet, a bug should be loud, and a bare `except Exception`
cannot tell them apart.

senko installs 29 packages including scikit-learn, scipy, umap-learn and
coremltools, and only runs on CoreML, so it is an extra rather than a core
dependency: `uv sync --extra diarize`.
"""

from __future__ import annotations

__all__ = [
    "INSTALL_HINT",
    "DiarizationUnavailable",
    "Diarization",
    "speaker_turns",
]

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from deixis.merge import Turn

INSTALL_HINT = "uv sync --extra diarize"


class DiarizationUnavailable(RuntimeError):
    """Diarization could not run. The unlabelled transcript is still correct."""


class Diarization(NamedTuple):
    """What one diarization pass yields, in the shape the transcript needs.

    `turns` carries integer speaker indices and `labels` maps them back to
    senko's names, because the transcript stores the index (1.7% of file size)
    rather than the name (3.3%) and needs the mapping written once at the top.
    `provenance` is the ~30 bytes that let a reader tell "diarization was not
    run" apart from "diarization ran and found one speaker".
    """

    turns: list[Turn]
    labels: list[str]
    provenance: str


def speaker_turns(wav: Path) -> Diarization:
    """Diarize `wav`, or raise DiarizationUnavailable.

    `wav` is the 16 kHz mono pcm_s16le file deixis.media already produced for
    the ASR pass, which is exactly what senko wants. The path is handed over
    rather than samples, so senko reads the rate from the header instead of both
    sides hardcoding an agreement that happens to hold today.

    The diarizer is constructed here, once per call: ~12s of warm model load
    against ~8s of inference on a 74-minute file, and there is nothing for a
    single-file CLI to cache it in.
    """
    senko = _import_senko()

    try:
        # quiet=True is already the default; passed explicitly because deixis
        # renders its own progress and a second progress tree on stderr would
        # fight it.
        result = senko.Diarizer(quiet=True).diarize(str(wav))
    except (senko.AudioFormatError, OSError) as exc:
        # OSError covers the model download and the CoreML cache: no network on
        # a first run, or a read-only cache directory.
        raise DiarizationUnavailable(
            f"senko could not diarize {wav}: {type(exc).__name__}: {exc}"
        ) from exc

    # merged_segments, never raw_segments: the raw list overlaps and nests (868
    # entries against 220 merged on the reference recording, with (0.0, 57.81)
    # immediately followed by (2.94, 4.66)), and the token vote assumes a
    # non-overlapping partition -- against the raw list it would count the same
    # second for two speakers.
    #
    # `result` itself is None when senko's VAD found no speech anywhere
    # (diarizer.py:359) -- a silent recording, which a screen capture with a
    # dead mic really is. Same answer as an empty segment list.
    segments = result["merged_segments"] if result else []
    if not segments:
        raise DiarizationUnavailable(
            f"senko found no speaker turns in {wav}, so there is nothing to "
            f"label sentences with."
        )

    turns, labels = _to_turns(segments)
    return Diarization(turns=turns, labels=labels, provenance=_provenance())


def _import_senko():
    """Import senko, or say how to get it.

    Function-local, mirroring how transcribe() imports parakeet: the extra being
    absent is the common case and must cost nothing at deixis import time.
    """
    try:
        import senko
    except ImportError as exc:
        raise DiarizationUnavailable(
            f"senko is not installed, so sentences cannot be labelled with who "
            f"spoke. Install it with `{INSTALL_HINT}`."
        ) from exc
    return senko


def _provenance() -> str:
    from importlib.metadata import version

    return f"senko {version('senko')}"


def _to_turns(segments: Sequence[dict]) -> tuple[list[Turn], list[str]]:
    """senko's named segments, as sorted turns over integer speaker indices.

    Labels are sorted so the index a sentence carries is stable across runs of
    the same file. It is still an arbitrary cluster id -- SPEAKER_01 in one
    recording has nothing to do with SPEAKER_01 in another.
    """
    labels = sorted({segment["speaker"] for segment in segments})
    index = {label: i for i, label in enumerate(labels)}
    turns = sorted(
        Turn(float(s["start"]), float(s["end"]), index[s["speaker"]]) for s in segments
    )
    return turns, labels
