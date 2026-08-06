"""The fail-soft boundary, with senko standing in for itself.

The four fast tests here never load the real diarizer: they install a fake
`senko` module in sys.modules, which is what the function-local `import senko`
inside speaker_turns then finds. A unit test that quietly pulled in 12 seconds
of CoreML model load would still pass, and would still be the wrong test, so
`no_real_senko` below fails it instead.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

import pytest

from deixis.diarize import Diarization, DiarizationUnavailable, speaker_turns
from deixis.merge import Turn

# senko ships no type information, so its result arrives as Any -- deixis.diarize
# pins the same shape at its own boundary (Sequence[Mapping[str, Any]]). The
# stand-in mirrors that: the keys are known, the values are whatever senko put
# there. None is what senko returns when its VAD finds no speech at all.
SenkoResult = dict[str, list[dict[str, Any]]] | None

# speaker_turns hands senko `str(wav)`, not the Path.
DiarizeCall = Callable[[str], SenkoResult]


class SenkoInstaller(Protocol):
    """What the `fake_senko` fixture hands a test.

    A Protocol rather than a bare Callable because the fixture also exposes the
    stand-in AudioFormatError as an attribute, which tests raise.
    """

    AudioFormatError: type[Exception]

    def __call__(self, diarize: DiarizeCall) -> ModuleType: ...


@pytest.fixture(autouse=True)
def no_real_senko(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a fast test that imported the real senko rather than a stand-in."""
    before = {name for name in sys.modules if name.startswith("senko")}
    yield
    # pytest leaves FixtureRequest.node unannotated upstream, so both it and
    # get_closest_marker arrive Unknown; there is no typed way to ask.
    if request.node.get_closest_marker("slow"):  # pyright: ignore[reportUnknownMemberType]
        return
    leaked = {name for name in sys.modules if name.startswith("senko")} - before
    # The fake is a bare ModuleType named "senko"; the real package drags in
    # senko.diarizer and friends, which is what this catches.
    assert not {name for name in leaked if name != "senko"}, (
        f"a fast test loaded the real senko: {leaked}"
    )


@pytest.fixture
def fake_senko(monkeypatch: pytest.MonkeyPatch) -> SenkoInstaller:
    """Install a stand-in `senko` module whose diarize() the test controls.

    Usage:
        fake_senko(lambda wav_path: {"merged_segments": [...]})
    """

    class AudioFormatError(Exception):
        pass

    def install(diarize: DiarizeCall) -> ModuleType:
        module = ModuleType("senko")

        class Diarizer:
            def __init__(self, **kwargs: object) -> None:
                module.constructed_with = kwargs  # type: ignore[attr-defined]

            def diarize(self, wav_path: str) -> SenkoResult:
                return diarize(wav_path)

        module.AudioFormatError = AudioFormatError  # type: ignore[attr-defined]
        module.Diarizer = Diarizer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "senko", module)
        return module

    install.AudioFormatError = AudioFormatError  # type: ignore[attr-defined]
    # cast, not ignore: `install` really does carry both halves of the Protocol
    # by the time it is returned, but a function object cannot declare an
    # attribute for the checker to see.
    return cast(SenkoInstaller, install)


def test_a_missing_senko_becomes_DiarizationUnavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None in sys.modules is how the import system spells "this module is not
    # available"; `import senko` against it raises ImportError.
    monkeypatch.setitem(sys.modules, "senko", None)

    with pytest.raises(DiarizationUnavailable) as caught:
        speaker_turns(Path("anything.wav"))

    # The message has to carry the fix, the way media.py's says
    # `brew install ffmpeg`. This is the common failure, not an exotic one:
    # senko is an extra and most installs will not have it.
    assert "uv sync --extra diarize" in str(caught.value)


def test_empty_turns_become_DiarizationUnavailable(fake_senko: SenkoInstaller) -> None:
    fake_senko(lambda wav_path: {"merged_segments": []})

    with pytest.raises(DiarizationUnavailable):
        speaker_turns(Path("silent.wav"))


def test_silence_becomes_DiarizationUnavailable(fake_senko: SenkoInstaller) -> None:
    # senko returns None rather than an empty result when its VAD finds no
    # speech at all. A screen capture with a dead mic is exactly that, and
    # subscripting the None would be an AttributeError in an optional pass.
    fake_senko(lambda wav_path: None)

    with pytest.raises(DiarizationUnavailable):
        speaker_turns(Path("silent.wav"))


def test_an_unreadable_file_becomes_DiarizationUnavailable(fake_senko: SenkoInstaller) -> None:
    def explode(wav_path: str) -> SenkoResult:
        raise fake_senko.AudioFormatError("not a wav")

    fake_senko(explode)

    with pytest.raises(DiarizationUnavailable):
        speaker_turns(Path("not-a-wav.mov"))


def test_a_failed_model_load_becomes_DiarizationUnavailable(fake_senko: SenkoInstaller) -> None:
    def explode(wav_path: str) -> SenkoResult:
        raise OSError("could not download the embedding model")

    fake_senko(explode)

    with pytest.raises(DiarizationUnavailable):
        speaker_turns(Path("clip.wav"))


def test_a_bug_is_not_swallowed(fake_senko: SenkoInstaller) -> None:
    # The deliberate narrowness of the boundary. A missing dependency should be
    # quiet; a bug should be loud. If this ever starts raising
    # DiarizationUnavailable, someone has widened a catch to `except Exception`
    # and every future bug in this pass has become invisible.
    def explode(wav_path: str) -> SenkoResult:
        raise TypeError("unsupported operand")

    fake_senko(explode)

    with pytest.raises(TypeError):
        speaker_turns(Path("clip.wav"))


def test_raw_segments_are_not_used(fake_senko: SenkoInstaller) -> None:
    # senko returns both, and the raw list overlaps and nests. The token vote
    # assumes a partition, so counting against the raw list would credit the
    # same second to two speakers.
    fake_senko(
        lambda wav_path: {
            "raw_segments": [
                {"speaker": "SPEAKER_01", "start": 0.0, "end": 57.81},
                {"speaker": "SPEAKER_02", "start": 2.94, "end": 4.66},
                {"speaker": "SPEAKER_01", "start": 3.0, "end": 50.0},
            ],
            "merged_segments": [
                {"speaker": "SPEAKER_01", "start": 0.0, "end": 57.81},
                {"speaker": "SPEAKER_02", "start": 57.81, "end": 73.80},
            ],
        }
    )

    result = speaker_turns(Path("clip.wav"))

    assert result.turns == [Turn(0.0, 57.81, 0), Turn(57.81, 73.80, 1)]


def test_labels_map_back_to_senkos_names(fake_senko: SenkoInstaller) -> None:
    fake_senko(
        lambda wav_path: {
            "merged_segments": [
                {"speaker": "SPEAKER_02", "start": 10.0, "end": 20.0},
                {"speaker": "SPEAKER_01", "start": 0.0, "end": 10.0},
            ]
        }
    )

    result = speaker_turns(Path("clip.wav"))

    assert isinstance(result, Diarization)
    # Sorted, so a sentence's stored index means the same thing on a re-run.
    assert result.labels == ["SPEAKER_01", "SPEAKER_02"]
    assert result.turns == [Turn(0.0, 10.0, 0), Turn(10.0, 20.0, 1)]
    assert result.provenance.startswith("senko ")


def test_the_diarizer_is_asked_to_stay_quiet(fake_senko: SenkoInstaller) -> None:
    # deixis renders its own progress; a second progress tree on stderr would
    # fight the bar.
    module = fake_senko(
        lambda wav_path: {"merged_segments": [{"speaker": "S", "start": 0.0, "end": 1.0}]}
    )

    speaker_turns(Path("clip.wav"))

    assert module.constructed_with == {"quiet": True}


# --- The real diarizer, on a real clip -----------------------------------


@pytest.mark.slow
def test_senko_labels_the_real_clip(chunked_audio_path: Path) -> None:
    # Asserts only what will not flake. Not a speaker count: senko finds three
    # clusters for the two people on this recording, so `== 2` would be red
    # today and `== 3` would pin a bug. Not a wall time either -- that is a
    # measurement, and a measurement in a test suite is a flake on a loaded
    # machine.
    result = speaker_turns(chunked_audio_path)

    assert len(result.turns) >= 2
    assert result.labels
    assert all(0 <= t.speaker < len(result.labels) for t in result.turns)

    for turn in result.turns:
        assert turn.end > turn.start
    for earlier, later in itertools.pairwise(result.turns):
        assert later.start >= earlier.end, "turns overlap; the vote assumes they do not"

    labelled = sum(t.end - t.start for t in result.turns)
    assert labelled <= result.turns[-1].end
