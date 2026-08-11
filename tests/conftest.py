"""Test doubles for parakeet-mlx, and the real clips the slow tests need.

from_pretrained downloads and loads ~2.4 GB of weights, which no unit test can
afford. transcribe() imports it inside the function body, so there is no
jaano.suno.from_pretrained attribute to patch -- the target is the
source module, parakeet_mlx.from_pretrained, which the function-local import
re-reads on every call.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from parakeet_mlx import DecodingConfig
    from parakeet_mlx.alignment import AlignedResult

    from jaano.media import AudioStream
    from jaano.merge import Turn

REPO = Path(__file__).resolve().parent.parent
SOURCE_AUDIO = REPO / "scratch" / "meeting.wav"

# 360s is the smallest clip that produces more than two chunks under the default
# 120s/15s geometry (starts at 0, 105, 210, 315), which is the minimum needed to
# show that a resumed run merges the way an uninterrupted one did. scratch/
# clip45.wav cannot serve: at 45 seconds it is under CHUNK_S and parakeet-mlx
# returns before the chunk loop ever runs (parakeet.py:173-175), so nothing this
# feature is about is observable on it.
CLIP_SECONDS = 360


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give every slow-marked test the long timeout ceiling.

    One chokepoint, so the ceiling cannot drift away from the mark. The global
    --timeout=120 is right for the fast lane and wrong here: a first run
    downloads ~2.4 GB of weights and the diarizer test runs a 6-minute clip.

    Done as a collection hook rather than a composed `slow = mark.slow(...)`
    alias because the six mark sites are a mix of module-level `pytestmark`
    (one of which also carries a skipif guard) and per-test decorators. Pairing
    the two marks by hand at each site is how they get out of sync, and it
    would need every test module to import a name from conftest.
    """
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.timeout(1800))


@dataclass
class FakeToken:
    """A token spec, converted to a real AlignedToken on every generate() call.

    Fresh objects per call, deliberately: transcribe_chunked mutates
    token.start in place to apply the chunk offset, so a shared instance would
    accumulate offsets across chunks.
    """

    start: float
    end: float
    text: str


class FakeModel:
    """Stands in for parakeet_mlx.parakeet.BaseParakeet.

    Mirrors what jaano.chunking.transcribe_chunked touches:
    preprocessor_config.sample_rate, preprocessor_config.hop_length, and
    generate(mel, decoding_config=...). The chunk loop, the offsets and the
    merge are the REAL ones -- only the decode is faked -- so these tests
    exercise the boundary arithmetic they appear to.

    `audio_s` defaults under CHUNK_S so a fake run is a single chunk and the
    overlap merge never runs on synthetic tokens, where its behaviour would be
    arbitrary. Tests that want several chunks pass a longer `audio_s` together
    with `tokens=[]`; the multi-chunk merge is proved against the real model in
    tests/test_chunking.py, not here.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        tokens: list[FakeToken] | None = None,
        audio_s: float = 100.0,
        hop_length: int = 160,
    ) -> None:
        self.preprocessor_config = SimpleNamespace(
            sample_rate=sample_rate, hop_length=hop_length
        )
        self.tokens = tokens if tokens is not None else []
        self.total_samples = int(audio_s * sample_rate)
        # Whatever jaano.chunking.get_logmel was stubbed to return -- the
        # tests assert on the SLICES handed to generate(), never on their
        # contents, so the element type is deliberately open.
        self.mels: list[Any] = []

    def generate(
        self, mel: Any, *, decoding_config: DecodingConfig | None = None
    ) -> list[AlignedResult]:
        """Decode `self.tokens` as one chunk, ignoring the mel entirely.

        Signature-compatible with jaano.chunking._Generates, which is the
        Protocol naming the only method jaano calls on a model. `mel` is Any
        for the same reason it is there: mlx's array type is a compiled
        extension with no stubs.
        """
        from parakeet_mlx.alignment import (
            AlignedToken,
            SentenceConfig,
            sentences_to_result,
            tokens_to_sentences,
        )

        self.mels.append(mel)
        decoded = [
            AlignedToken(id=i, text=t.text, start=t.start, duration=t.end - t.start)
            for i, t in enumerate(self.tokens)
        ]
        cfg: Any = (decoding_config.sentence if decoding_config else None) or SentenceConfig()
        return [sentences_to_result(tokens_to_sentences(decoded, cfg))]


@pytest.fixture
def fake_parakeet(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeModel]:
    """Install a FakeModel in place of the real weights.

    Also stubs the two library functions the fake model cannot stand in for:
    load_audio (which would shell out to ffmpeg on a path that need not exist)
    and get_logmel (which would compute a real mel spectrogram the fake decoder
    then ignores). `range` stands in for the decoded samples -- it has a len,
    it slices, and it costs nothing at 74-minute lengths.

    Usage:
        model = fake_parakeet(sample_rate=16_000, tokens=[...])
    """

    def install(**kwargs: Any) -> FakeModel:
        model = FakeModel(**kwargs)

        # def, not lambda: an annotated lambda is not expressible, and under
        # strict every unannotated parameter here is an error apiece.
        def from_pretrained(model_id: str) -> FakeModel:
            return model

        def load_audio(path: Path, rate: int, *a: Any, **k: Any) -> range:
            return range(model.total_samples)

        def get_logmel(audio: Any, cfg: Any) -> Any:
            return audio

        monkeypatch.setattr("parakeet_mlx.from_pretrained", from_pretrained)
        monkeypatch.setattr("parakeet_mlx.audio.load_audio", load_audio)
        monkeypatch.setattr("jaano.chunking.get_logmel", get_logmel)
        return model

    return install


@pytest.fixture
def fake_media(tmp_path: Path) -> Path:
    """A real file to stand in as the source media for a faked run.

    It needs to exist even though load_audio is stubbed and never opens it:
    transcribe() fingerprints the source by size and mtime, and stat() on a
    path that is not there raises.
    """
    p = tmp_path / "in.wav"
    p.write_bytes(b"RIFF")
    return p


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[float]], None]:
    """Pin time.monotonic so elapsed_s is exactly reproducible.

    Deliberately forgiving past the end of `ticks` (it keeps returning the last
    value) so a test does not have to count internal time.monotonic() calls.

    Usage:
        frozen_clock([100.0, 100.0])   # start and end -> elapsed == 0.0
    """

    def install(ticks: list[float]) -> None:
        it = iter(ticks)
        last = [ticks[-1]]

        def monotonic() -> float:
            # suppress, not try/except/pass: running past the end of the tick
            # list means the test asked for more clock reads than it scripted,
            # and holding the last value is the intended behaviour.
            with contextlib.suppress(StopIteration):
                last[0] = next(it)
            return last[0]

        monkeypatch.setattr("jaano.suno.time.monotonic", monotonic)

    return install


def _probe(path: Path) -> AudioStream:
    """Stand-in for media.probe: always an already-conforming 16k mono wav."""
    from jaano.media import AudioStream

    return AudioStream(
        codec_name="pcm_s16le", sample_rate=16_000, channels=1, duration_s=4427.028
    )


def _needs_conversion(stream: AudioStream, rate: int) -> bool:
    """Stand-in for media.needs_conversion: the stub wav never needs converting."""
    return False


@pytest.fixture
def already_extracted_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make transcribe() treat any path as an already-conforming wav.

    Those tests are about the callback wiring and the emitted index, not about
    ffmpeg. Stubbing probe() keeps them free of both model weights AND a real
    media file, so they stay fast and hermetic.

    Opt-in, not autouse: test_media.py and test_transcribe_e2e.py exercise the
    real ffmpeg path, and an autouse stub of probe()/needs_conversion() would
    silently replace the behaviour they exist to assert. A module that wants
    the stub asks for it with
    `pytestmark = pytest.mark.usefixtures("already_extracted_media")`.
    """
    from jaano import media

    monkeypatch.setattr(
        media,
        "probe",
        # def, not lambda: strict flags every unannotated lambda parameter.
        _probe,
    )
    monkeypatch.setattr(media, "needs_conversion", _needs_conversion)


@pytest.fixture
def fake_turns(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[Path]]:
    """Install a stand-in for the diarization pass.

    Loading senko costs ~12s of CoreML model load, so no fast test may reach
    the real one. transcribe() imports jaano.diarize inside the function body
    for the same reason it imports parakeet there -- the extra is usually
    absent -- so the patch target is the source module, whose attribute the
    function-local import re-reads on every call.

    Usage:
        fake_turns(turns=[Turn(0.0, 400.0, 0)], labels=["SPEAKER_01"])
        fake_turns(raises=DiarizationUnavailable("nope"))
        fake_turns(turns=[...], labels=[...], then=lambda wav: ...)

    `then` runs inside the faked diarization call, which is where a test can
    observe what was already on disk when the pass began.
    """
    from jaano.diarize import Diarization

    def install(
        turns: list[Turn] | None = None,
        labels: list[str] | None = None,
        raises: Exception | None = None,
        then: Callable[[Path], None] | None = None,
    ) -> list[Path]:
        calls: list[Path] = []

        def speaker_turns(wav: Path) -> Diarization:
            calls.append(wav)
            if then is not None:
                then(wav)
            if raises is not None:
                raise raises
            return Diarization(
                turns=list(turns or []),
                labels=list(labels or []),
                provenance="senko 0.0.0-fake",
            )

        monkeypatch.setattr("jaano.diarize.speaker_turns", speaker_turns)
        return calls

    return install


@pytest.fixture
def no_real_diarizer(fake_turns: Callable[..., list[Path]]) -> None:
    """Make diarization fail cheaply, for tests that are about something else.

    Diarization is on by default, so every faked run would otherwise reach the
    real senko: ~12s of CoreML model load (~51s the first time on a machine),
    against a stand-in media file it cannot open anyway. The run degrades and
    the test still passes -- which is exactly the problem, because it passes
    twelve seconds slower and nothing says why.

    A module asks for it with
    `pytestmark = pytest.mark.usefixtures("no_real_diarizer")`, and any test in
    it that actually wants labels calls `fake_turns` itself, replacing this.
    """
    from jaano.diarize import DiarizationUnavailable

    fake_turns(raises=DiarizationUnavailable("no diarizer in this test"))


@pytest.fixture(scope="session")
def chunked_audio_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real clip long enough to cross several chunk boundaries."""
    if not SOURCE_AUDIO.exists():
        pytest.skip(f"{SOURCE_AUDIO} not present")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    # Cached beside the source rather than under tmp_path: cutting it costs a
    # couple of seconds and every test session would otherwise pay it again.
    clip = SOURCE_AUDIO.parent / f"clip{CLIP_SECONDS}.wav"
    if not clip.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-i", str(SOURCE_AUDIO), "-t", str(CLIP_SECONDS),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(clip)],
            check=True, capture_output=True,
        )
    return clip


@pytest.fixture(scope="session")
def model_id() -> str:
    from jaano.suno import DEFAULT_MODEL

    return DEFAULT_MODEL
