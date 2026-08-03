"""Test doubles for parakeet-mlx, and the real clips the slow tests need.

from_pretrained downloads and loads ~2.4 GB of weights, which no unit test can
afford. transcribe() imports it inside the function body, so there is no
deixis.transcribe.from_pretrained attribute to patch -- the target is the
source module, parakeet_mlx.from_pretrained, which the function-local import
re-reads on every call.
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
SOURCE_AUDIO = REPO / "scratch" / "meeting.wav"

# 360s is the smallest clip that produces more than two chunks under the default
# 120s/15s geometry (starts at 0, 105, 210, 315), which is the minimum needed to
# show that a resumed run merges the way an uninterrupted one did. scratch/
# clip45.wav cannot serve: at 45 seconds it is under CHUNK_S and parakeet-mlx
# returns before the chunk loop ever runs (parakeet.py:173-175), so nothing this
# feature is about is observable on it.
CLIP_SECONDS = 360


@dataclass
class FakeToken:
    start: float
    end: float
    text: str


@dataclass
class FakeSentence:
    start: float
    end: float
    text: str
    tokens: list[FakeToken] = field(default_factory=list)


class FakeAlignedResult:
    def __init__(self, sentences: list[FakeSentence]) -> None:
        self.sentences = sentences
        self.text = " ".join(s.text for s in sentences)


class FakeModel:
    """Stands in for parakeet_mlx.parakeet.BaseParakeet.

    Mirrors the two things transcribe() touches: preprocessor_config.sample_rate
    and transcribe(path, *, chunk_duration, overlap_duration, chunk_callback).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        sentences: list[FakeSentence] | None = None,
        chunk_positions: list[tuple[float, float]] | None = None,
    ) -> None:
        self.preprocessor_config = SimpleNamespace(sample_rate=sample_rate)
        self.sentences = sentences if sentences is not None else []
        # (current, full) pairs in SAMPLES, exactly as parakeet passes them.
        self.chunk_positions = chunk_positions or []
        self.calls: list[dict] = []

    def transcribe(
        self,
        path,
        *,
        chunk_duration=None,
        overlap_duration=15.0,
        chunk_callback=None,
    ) -> FakeAlignedResult:
        self.calls.append(
            {
                "path": path,
                "chunk_duration": chunk_duration,
                "overlap_duration": overlap_duration,
            }
        )
        for current, full in self.chunk_positions:
            if chunk_callback is not None:
                chunk_callback(current, full)
        return FakeAlignedResult(self.sentences)


@pytest.fixture
def fake_parakeet(monkeypatch):
    """Install a FakeModel in place of the real weights.

    Usage:
        model = fake_parakeet(sample_rate=16_000, sentences=[...])
    """

    def install(**kwargs) -> FakeModel:
        model = FakeModel(**kwargs)
        monkeypatch.setattr("parakeet_mlx.from_pretrained", lambda model_id: model)
        return model

    return install


@pytest.fixture
def frozen_clock(monkeypatch):
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
            try:
                last[0] = next(it)
            except StopIteration:
                pass
            return last[0]

        monkeypatch.setattr("deixis.transcribe.time.monotonic", monotonic)

    return install


@pytest.fixture
def already_extracted_media(monkeypatch):
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
    from deixis import media

    monkeypatch.setattr(
        media,
        "probe",
        lambda path: media.AudioStream(
            codec_name="pcm_s16le", sample_rate=16_000, channels=1, duration_s=4427.028
        ),
    )
    monkeypatch.setattr(media, "needs_conversion", lambda stream, rate: False)


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
    from deixis.transcribe import DEFAULT_MODEL

    return DEFAULT_MODEL
