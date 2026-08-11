"""The bead's acceptance criterion: a .mov transcribes like its .wav.

Marked slow because it loads the ASR model. The marker is registered but not
deselected -- these run in the default `uv run pytest`, because a test carved
out of the normal invocation is a test nobody observes.

No conftest stubbing here: this is the whole probe -> extract -> transcribe
path against real ffmpeg and real weights.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from jaano import media
from jaano.suno import CHUNK_S, transcribe

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"),
]


@pytest.fixture(scope="module")
def spoken_clip(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A short clip that exists as both a .mov and a hand-extracted .wav.

    Uses scratch/clip45.wav as the audio bed -- synthesized tones transcribe to
    nothing, and "same output" between two empty transcripts proves nothing.
    """
    src = Path("scratch/clip45.wav")
    if not src.exists():
        pytest.skip("scratch/clip45.wav not present")

    d = tmp_path_factory.mktemp("e2e")
    mov, wav = d / "clip.mov", d / "clip.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
         "-i", str(src), "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", str(mov)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mov),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    return mov, wav


@pytest.fixture(scope="module")
def chunked_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A .mov whose audio is longer than CHUNK_S, so the chunk loop runs.

    parakeet_mlx.parakeet.py:173 short-circuits to a single-pass transcription
    when the audio is no longer than chunk_duration, and chunk_callback -- the
    only source of the "running" state -- is never called on that path. A clip
    shorter than CHUNK_S therefore reports "extracting" then "done", with no
    running phase in between. Loop the 45s bed past the 120s boundary so the
    phase this test is about actually exists.
    """
    src = Path("scratch/clip45.wav")
    if not src.exists():
        pytest.skip("scratch/clip45.wav not present")

    out = tmp_path_factory.mktemp("e2e-chunked") / "long.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
         "-stream_loop", "2", "-i", str(src), "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out)],
        check=True,
    )
    assert media.probe(out).duration_s > CHUNK_S, "fixture is too short to chunk"
    return out


def test_mov_and_wav_produce_the_same_transcript(
    spoken_clip: tuple[Path, Path], tmp_path: Path
) -> None:
    mov, wav = spoken_clip
# diarize=False: these are about ASR, extraction and resume. Running the
    # real diarizer here would add ~13s of CoreML model load per call and put
    # an unsupervised clustering result inside equality assertions that must
    # hold exactly. The pass has its own tests, and scratch/diarize_gate.py
    # runs it end to end on the real meeting.
    from_mov = transcribe(mov, tmp_path / "mov.json", diarize=False)
    from_wav = transcribe(wav, tmp_path / "wav.json", diarize=False)

    # "audio" differs by construction -- it records which file was handed in.
    # Everything the transcript *is* must match.
    assert from_mov["text"] == from_wav["text"]
    assert from_mov["sentences"] == from_wav["sentences"]
    assert from_mov["audio"] == str(mov)
    assert from_wav["audio"] == str(wav)


def test_the_temp_wav_does_not_outlive_the_run(
    spoken_clip: tuple[Path, Path], tmp_path: Path
) -> None:
    mov, _ = spoken_clip
    before = set(Path(tempfile.gettempdir()).glob("jaano-*"))
    transcribe(mov, tmp_path / "out.json", diarize=False)
    assert set(Path(tempfile.gettempdir()).glob("jaano-*")) == before


def test_status_file_shows_every_phase(chunked_clip: Path, tmp_path: Path) -> None:
    """The one slow test that keeps diarization on.

    It costs ~13s of real CoreML model load, and it buys the only in-suite
    evidence that all four phases reach a watcher in order on a real .mov.
    "diarizing" is asserted rather than a speaker count: the state is emitted
    before the pass runs, so this stays green on a machine without the extra --
    which is the point, because it is the phase wiring under test, not senko.
    """
    seen: list[str] = []
    status = tmp_path / "status.json"
    transcribe(
        chunked_clip, tmp_path / "out.json", status_path=status,
        on_progress=lambda _p, state: seen.append(state),
    )
    assert "extracting" in seen
    assert "running" in seen
    assert "diarizing" in seen
    assert seen.index("running") < seen.index("diarizing")
    assert json.loads(status.read_text())["state"] == "done"
