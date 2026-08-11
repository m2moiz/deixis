"""media.py against real ffmpeg, on fixtures ffmpeg synthesizes at test time.

Nothing here is stubbed: probe really shells out to ffprobe and extract_audio
really runs ffmpeg. That is the point -- this module exists to own the
subprocess boundary and its error surfaces, and a stubbed test of a subprocess
wrapper asserts only that the stub was called.

The fixtures are seconds long and a few kilobytes; no binary is committed.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from jaano import media

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _which_finds_nothing(_name: str) -> str | None:
    """Stand-in for shutil.which for the not-on-PATH tests.

    def, not lambda: an annotated lambda is not expressible, and under strict
    every unannotated lambda parameter is an error apiece.
    """
    return None


@pytest.fixture(scope="session")
def video_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2s stand-in for a screen recording: H.264 video, 48kHz stereo AAC.

    Tiny on purpose -- the properties under test are container and stream
    metadata, and none of them care how many pixels or seconds there are.
    """
    out = tmp_path_factory.mktemp("fixtures") / "clip.mov"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
        "-ac", "2", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    )
    return out


@pytest.fixture(scope="session")
def video_without_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("fixtures") / "silent.mov"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    )
    return out


@pytest.fixture(scope="session")
def ready_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Already exactly what the model wants: 16kHz mono s16le."""
    out = tmp_path_factory.mktemp("fixtures") / "ready.wav"
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=16000",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out),
    )
    return out


@pytest.fixture(scope="session")
def video_outlasting_its_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4s of video over 2s of audio -- the mic stopped, the recorder did not.

    Deliberately built without `-shortest`, which is what a screen recorder
    does when it keeps capturing after the microphone drops out.
    """
    out = tmp_path_factory.mktemp("fixtures") / "video_outlasts_audio.mov"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
        "-ac", "2", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out),
    )
    return out


def test_probe_reads_stream_metadata(video_with_audio: Path) -> None:
    s = media.probe(video_with_audio)
    assert s.codec_name == "aac"
    assert s.sample_rate == 48000
    assert s.channels == 2
    assert s.duration_s == pytest.approx(2.0, abs=0.2)


def test_probe_reads_a_plain_wav(ready_wav: Path) -> None:
    s = media.probe(ready_wav)
    assert s.codec_name == "pcm_s16le"
    assert s.sample_rate == 16000
    assert s.channels == 1


def test_probe_prefers_the_streams_duration_over_the_containers(
    video_outlasting_its_audio: Path,
) -> None:
    """The regression this file exists for.

    The container's duration is the max across its streams, so denominating
    the extraction bar in it leaves the bar stalled short of 100% for the whole
    stretch where video rolls on without audio -- observed at 88% on a real
    recording. probe() must report the audio stream's own duration.
    """
    probed = media.probe(video_outlasting_its_audio)
    container = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_outlasting_its_audio)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )

    # The fixture is only meaningful if the two actually disagree.
    assert container == pytest.approx(4.0, abs=0.3)
    assert probed.duration_s == pytest.approx(2.0, abs=0.2)


def test_probe_rejects_a_file_with_no_audio(video_without_audio: Path) -> None:
    # ffprobe exits 0 here and returns an empty stream list, so this is a
    # content check, not an exit-code check.
    with pytest.raises(media.NoAudioStream) as exc:
        media.probe(video_without_audio)
    assert "silent.mov" in str(exc.value)
    assert "nothing to transcribe" in str(exc.value)


def test_probe_reports_a_corrupt_container(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.mov"
    bad.write_bytes(b"\x00" * 200_000)
    with pytest.raises(media.MediaError) as exc:
        media.probe(bad)
    assert "corrupt.mov" in str(exc.value)
    assert "ffprobe could not read" in str(exc.value)


def test_probe_raises_before_shelling_out_for_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mov"
    with pytest.raises(FileNotFoundError) as exc:
        media.probe(missing)
    assert "nope.mov" in str(exc.value)


def test_missing_ffprobe_names_the_binary(
    monkeypatch: pytest.MonkeyPatch, video_with_audio: Path
) -> None:
    monkeypatch.setattr(media.shutil, "which", _which_finds_nothing)
    with pytest.raises(media.FFmpegNotFound) as exc:
        media.probe(video_with_audio)
    assert "ffprobe is not on PATH" in str(exc.value)
    assert "brew install ffmpeg" in str(exc.value)


def test_conversion_needed_for_a_screen_recording(video_with_audio: Path) -> None:
    assert media.needs_conversion(media.probe(video_with_audio), 16000) is True


def test_conversion_skipped_for_an_already_ready_wav(ready_wav: Path) -> None:
    assert media.needs_conversion(media.probe(ready_wav), 16000) is False


def test_conversion_needed_when_the_rate_does_not_match_the_model(ready_wav: Path) -> None:
    # A 16kHz mono wav is still wrong if the model asks for something else --
    # which is why the rate is a parameter and not a constant.
    assert media.needs_conversion(media.probe(ready_wav), 22050) is True


def test_extract_produces_model_shaped_audio(video_with_audio: Path, tmp_path: Path) -> None:
    dest = media.extract_audio(video_with_audio, tmp_path / "a.wav", 16000)
    assert dest.exists()
    s = media.probe(dest)
    assert (s.codec_name, s.sample_rate, s.channels) == ("pcm_s16le", 16000, 1)
    assert s.duration_s == pytest.approx(2.0, abs=0.2)


def test_extract_reports_progress_in_audio_seconds(
    video_with_audio: Path, tmp_path: Path
) -> None:
    # A 2s fixture finishes well inside one -stats_period, so the assertion
    # below rides on ffmpeg's terminal `progress=end` block, which always
    # carries a final out_time_us. Do not lengthen the fixture to make progress
    # appear -- that would hide a regression in the parsing.
    seen: list[float] = []
    media.extract_audio(
        video_with_audio, tmp_path / "a.wav", 16000,
        on_progress=seen.append,
    )
    assert seen, "no progress was reported"
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(2.0, abs=0.2)


def test_extract_names_the_file_when_there_is_no_audio(
    video_without_audio: Path, tmp_path: Path
) -> None:
    with pytest.raises(media.MediaError) as exc:
        media.extract_audio(video_without_audio, tmp_path / "a.wav", 16000)
    assert "silent.mov" in str(exc.value)
    assert "no audio track" in str(exc.value)


def test_missing_ffmpeg_names_the_binary(
    monkeypatch: pytest.MonkeyPatch, video_with_audio: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(media.shutil, "which", _which_finds_nothing)
    with pytest.raises(media.FFmpegNotFound) as exc:
        media.extract_audio(video_with_audio, tmp_path / "a.wav", 16000)
    assert "ffmpeg is not on PATH" in str(exc.value)
    assert "brew install ffmpeg" in str(exc.value)


def test_extraction_closes_the_ffmpeg_pipe(tmp_path: Path, video_with_audio: Path) -> None:
    """The ffmpeg stdout pipe is closed explicitly, not left to the collector.

    Popen with stdout=PIPE hands back a TextIOWrapper that stays open until
    something closes it, and `proc.wait()` does not. CPython's refcounting then
    reclaims it when `proc` falls out of scope -- so a descriptor COUNT cannot
    see the bug, which is why this test asserts on the ResourceWarning instead.

    That warning is the real signal, and it is invisible in a normal run
    because Python ignores ResourceWarning by default. It fired throughout this
    project's history until it was found by running the suite once with
    `-W default` and reading the output rather than the pass count.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        media.extract_audio(video_with_audio, tmp_path / "a.wav", 16000)
        gc.collect()

    leaked = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not leaked, f"extract_audio leaked: {[str(w.message) for w in leaked]}"
