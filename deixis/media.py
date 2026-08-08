"""Everything that talks to ffmpeg.

deixis keeps the video on disk and treats it as a random-access resource, so
this module is the single place that knows how to open one. Today it extracts
audio for the transcript; frame retrieval at a timestamp belongs here too.

parakeet-mlx will happily shell out to ffmpeg itself -- `load_audio` in
parakeet_mlx/audio.py does exactly the conversion below. The reason deixis
does it instead is that parakeet's call is opaque: no progress for the minute
it spends on an hour-long recording, and a failure message that is the ffmpeg
build banner with the actual diagnosis buried in it.
"""

from __future__ import annotations

__all__ = [
    "AudioStream",
    "FFmpegNotFound",
    "MediaError",
    "NoAudioStream",
    "NoVideoStream",
    "extract_audio",
    "extract_tile_grid",
    "needs_conversion",
    "probe",
]

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class MediaError(RuntimeError):
    """ffmpeg could not do what we asked of this file."""


class FFmpegNotFound(MediaError):
    """ffmpeg or ffprobe is not on PATH."""


class NoAudioStream(MediaError):
    """The file opened, but carries no audio track to transcribe."""


class NoVideoStream(MediaError):
    """The file opened, but carries no picture to scan for changes."""


@dataclass(frozen=True)
class AudioStream:
    """What ffprobe reports about the one audio stream we intend to read."""

    codec_name: str
    sample_rate: int
    channels: int
    duration_s: float


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegNotFound(
            f"{name} is not on PATH. deixis reads video through ffmpeg; "
            f"install it with `brew install ffmpeg`."
        )
    return path


def probe(media: Path) -> AudioStream:
    """Describe `media`'s first audio stream.

    Raises NoAudioStream when there is none -- which ffprobe reports as a
    successful run with an empty stream list, not as an error.
    """
    if not media.exists():
        raise FileNotFoundError(media)

    proc = subprocess.run(
        [
            _tool("ffprobe"),
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration",
            "-show_entries", "format=duration",
            "-print_format", "json",
            str(media),
        ],
        capture_output=True,
        text=True,
        # Explicitly not check=True: a non-zero exit is handled two lines down,
        # where ffprobe's own stderr becomes an actionable MediaError. Raising
        # CalledProcessError here would replace that with a bare exit code.
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(
            f"ffprobe could not read {media}: {proc.stderr.strip()}\n"
            f"Check the file is complete and is a format ffmpeg supports "
            f"(`ffprobe {media}` shows the same detail)."
        )

    # Annotated at the boundary: json.loads returns Any, and every downstream
    # read inherits that Unknown-ness under strict. One name typed here is
    # cheaper than a cast at each of the five field reads below.
    info: dict[str, Any] = json.loads(proc.stdout)
    streams: list[dict[str, Any]] = info.get("streams") or []
    if not streams:
        raise NoAudioStream(
            f"{media} has no audio stream, so there is nothing to transcribe. "
            f"If this is a silent screen capture, re-record with audio enabled."
        )

    stream: dict[str, Any] = streams[0]
    # This field describes the audio stream, so the stream's own duration wins.
    # The container's is the max across every stream, and a recorder that keeps
    # rolling video after the mic stops gives a container longer than its audio
    # -- an extraction bar denominated in that never reaches 100%. Fall back to
    # the container only when the stream carries no duration of its own, as some
    # do not. Zero means ffprobe genuinely does not know; progress then reports
    # elapsed only, which Progress already handles.
    fmt: dict[str, Any] = info.get("format", {})
    duration: str | float = stream.get("duration") or fmt.get("duration") or 0.0
    return AudioStream(
        codec_name=stream["codec_name"],
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        duration_s=float(duration),
    )


def needs_conversion(stream: AudioStream, sample_rate: int) -> bool:
    """Is `stream` already exactly what the ASR preprocessor consumes?

    The target is not a guess: parakeet_mlx.audio.load_audio asks ffmpeg for
    `-ac 1 -acodec pcm_s16le -ar <preprocessor rate>` and nothing else, so a
    file already in that shape can be handed straight to the model. Anyone who
    pre-extracted their audio by hand lands here and pays nothing.
    """
    return not (
        stream.codec_name == "pcm_s16le"
        and stream.sample_rate == sample_rate
        and stream.channels == 1
    )


def extract_audio(
    media: Path,
    dest: Path,
    sample_rate: int,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Write `media`'s audio to `dest` as mono pcm_s16le at `sample_rate`.

    `on_progress` is called with seconds of audio written so far. Extraction
    of an hour-long 4GB recording is tens of seconds -- a small fraction of the
    run, but not a fraction anyone should spend watching a frozen bar.
    """
    cmd = [
        _tool("ffmpeg"),
        "-nostdin",            # never block on a tty; this runs detached
        "-hide_banner",
        "-loglevel", "error",  # parakeet's own call omits this, which is why
                               # its failures arrive as a build-config dump
        "-progress", "pipe:1",
        "-nostats",
        "-stats_period", "0.5",
        "-y",
        "-i", str(media),
        "-vn",                 # the video stays on disk; deixis retrieves
                               # frames from it later, at timestamps the
                               # transcript justifies
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dest),
    ]

    # stderr goes to a file rather than a pipe: with two pipes and only one
    # reader, a chatty decoder can fill the stderr buffer and deadlock while we
    # sit reading stdout.
    # Popen as a context manager, not a bare call: __exit__ closes the stdout
    # pipe and waits. Without it the pipe survives until the garbage collector
    # happens to run, which leaks a file descriptor per extraction -- invisible
    # in a normal run because Python ignores ResourceWarning by default, and
    # unbounded in a long-lived process that transcribes more than one file.
    with (
        tempfile.TemporaryFile("w+") as errfile,
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile, text=True) as proc,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            # ffmpeg 8.1.2 also emits out_time_ms, whose value is microseconds
            # too (out_time_ms=30016000 for a 30.016s file). out_time_us is the
            # only one of the pair that means what it says.
            if key == "out_time_us" and value != "N/A" and on_progress:
                on_progress(int(value) / 1_000_000)
        returncode = proc.wait()
        errfile.seek(0)
        stderr = errfile.read().strip()

    if returncode != 0:
        raise MediaError(
            f"ffmpeg failed to extract audio from {media} (exit {returncode}):\n"
            f"{stderr}\n"
            f"If the file has no audio track there is nothing to transcribe; "
            f"otherwise try `ffmpeg -i {media} -vn -ac 1 -ar {sample_rate} "
            f"-c:a pcm_s16le out.wav` by hand to see the full log."
        )
    return dest


def has_video(media: Path) -> bool:
    """Does `media` carry a picture at all?

    Asked separately from probe() because that one describes the audio stream
    and raises when there is none -- an audio-only file is a perfectly good
    transcription input and a hopeless input to a change scan.
    """
    if not media.exists():
        raise FileNotFoundError(media)
    proc = subprocess.run(
        [
            _tool("ffprobe"),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-print_format", "json",
            str(media),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise MediaError(f"ffprobe could not read {media}: {proc.stderr.strip()}")
    info: dict[str, Any] = json.loads(proc.stdout)
    return bool(info.get("streams"))


def extract_tile_grid(
    media: Path,
    *,
    fps: float,
    width: int,
    height: int,
    on_progress: Callable[[float], None] | None = None,
) -> NDArray[np.uint8]:
    """Sample `media`'s picture down to a grid of greyscale tile means.

    ffmpeg does the decode, the downscale and the colour conversion; what
    arrives here is `width * height` bytes per sampled frame and nothing else.
    A 33-minute 2940x1912 recording at 1 fps and a 128x84 grid is 21 MB in
    total, which is why the whole thing is returned as one array rather than
    streamed -- and why the caller can afford to sweep parameters over it
    without decoding twice.

    The downscale is not merely a size reduction. Averaging each ~23x23-pixel
    tile suppresses smooth low-contrast motion (a webcam tile) while preserving
    the thin high-contrast edges of text and UI chrome, which is the behaviour
    change detection wants and would otherwise have to implement.

    Args:
        media: The video to scan.
        fps: Frames to sample per second of video.
        width: Tiles across.
        height: Tiles down.
        on_progress: Called with seconds of video scanned so far. Derived from
            the frame count rather than from ffmpeg's own `-progress`, which
            would need stdout -- and stdout is carrying the pixels.

    Returns:
        An (N, height, width) uint8 array. N is 0 for a file with no frames.

    Raises:
        NoVideoStream: if `media` has no video stream.
        MediaError: if ffmpeg exits non-zero.
        ValueError: if `fps`, `width` or `height` is not positive.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"grid must be positive, got {width}x{height}")
    if not has_video(media):
        raise NoVideoStream(
            f"{media} has no video stream, so there are no frames to scan. "
            f"A transcript of an audio-only file is complete on its own."
        )

    cmd = [
        _tool("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(media),
        # fps before scale: filtering at the source resolution and then
        # discarding 35 of every 36 frames would do the expensive part 36 times
        # over.
        "-vf", f"fps={fps},scale={width}:{height},format=gray",
        "-an",                 # the audio half of this file is already indexed
        "-f", "rawvideo",
        "-",
    ]

    frame_bytes = width * height
    frames: list[NDArray[np.uint8]] = []
    # stderr to a file and Popen as a context manager, for the same two reasons
    # as extract_audio: a second pipe with no reader can deadlock, and an
    # unclosed stdout leaks a descriptor until the collector happens to run.
    with (
        tempfile.TemporaryFile("w+") as errfile,
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile) as proc,
    ):
        assert proc.stdout is not None
        while True:
            # readexactly, spelled out: a pipe read returns what is available,
            # not what was asked for, and a short read stitched onto the next
            # one as if it were a whole frame would shear every frame after it.
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            # No .copy(). frombuffer returns a read-only VIEW, which looks like
            # it wants one -- but each read() hands back a fresh bytes object
            # that the view keeps alive, so the frames do not alias each other,
            # and np.stack below copies into a fresh writable array regardless.
            # A per-frame copy here was written first and measured worthless: a
            # mutant that removed it killed no test, because there was no defect
            # for a test to catch.
            frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(height, width))
            if on_progress:
                on_progress(len(frames) / fps)
        returncode = proc.wait()
        errfile.seek(0)
        stderr = errfile.read().strip()

    if returncode != 0:
        raise MediaError(
            f"ffmpeg failed to scan {media} (exit {returncode}):\n{stderr}\n"
            f"Try `ffmpeg -i {media} -vf fps={fps},scale={width}:{height} -f null -` "
            f"by hand to see the full log."
        )
    if not frames:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.stack(frames)
