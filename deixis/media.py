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

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    """ffmpeg could not do what we asked of this file."""


class FFmpegNotFound(MediaError):
    pass


class NoAudioStream(MediaError):
    pass


@dataclass(frozen=True)
class AudioStream:
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

    info = json.loads(proc.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise NoAudioStream(
            f"{media} has no audio stream, so there is nothing to transcribe. "
            f"If this is a silent screen capture, re-record with audio enabled."
        )

    stream = streams[0]
    # This field describes the audio stream, so the stream's own duration wins.
    # The container's is the max across every stream, and a recorder that keeps
    # rolling video after the mic stops gives a container longer than its audio
    # -- an extraction bar denominated in that never reaches 100%. Fall back to
    # the container only when the stream carries no duration of its own, as some
    # do not. Zero means ffprobe genuinely does not know; progress then reports
    # elapsed only, which Progress already handles.
    duration = stream.get("duration") or info.get("format", {}).get("duration") or 0.0
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
    with tempfile.TemporaryFile("w+") as errfile:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=errfile, text=True
        )
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
