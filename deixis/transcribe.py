"""Transcribe a media file to a timestamped index, with live progress.

The transcript is deixis' index into the video, so this is the one step that
must not fail quietly. Progress is reported two ways at once: a live line on
the terminal, and a JSON status file that a detached run can be inspected
through. Background jobs are the normal case here -- an hour of audio is not
something you sit and watch -- and a run you cannot inspect is a run you cannot
trust.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

# Imported as media_mod because the parameter it serves is named `media` and
# would shadow the module inside the function body.
from deixis import media as media_mod
from deixis.atomic import atomic_write_text

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# parakeet-mlx defaults chunk_duration to None, which feeds the whole file to
# Metal in one buffer. An hour of audio asks for ~14.5GB against a ~9.5GB max
# buffer and dies. Chunking is not optional at meeting length -- and it is also
# what makes chunk_callback fire, so progress reporting depends on it too.
CHUNK_S = 120.0
OVERLAP_S = 15.0


@dataclass
class Progress:
    audio_done_s: float
    audio_total_s: float
    elapsed_s: float

    @property
    def fraction(self) -> float:
        return self.audio_done_s / self.audio_total_s if self.audio_total_s else 0.0

    @property
    def speed(self) -> float:
        """Realtime multiple: seconds of audio per second of wall clock."""
        return self.audio_done_s / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def eta_s(self) -> float | None:
        if self.speed <= 0:
            return None
        return (self.audio_total_s - self.audio_done_s) / self.speed


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def render_bar(p: Progress, state: str, width: int = 24) -> str:
    filled = int(p.fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"{state:>10} [{bar}] {p.fraction * 100:3.0f}%  "
        f"{_clock(p.audio_done_s)}/{_clock(p.audio_total_s)} audio  "
        f"elapsed {_clock(p.elapsed_s)}  eta {_clock(p.eta_s)}  {p.speed:.1f}x"
    )


def _make_chunk_callback(
    rate: float,
    clock: Callable[[], float],
    emit: Callable[[Progress], None],
) -> Callable[[float, float], None]:
    """Build the callback parakeet-mlx fires once per chunk.

    `current` and `full` arrive in SAMPLES, not seconds -- the upstream CLI only
    ever feeds them to a ratio, so the units never mattered there. Dividing by
    `rate` is the whole point of this function, and a ratio-only assertion
    cannot see whether it happened, because the units cancel.
    """

    def chunk_callback(current: float, full: float) -> None:
        emit(Progress(current / rate, full / rate, clock()))

    return chunk_callback


def transcribe(
    media: Path,
    out: Path,
    model_id: str = DEFAULT_MODEL,
    status_path: Path | None = None,
    on_progress: Callable[[Progress, str], None] | None = None,
) -> dict:
    """Transcribe `media`, writing a sentence+token timestamped JSON to `out`.

    `media` is any file ffmpeg can open -- the .mov straight off the screen
    recorder is the normal case. Audio is extracted to a temp file first,
    unless the input is already in the shape the model consumes.

    Returns the parsed result. `status_path` receives a JSON heartbeat during
    both phases so a detached run stays observable.
    """
    from parakeet_mlx import from_pretrained

    started = time.monotonic()
    # Load first: the extraction target rate is a property of this model's
    # preprocessor, not a constant we get to assume.
    model = from_pretrained(model_id)
    rate = model.preprocessor_config.sample_rate

    def write_status(p: Progress, state: str) -> None:
        if status_path is None:
            return
        payload = asdict(p) | {
            "state": state,
            "fraction": round(p.fraction, 4),
            "speed": round(p.speed, 2),
            "eta_s": p.eta_s,
        }
        # Not fsynced: a reader is protected by the rename alone, and a
        # heartbeat lost to a power cut costs nothing to regenerate.
        atomic_write_text(status_path, json.dumps(payload))

    def report(p: Progress, state: str) -> None:
        write_status(p, state)
        if on_progress:
            on_progress(p, state)

    stream = media_mod.probe(media)

    with tempfile.TemporaryDirectory(prefix="deixis-") as tmp:
        if media_mod.needs_conversion(stream, rate):
            # Per-phase clock. Extraction runs three orders of magnitude faster
            # than realtime and the transcription that follows around 20x;
            # sharing one elapsed would make both speeds meaningless.
            extract_started = time.monotonic()

            def on_extract(done_s: float) -> None:
                report(
                    Progress(done_s, stream.duration_s, time.monotonic() - extract_started),
                    "extracting",
                )

            audio = media_mod.extract_audio(
                media, Path(tmp) / "audio.wav", rate, on_progress=on_extract
            )
        else:
            audio = media

        # Per-phase clock, as above: transcription runs at a different order of
        # magnitude from extraction, so they cannot share an elapsed.
        transcribe_started = time.monotonic()
        chunk_callback = _make_chunk_callback(
            rate,
            lambda: time.monotonic() - transcribe_started,
            lambda p: report(p, "running"),
        )

        result = model.transcribe(
            audio,
            chunk_duration=CHUNK_S,
            overlap_duration=OVERLAP_S,
            chunk_callback=chunk_callback,
        )

        payload = {
            # The source the user handed us, never the temp wav -- this JSON is
            # an index into that file and has to keep pointing at it.
            "audio": str(media),
            "model": model_id,
            "text": result.text,
            "sentences": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "tokens": [{"t": t.start, "w": t.text} for t in s.tokens],
                }
                for s in result.sentences
            ],
        }
        out.write_text(json.dumps(payload))

        elapsed = time.monotonic() - started
        total = result.sentences[-1].end if result.sentences else 0.0
        report(Progress(total, total, elapsed), "done")
        return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe media to a timestamped index.")
    ap.add_argument("media", type=Path, help="video or audio file; a .mov is the normal case")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--status", type=Path, help="JSON heartbeat file for detached runs")
    args = ap.parse_args(argv)

    # \r only rewrites the line on a terminal; when piped, print one line per
    # chunk instead so a captured log stays readable rather than becoming one
    # enormous line of control characters.
    tty = sys.stderr.isatty()
    last_state = ""

    def show(p: Progress, state: str) -> None:
        # A phase change ends the rewritten line, so the finished extraction bar
        # stays on screen instead of being overwritten by transcription's 0%.
        nonlocal last_state
        if tty and last_state and state != last_state:
            print(file=sys.stderr)
        last_state = state
        end = "\r" if tty else "\n"
        print(render_bar(p, state), end=end, file=sys.stderr, flush=True)

    started = time.monotonic()
    try:
        result = transcribe(
            args.media, args.out, args.model, status_path=args.status, on_progress=show
        )
    except Exception as exc:
        # Record and re-raise: a detached watcher polling the heartbeat has no
        # other way to distinguish "died" from "not started yet". The traceback
        # still reaches the terminal untouched.
        if args.status:
            # Atomic for the same reason as the heartbeat, and more so: the
            # watcher polling for exactly this document is in a tight read loop,
            # which makes it the reader most likely to land inside a torn write.
            atomic_write_text(
                args.status,
                json.dumps({"state": "failed", "error": f"{type(exc).__name__}: {exc}"}),
            )
        raise
    if tty:
        print(file=sys.stderr)
    elapsed = time.monotonic() - started
    total = result["sentences"][-1]["end"] if result["sentences"] else 0.0
    # Progress.speed carries the zero-elapsed guard; recomputing the division
    # here is what left this line unguarded in the first place. `--x` matches
    # _clock's `--:--` for an unknown quantity -- `0.0x` would read as "slower
    # than realtime", the opposite of the truth.
    speed = Progress(total, total, elapsed).speed
    rate = f"{speed:.1f}x" if speed else "--x"
    print(
        f"done: {_clock(total)} audio in {_clock(elapsed)} "
        f"({rate} realtime) -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
