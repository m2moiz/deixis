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
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

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


def render_bar(p: Progress, width: int = 24) -> str:
    filled = int(p.fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {p.fraction * 100:3.0f}%  "
        f"{_clock(p.audio_done_s)}/{_clock(p.audio_total_s)} audio  "
        f"elapsed {_clock(p.elapsed_s)}  eta {_clock(p.eta_s)}  {p.speed:.1f}x"
    )


def transcribe(
    audio: Path,
    out: Path,
    model_id: str = DEFAULT_MODEL,
    status_path: Path | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> dict:
    """Transcribe `audio`, writing a sentence+token timestamped JSON to `out`.

    Returns the parsed result. `status_path` receives a JSON heartbeat on every
    chunk so a detached run stays observable.
    """
    from parakeet_mlx import from_pretrained

    started = time.monotonic()
    model = from_pretrained(model_id)

    def write_status(p: Progress, state: str) -> None:
        if status_path is None:
            return
        payload = asdict(p) | {
            "state": state,
            "fraction": round(p.fraction, 4),
            "speed": round(p.speed, 2),
            "eta_s": p.eta_s,
        }
        status_path.write_text(json.dumps(payload))

    # chunk_callback reports SAMPLES, not seconds -- the upstream CLI only ever
    # feeds them to a ratio, so the units never mattered there. Read the rate off
    # the model rather than assuming 16kHz.
    rate = model.preprocessor_config.sample_rate

    def chunk_callback(current: float, full: float) -> None:
        p = Progress(current / rate, full / rate, time.monotonic() - started)
        write_status(p, "running")
        if on_progress:
            on_progress(p)

    result = model.transcribe(
        audio,
        chunk_duration=CHUNK_S,
        overlap_duration=OVERLAP_S,
        chunk_callback=chunk_callback,
    )

    payload = {
        "audio": str(audio),
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
    write_status(Progress(total, total, elapsed), "done")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe media to a timestamped index.")
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--status", type=Path, help="JSON heartbeat file for detached runs")
    args = ap.parse_args(argv)

    # \r only rewrites the line on a terminal; when piped, print one line per
    # chunk instead so a captured log stays readable rather than becoming one
    # enormous line of control characters.
    tty = sys.stderr.isatty()

    def show(p: Progress) -> None:
        end = "\r" if tty else "\n"
        print(render_bar(p), end=end, file=sys.stderr, flush=True)

    started = time.monotonic()
    result = transcribe(
        args.audio, args.out, args.model, status_path=args.status, on_progress=show
    )
    if tty:
        print(file=sys.stderr)
    elapsed = time.monotonic() - started
    total = result["sentences"][-1]["end"] if result["sentences"] else 0.0
    print(
        f"done: {_clock(total)} audio in {_clock(elapsed)} "
        f"({total / elapsed:.1f}x realtime) -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
