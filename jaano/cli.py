"""The `jaano` command -- one Typer app, three verbs.

    jaano suno   recording.mov -o transcript.json    # listen
    jaano dekho  recording.mov -t transcript.json    # look
    jaano dikhao recording.mov 431.5 -o frame.jpg    # show me

Urdu imperatives, and they are not decoration: they name the three things the
tool does in the order it does them. Suno gives you what was said, dekho gives
you when the picture changed, dikhao gives you the picture itself. Jaano -- know
-- is what you get from all three, which is why it is the command.

This file owns ALL argument parsing for the project. `jaano.suno.main`
and `jaano.dekho.main` are thin shims onto the commands below, so
`python -m jaano.suno` keeps working and there is exactly one definition
of every flag rather than one per entry point.

WHY standalone_mode, AND WHY THE SystemExit CATCH. Typer's other calling
convention, `app(args=..., standalone_mode=False)`, returns the command's value
directly and looks like the obvious fit for `main(argv) -> int`. It is a trap
here for two measured reasons:

  * Typer 0.27 VENDORS its own Click at `typer._click`. Its `UsageError` is not
    the `click.exceptions.UsageError` class, so the obvious `except
    click.exceptions.UsageError` never fires and a bad flag escapes as an
    unhandled NoSuchOption traceback. Catching it properly means importing from
    a private module.
  * standalone_mode=False also skips Click's error rendering, so every usage
    message would have to be reimplemented here.

standalone_mode=True renders errors the way every other Click program does and
raises SystemExit with the right code; catching that gives back the int. Checked
against all six cases that matter: success 0, --help 0, unknown flag 2, no args
2, a caller's own exception PROPAGATES (which the resume tests depend on), and
KeyboardInterrupt becomes 130.
"""

from __future__ import annotations

__all__ = ["app", "main", "run"]

import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

# Module level, not lazy. These are DEFAULTS, and a lazily-resolved default
# cannot appear in --help: the first version of this file used 0 as a "not
# given" sentinel and Typer duly advertised `[default: 0]` for a budget whose
# real default is 150. Measured cost of the import: ~90ms, inside the noise of
# `jaano --help` at 120-200ms. The heavy workers below stay lazy.
from jaano.dekho import DEFAULT_BUDGET, DEFAULT_DELTA, DEFAULT_FPS, DEFAULT_MIN_GAP_S
from jaano.suno import DEFAULT_MODEL, DEFAULT_WHISPER_MODEL, ENGINES

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Tracebacks are the project's failure surface -- media.py raises
    # MediaError with the ffmpeg log and a command to reproduce it by hand.
    # Typer's pretty exceptions reformat that into a box and truncate the
    # frames, so a real diagnosis gets prettier and less useful.
    pretty_exceptions_enable=False,
    help="Make a long screen recording answerable.",
)


def _stderr_logger(name: str) -> None:
    """Attach the one stderr handler a CLI is allowed to install.

    A bare handler rather than basicConfig: basicConfig is a no-op once the root
    logger has handlers, which is exactly the case under pytest, and a gate that
    silently does nothing is the failure this project exists to remove.
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@app.command("suno")
def suno(
    media: Annotated[Path, typer.Argument(help="video or audio file; a .mov is the normal case")],
    out: Annotated[Path, typer.Option("--out", "-o", help="where the transcript JSON goes")],
    model: Annotated[
        str | None,
        typer.Option("--model", help=f"the ASR model (default: {DEFAULT_MODEL} per --engine)"),
    ] = None,
    engine: Annotated[
        str, typer.Option("--engine", help=f"ASR backend: {' | '.join(ENGINES)}")
    ] = "parakeet",
    language: Annotated[
        str | None,
        typer.Option("--language", help="whisper only: ISO code, e.g. ur. Detected if omitted"),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="whisper only: seeds the decoder; biases spelling+script"),
    ] = None,
    roman_urdu: Annotated[
        bool,
        typer.Option(
            "--roman-urdu",
            help="whisper only: Urdu, written in Latin. Sets --language ur and a measured --prompt",
        ),
    ] = False,
    status: Annotated[
        Path | None, typer.Option("--status", help="JSON heartbeat file for detached runs")
    ] = None,
    no_resume: Annotated[
        bool, typer.Option("--no-resume", help="ignore any checkpoint and start over")
    ] = False,
    no_diarize: Annotated[
        bool, typer.Option("--no-diarize", help="skip speaker labelling")
    ] = False,
    require_diarize: Annotated[
        bool,
        typer.Option("--require-diarize", help="fail rather than degrade if labelling cannot run"),
    ] = False,
) -> int:
    """Listen: transcribe media to a timestamped index."""
    from jaano.atomic import atomic_write_text
    from jaano.suno import Progress, clock, render_bar
    from jaano.suno import transcribe as run_transcribe
    from jaano.whisper import ROMAN_URDU_PROMPT

    _stderr_logger("jaano.suno")

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
        print(render_bar(p, state), end="\r" if tty else "\n", file=sys.stderr, flush=True)

    # --roman-urdu is sugar over the two flags under it, and it is spelled as
    # sugar rather than as a mode so that an explicit --language or --prompt
    # beside it still wins. The prompt it sets is measured, not invented: see
    # jaano/whisper.py.
    if roman_urdu:
        engine = "whisper" if engine == "parakeet" else engine
        language = language or "ur"
        prompt = prompt or ROMAN_URDU_PROMPT

    # Resolved here, not in transcribe(): this file owns every default in the
    # project, and a default that lives in two places is a default that will
    # disagree with `--help` eventually.
    model = model or (DEFAULT_WHISPER_MODEL if engine == "whisper" else DEFAULT_MODEL)

    started = time.monotonic()
    try:
        result = run_transcribe(
            media,
            out,
            model,
            status_path=status,
            on_progress=show,
            resume=not no_resume,
            diarize=not no_diarize,
            require_diarize=require_diarize,
            engine=engine,
            language=language,
            prompt=prompt,
        )
    except Exception as exc:
        # Record and re-raise: a detached watcher polling the heartbeat has no
        # other way to distinguish "died" from "not started yet". The traceback
        # still reaches the terminal untouched.
        if status:
            # Atomic for the same reason as the heartbeat, and more so: the
            # watcher polling for exactly this document is in a tight read loop,
            # which makes it the reader most likely to land inside a torn write.
            atomic_write_text(
                status, json.dumps({"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
            )
        raise
    if tty:
        print(file=sys.stderr)
    elapsed = time.monotonic() - started
    total = result["sentences"][-1]["end"] if result["sentences"] else 0.0
    # A count, not a rate. `total / elapsed` would credit a resumed run with
    # work a previous process paid for -- an hour finished in two minutes reads
    # as 30x. Guarded on the key because a degraded run has no speakers.
    speakers = f" · {len(result['speakers'])} speakers" if "speakers" in result else ""
    print(f"done: {clock(total)} audio in {clock(elapsed)}{speakers} -> {out}", file=sys.stderr)
    return 0


@app.command("dekho")
def dekho(
    media: Annotated[Path, typer.Argument(help="the video the transcript indexes")],
    transcript: Annotated[
        Path, typer.Option("--transcript", "-t", help="transcript to add marks to")
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="where to write; defaults to overwriting --transcript"),
    ] = None,
    fps: Annotated[
        float, typer.Option(help="frames sampled per second of video")
    ] = DEFAULT_FPS,
    delta: Annotated[
        int, typer.Option(help="grey levels a tile must move to count")
    ] = DEFAULT_DELTA,
    budget: Annotated[int, typer.Option(help="how many marks to keep")] = DEFAULT_BUDGET,
    min_gap: Annotated[
        float, typer.Option("--min-gap", help="seconds two marks must be apart")
    ] = DEFAULT_MIN_GAP_S,
) -> int:
    """Look: mark the moments the picture changed most."""
    from jaano.dekho import logger, mark_video

    _stderr_logger("jaano.dekho")
    tty = sys.stderr.isatty()
    started = time.monotonic()

    def show(done_s: float) -> None:
        elapsed = time.monotonic() - started
        speed = done_s / elapsed if elapsed > 0 else 0.0
        print(
            f"scanning {done_s:7.0f}s of video  {speed:4.1f}x",
            end="\r" if tty else "\n",
            file=sys.stderr,
            flush=True,
        )

    result = mark_video(
        media,
        transcript,
        out or transcript,
        fps=fps,
        delta=delta,
        budget=budget,
        min_gap_s=min_gap,
        on_progress=show,
    )
    if tty:
        print(file=sys.stderr)
    logger.info(
        "%d marks over %d sampled frames in %.0fs",
        len(result["marks"]),
        result["marks_meta"]["frames_sampled"],
        time.monotonic() - started,
    )
    return 0


@app.command("dikhao")
def dikhao(
    video: Annotated[Path, typer.Argument(help="the recording to seek into")],
    seconds: Annotated[float, typer.Argument(help="offset into the recording")],
    out: Annotated[Path, typer.Option("--out", "-o", help=".jpg is what a vision model wants")],
    width: Annotated[
        int,
        typer.Option(
            help="scale to this width, aspect preserved; 0 keeps the source resolution"
        ),
        # 1500 is a measured ceiling, not a taste: a full 2940px frame is
        # ~776 KB as a JPEG, over what most vision APIs accept, and legibility
        # stopped improving well below it -- 700px to 1600px moved recall by one
        # string in fifteen (docs/vlm-legibility.md).
    ] = 1500,
) -> int:
    """Show me: write the frame at a given second to an image file."""
    from jaano.media import extract_frame

    dest = extract_frame(video, seconds, out, width=width or None)
    # The path alone on stdout, and nothing else. The caller is usually a
    # program, and a path on stdout composes:
    #     open "$(jaano frame rec.mov 431.5 -o /tmp/f.jpg)"
    print(dest)
    return 0


def run(argv: list[str]) -> int:
    """Invoke the app on `argv` and return an exit code instead of exiting.

    The one place SystemExit is converted back into a value, so every
    `main(argv) -> int` in the package can share it. Exceptions other than
    SystemExit propagate untouched -- a failing transcription must still reach
    the caller as itself, not as an exit code.
    """
    try:
        app(args=argv, standalone_mode=True)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `jaano` console script.

    Returns:
        A process exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    # `jaano help` as well as `jaano --help`. Typer offers no `help` command
    # and the bare word is what people type.
    if args and args[0] == "help":
        args = ["--help"]
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
