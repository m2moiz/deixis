"""One command, three verbs -- the surface an agent actually reaches for.

Until this file existed, the only way to pull a frame out of a video was to
import `deixis.media` from Python. That is a real gap and not a cosmetic one:
frame retrieval is the step measured to take an agent from 5/16 to 16/16 on
questions about a recording (docs/do-marks-help.md), and the agent that scored
16/16 only managed it because a `uv run python -c ...` snippet was hand-written
into its prompt. A capability nothing can invoke is a capability nobody has.

    deixis transcribe recording.mov -o transcript.json
    deixis mark       recording.mov -t transcript.json
    deixis frame      recording.mov 431.5 -o frame.jpg

`transcribe` and `mark` delegate to the module CLIs that already existed, so
`python -m deixis.transcribe` and `python -m deixis.frames` keep working exactly
as before and there is one implementation of each, not two.

`frame` prints the path it wrote to stdout and nothing else. That is deliberate:
the caller is usually a program, and a path on stdout composes.
"""

from __future__ import annotations

__all__ = ["main"]

import argparse
import sys
from pathlib import Path

USAGE = """deixis <command> [options]

  transcribe VIDEO -o OUT.json      what was said, when
  mark       VIDEO -t TRANSCRIPT    when the picture changed
  frame      VIDEO SECONDS -o IMG   the picture at that moment

Run `deixis <command> --help` for the options of one command."""


def _frame(argv: list[str]) -> int:
    """Write one frame to a file and print where it went."""
    from deixis.media import extract_frame

    ap = argparse.ArgumentParser(
        prog="deixis frame",
        description="Write the frame at a given second of a video to an image file.",
    )
    ap.add_argument("video", type=Path)
    ap.add_argument("seconds", type=float, help="offset into the recording")
    ap.add_argument("-o", "--out", type=Path, required=True, help=".jpg for a vision model")
    ap.add_argument(
        "--width",
        type=int,
        # 1500px is not a default so much as a measured ceiling: on the
        # reference recording a full 2940px frame is ~776 KB as a JPEG, over
        # what most vision APIs want, and legibility stopped improving well
        # before that (docs/vlm-legibility.md -- 700 -> 1600px moved recall by
        # one string out of fifteen). None keeps the source resolution.
        default=1500,
        help="scale to this width, preserving aspect (default 1500; 0 keeps source)",
    )
    args = ap.parse_args(argv)

    dest = extract_frame(
        args.video, args.seconds, args.out, width=args.width or None
    )
    print(dest)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to a subcommand.

    Returns:
        A process exit code -- 2 for an unknown or missing command, otherwise
        whatever the subcommand returns.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        # 0 for an explicit --help, 2 for a bare invocation: a caller that
        # forgot the verb has made an error, and a shell script testing the
        # exit code should see one.
        return 0 if args else 2

    command, rest = args[0], args[1:]
    if command == "frame":
        return _frame(rest)
    if command == "transcribe":
        from deixis.transcribe import main as transcribe_main

        return transcribe_main(rest)
    if command == "mark":
        from deixis.frames import main as frames_main

        return frames_main(rest)

    print(f"deixis: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
