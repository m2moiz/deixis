"""Find the moments the screen changed, and write them into the transcript.

The transcript indexes what was said. This indexes what was shown: a bounded
list of timestamps where the picture moved enough to be worth a look. Nothing
here describes a frame -- a mark says "something happened at t", which is
enough to make the video navigable and costs no tokens at all.

WHY A BUDGET AND NOT A THRESHOLD. The obvious design is "mark every frame whose
change score crosses a threshold", and it was built and measured first. It does
not work, for a reason that is a property of the content rather than of the
detector: during an active working session the screen changes MOST seconds. On
the 33-minute reference recording the median one-second interval already moves
38 tiles, and a sweep of 54 threshold configurations could not find one that
marked a ticked checkbox without also marking a frame where nothing moved but
the mouse pointer. "The screen changed" is not a rare event, so it cannot be an
index -- a marker is only worth reading because it is uncommon.

Ranking sidesteps this. Score every interval, take the largest `budget` of
them, and keep them apart by `min_gap_s`. The output is bounded and useful
whatever the content: a static lecture yields well-separated marks at each
slide, a frantic screen-share yields the biggest transitions in it. No
per-video tuning, and no threshold to be wrong about.

WHY TILE COUNTS AND NOT A PERCEPTUAL HASH. Every off-the-shelf tool for this
(video-sampler, videostil, framewise, PySceneDetect's detect-hash) reduces a
frame to one 64-bit hash and compares Hamming distances. Measured on the
reference recording, a 9x8 dHash resolves 327x239-pixel cells -- coarse enough
to see an application switch and nothing smaller. It fired on pointer movement
alone and missed a ticked checkbox. Counting changed tiles on a finer grid
keeps the spatial information the hash throws away, and the count itself is the
magnitude the ranking needs; a Hamming distance is not one.

The mean-pooling ffmpeg does on the way down is load-bearing rather than
incidental. Averaging a ~23x23-pixel tile flattens smooth low-contrast motion
-- a webcam tile, a face -- while preserving the thin high-contrast edges that
text and UI chrome are made of. That is the behaviour the screen-content image
hashing literature prescribes deliberately, and here it falls out of the
downscale for free: on the reference recording the busiest rows are the browser
content, not the video-call tiles.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_DELTA",
    "DEFAULT_FPS",
    "DEFAULT_MIN_GAP_S",
    "GRID_H",
    "GRID_W",
    "Mark",
    "change_scores",
    "main",
    "mark_video",
    "select_marks",
    "with_marks",
]

import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from deixis import media as media_mod
from deixis.atomic import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

type Payload = dict[str, Any]

logger = logging.getLogger("deixis.frames")

# 128x84 over a 2940x1912 source is ~23x23-pixel tiles. Chosen by sweep, not by
# taste: 32x21 was too coarse to see a form scroll, and 128x84 costs 518s of
# decode against 205s for a 9x8 grid on the reference recording -- ffmpeg's
# scaler, not the arithmetic, which is 33ms for the whole 33 minutes.
GRID_W = 128
GRID_H = 84

# One frame per second. The reference recording is 36 fps; at 1 fps a 33-minute
# file is 1,997 samples, and the marks it produces are already closer together
# (median 9s) than any reader wants. Sampling faster buys resolution nobody has
# asked for and costs decode time linearly.
DEFAULT_FPS = 1.0

# A tile counts as changed when its mean grey level moves by more than this.
# 8/255 sits above sensor and compression noise and below a text redraw.
DEFAULT_DELTA = 8

# 150 marks is roughly one every 13 seconds of a 33-minute recording. The
# weakest of the 150 picked on the reference file was inspected by eye and is a
# real event (a form scrolled and saved), so the budget is if anything low.
DEFAULT_BUDGET = 150

# Two marks closer than this describe the same moment. Without it the ranking
# spends its whole budget inside the busiest minute of the file; with it the
# reference recording spreads 31/30/25/25/14/11 marks across its six five-minute
# buckets.
DEFAULT_MIN_GAP_S = 5.0


class Mark(NamedTuple):
    """One moment worth looking at.

    `score` is the number of tiles that changed since the previous sample. It
    is kept rather than dropped because it is the only thing distinguishing a
    window switch from a scroll, and a consumer deciding which marks to spend a
    VLM call on needs that ordering.
    """

    t: float
    score: int


def change_scores(tiles: NDArray[np.uint8], delta: int = DEFAULT_DELTA) -> NDArray[np.int32]:
    """Count changed tiles between each consecutive pair of frames.

    Args:
        tiles: An (N, h, w) array of greyscale tile means, as
            media.extract_tile_grid returns.
        delta: Grey levels a tile must move by to count as changed.

    Returns:
        An (N-1,) array; element i is the count between frames i and i+1. Empty
        when fewer than two frames were sampled, which is the honest answer for
        a file too short to have an interval in it.

    Raises:
        ValueError: if `tiles` is not three-dimensional, or `delta` is
            negative. A negative delta counts every tile as changed and would
            produce a confident, uniform, meaningless ranking.
    """
    if tiles.ndim != 3:
        raise ValueError(f"expected an (N, h, w) tile array, got shape {tiles.shape}")
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    if len(tiles) < 2:
        return np.zeros(0, dtype=np.int32)

    # int16, not the uint8 the array arrives as: an unsigned subtraction wraps,
    # so a tile going from 10 to 250 would read as a change of 16 rather than
    # 240 -- the largest changes silently becoming the smallest.
    diff = np.abs(tiles[1:].astype(np.int16) - tiles[:-1].astype(np.int16))
    return (diff > delta).sum(axis=(1, 2)).astype(np.int32)


def select_marks(
    scores: NDArray[np.int32],
    *,
    fps: float = DEFAULT_FPS,
    budget: int = DEFAULT_BUDGET,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
) -> list[Mark]:
    """Take the highest-scoring intervals, keeping them `min_gap_s` apart.

    Greedy rather than optimal: walk the scores from largest down, keep one if
    nothing already kept sits within `min_gap_s` of it, stop at `budget`. The
    optimal spacing-constrained subset is a different and much slower problem,
    and the difference between them is not visible in an index a human skims.

    Args:
        scores: Per-interval change counts from change_scores.
        fps: The rate the frames were sampled at, which turns an index into a
            timestamp.
        budget: The most marks to return.
        min_gap_s: Seconds two marks must be apart.

    Returns:
        Marks in ascending time order -- the order a reader wants, not the
        ranked order the selection happened in. Fewer than `budget` when the
        gap constraint or the supply of non-zero scores runs out first.

    Raises:
        ValueError: if `fps` is not positive, `budget` is negative, or
            `min_gap_s` is negative.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if min_gap_s < 0:
        raise ValueError(f"min_gap_s must be non-negative, got {min_gap_s}")

    # Stable, so equal scores are taken earliest-first and two runs over the
    # same file cannot disagree. numpy's default quicksort is not stable and
    # would make the output depend on the array's memory layout.
    order = np.argsort(-scores, kind="stable")

    picked: list[int] = []
    for i in order:
        if len(picked) >= budget:
            break
        # A zero score is a frame identical to its predecessor. Padding the
        # budget with those would put marks on nothing at all, which is worse
        # than returning fewer.
        if scores[i] <= 0:
            break
        idx = int(i)
        if all(abs(idx - p) * (1.0 / fps) >= min_gap_s for p in picked):
            picked.append(idx)

    # i indexes the interval between frames i and i+1, so the change is visible
    # at frame i+1 and that is the timestamp worth seeking to.
    return sorted(Mark(t=(i + 1) / fps, score=int(scores[i])) for i in picked)


def with_marks(payload: Payload, marks: list[Mark], meta: dict[str, Any]) -> Payload:
    """Return `payload` with visual marks added, leaving the original alone.

    A copy rather than a mutation: the caller's transcript is on disk and may
    be read again, and a function that edits its argument in place makes a
    partial failure halfway through unrecoverable.

    `marks` lands as a sibling of `sentences` rather than being distributed
    into them. A mark is a fact about the video at a time, not about a
    sentence; nesting it inside whichever sentence happens to span that second
    would invent a relationship the detector never observed.
    """
    out = dict(payload)
    out["marks"] = [{"t": round(m.t, 3), "score": m.score} for m in marks]
    # The parameters travel with the result. Marks from a budget of 150 and
    # marks from a budget of 20 are different documents, and a reader with only
    # the list cannot tell which one they have.
    out["marks_meta"] = meta
    return out


def mark_video(
    media: Path,
    transcript: Path,
    out: Path,
    *,
    fps: float = DEFAULT_FPS,
    delta: int = DEFAULT_DELTA,
    budget: int = DEFAULT_BUDGET,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    on_progress: Callable[[float], None] | None = None,
) -> Payload:
    """Decode `media`, rank its changes, and write `transcript` + marks to `out`.

    A separate pass over the video rather than a step inside transcribe(). The
    transcript is worth having on its own and arrives at ~35x realtime; this
    decodes every sampled frame and runs at ~4x. Coupling them would make the
    fast half wait for the slow one.

    Raises:
        FileNotFoundError: if `transcript` does not exist. Marks index a
            transcript; there is nothing to add them to otherwise.
    """
    if not transcript.exists():
        raise FileNotFoundError(transcript)
    payload: Payload = json.loads(transcript.read_text())

    tiles = media_mod.extract_tile_grid(
        media, fps=fps, width=GRID_W, height=GRID_H, on_progress=on_progress
    )
    scores = change_scores(tiles, delta=delta)
    marks = select_marks(scores, fps=fps, budget=budget, min_gap_s=min_gap_s)

    result = with_marks(
        payload,
        marks,
        {
            "source": str(media),
            "fps": fps,
            "grid": [GRID_W, GRID_H],
            "delta": delta,
            "budget": budget,
            "min_gap_s": min_gap_s,
            "frames_sampled": len(tiles),
        },
    )
    atomic_write_text(out, json.dumps(result))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the CLI: add visual change marks to an existing transcript.

    Returns:
        A process exit code -- 0 on success.
    """
    import argparse

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    ap = argparse.ArgumentParser(
        description="Mark the moments a recording's picture changed most."
    )
    ap.add_argument("media", type=Path, help="the video the transcript indexes")
    ap.add_argument("-t", "--transcript", type=Path, required=True)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        help="where to write transcript+marks; defaults to overwriting --transcript",
    )
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--delta", type=int, default=DEFAULT_DELTA)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP_S)
    args = ap.parse_args(argv)

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
        args.media,
        args.transcript,
        args.out or args.transcript,
        fps=args.fps,
        delta=args.delta,
        budget=args.budget,
        min_gap_s=args.min_gap,
        on_progress=show,
    )
    if tty:
        print(file=sys.stderr)

    marks: list[dict[str, Any]] = result["marks"]
    scanned = result["marks_meta"]["frames_sampled"]
    # A count and the span it covers, not a rate: "150 marks" alone reads as a
    # lot or a little depending on a duration the reader has to go and find.
    logger.info(
        "%d marks over %d sampled frames in %.0fs",
        len(marks),
        scanned,
        time.monotonic() - started,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
