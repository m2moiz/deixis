"""Score the mark scheme on any video, against a random baseline.

The repeatable test. Everything deixis knows about its own change detection came
from one 33-minute recording, which is n=1. This harness runs the shipped code
on an arbitrary video and reports the two things that decide whether the marks
are worth having -- with no ground truth needed, because both metrics are
self-supervised on the video's own pixels:

  FRAME QUALITY   How settled is the frame you would send to a vision model?
                  Measured as the fraction of tiles differing from the frames
                  either side of it. A mark sits on a transition by definition,
                  so `t` should score badly and `look` should score well. If
                  `look` is not clearly better than `t`, the midpoint idea is
                  wrong for this video.

  COVERAGE        Pick any second of the video. Find its segment by the mark
                  boundaries, take that segment's `look` frame -- does it show
                  the same screen? This is what a consumer enumerating a video
                  actually does.

Both are compared against N random draws of the same number of timestamps.
Random is the baseline that matters (Principles of Visual Tokens, ICCV 2025:
random sampling matches or beats most sophisticated frame selection), and it is
the baseline this project skipped once already.

Usage:
  python scratch/mark_eval.py <video> [--fps 1] [--per-min 4.5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deixis.frames import GRID_H, GRID_W, change_scores, select_marks
from deixis.media import extract_tile_grid

DELTA = 8
RANDOM_DRAWS = 30


def instability(tiles: np.ndarray, i: float) -> float:
    """How much the frame at index i differs from the frames either side of it."""
    j = max(1, min(int(i), len(tiles) - 2))
    back = (np.abs(tiles[j] - tiles[j - 1]) > DELTA).mean()
    fwd = (np.abs(tiles[j + 1] - tiles[j]) > DELTA).mean()
    return float((back + fwd) / 2)


def same_screen(tiles: np.ndarray, a: float, b: float) -> float:
    """Fraction of tiles unchanged between the frames at a and b."""
    n = len(tiles) - 1
    return float((np.abs(tiles[min(int(a), n)] - tiles[min(int(b), n)]) <= DELTA).mean())


def coverage(tiles: np.ndarray, boundaries: np.ndarray, samples: np.ndarray) -> float:
    """Average same-screen score over every second, via each second's segment.

    Boundaries and samples are separate on purpose. Finding which segment a
    second belongs to is the boundaries' job; representing that segment with a
    frame is the samples' job. Conflating them scores midpoints as if they were
    boundaries and makes a working scheme look worse than random -- which it
    did, the first time this was measured.
    """
    return float(
        np.mean([
            same_screen(tiles, samples[max(0, int(np.searchsorted(boundaries, t, "right")) - 1)], t)
            for t in range(1, len(tiles) - 1)
        ])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--per-min", type=float, default=4.5, help="marks per minute of video")
    ap.add_argument("--min-gap", type=float, default=5.0)
    ap.add_argument("--json", help="append the result to this JSON-lines file")
    args = ap.parse_args()

    name = Path(args.video).name
    t0 = time.monotonic()
    tiles = extract_tile_grid(Path(args.video), fps=args.fps, width=GRID_W, height=GRID_H)
    decode_s = time.monotonic() - t0
    minutes = len(tiles) / (args.fps * 60.0)
    budget = max(2, round(args.per_min * minutes))

    scores = change_scores(tiles)
    marks = select_marks(scores, fps=args.fps, budget=budget, min_gap_s=args.min_gap)
    if len(marks) < 2:
        print(f"{name}: only {len(marks)} marks -- too short or too static to score")
        return 0

    ti = tiles.astype(np.int16)
    starts = np.array([m.t for m in marks])
    looks = np.array([m.look for m in marks])

    at_t = float(np.mean([instability(ti, x) for x in starts]))
    at_look = float(np.mean([instability(ti, x) for x in looks]))
    cov_t = coverage(ti, starts, starts)
    cov_look = coverage(ti, starts, looks)

    rng = np.random.default_rng(0)
    hi = max(2, len(tiles) - 1)
    n = min(len(marks), hi - 1)
    rand_inst: list[float] = []
    rand_cov: list[float] = []
    for _ in range(RANDOM_DRAWS):
        r = np.sort(rng.choice(np.arange(1, hi), n, replace=False)).astype(float)
        rand_inst.append(np.mean([instability(ti, x) for x in r]))
        rand_cov.append(coverage(ti, r, r))

    res: dict[str, Any] = {
        "video": name,
        "duration_s": round(len(tiles) / args.fps, 1),
        "frames": len(tiles),
        "decode_s": round(decode_s, 1),
        "realtime_x": round((len(tiles) / args.fps) / decode_s, 1),
        "marks": len(marks),
        "median_score": int(np.median(scores)),
        "instability": {
            "at_t": round(at_t, 4),
            "at_look": round(at_look, 4),
            "random": round(float(np.mean(rand_inst)), 4),
            "look_vs_t": round(at_t / at_look, 2) if at_look else None,
        },
        "coverage": {
            "at_t": round(cov_t, 4),
            "at_look": round(cov_look, 4),
            "random_mean": round(float(np.mean(rand_cov)), 4),
            "random_std": round(float(np.std(rand_cov)), 4),
            "sigma_above_random": (
                round((cov_look - float(np.mean(rand_cov))) / float(np.std(rand_cov)), 1)
                if np.std(rand_cov) > 0 else None
            ),
        },
    }
    print(json.dumps(res, indent=1))
    if args.json:
        with Path(args.json).open("a") as fh:
            fh.write(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
