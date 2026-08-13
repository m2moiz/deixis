"""Score dsj' marks against GUI-World's human-annotated keyframes.

The external benchmark. Every other number about this detector was measured on
recordings from one laptop; GUI-World (ICLR 2025, CC BY 4.0) is 496 desktop
screen recordings with human-placed keyframes marking the moments a user action
changed the screen. That is ground truth nobody here produced.

The question: given the SAME number of timestamps a human annotator placed, do
our change-ranked marks land nearer those moments than random timestamps do?
Random is the baseline that matters and the one this project skipped once.

Budget is set to exactly the number of human keyframes in that video, so the
comparison is like-for-like rather than "who emits more guesses".

A note on --fps, because the first version of this file asserted the opposite
and was wrong. These clips are short (median ~15s), so it looked obvious that
1 fps would be too coarse -- ~15 samples, quantisation error comparable to the
inter-event spacing -- and the default here was 4. Measured across 1/2/4/8 fps,
1 fps is the BEST setting and accuracy degrades monotonically as sampling gets
finer:

    fps=1  ours 0.79s vs random 1.36s   37/38 videos
    fps=2  ours 0.84s vs random 1.31s   34/39
    fps=4  ours 1.00s vs random 1.31s   30/39
    fps=8  ours 1.11s vs random 1.31s   28/39

The likely mechanism, untested: with more samples per transition, several marks
land inside one change event and the fixed budget is spent twice on the same
moment. Coarse sampling forces them apart. Whatever the cause, the shipped 1 fps
default is not a compromise made for decode cost -- it is the better choice on
this benchmark.

Usage: python scratch/guiworld_eval.py [--fps 4] [--limit 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsj.dekho import GRID_H, GRID_W, change_scores, select_marks
from dsj.media import extract_tile_grid

ROOT = Path(__file__).resolve().parent / "guiworld"


def annotations(split: str) -> Path:
    return ROOT / "Annotation" / "benchmark" / f"{split}.jsonl"


def video_fps(path: Path) -> float | None:
    """The real frame rate, probed, or None if the file will not open.

    None rather than a raise: one of the 40 files downloaded from the hub is
    truncated (120 KB, "moov atom not found"). Skipping it and SAYING SO is
    right; crashing the sweep on it, or silently dropping it, are both wrong.
    """
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    num, _, den = proc.stdout.strip().partition("/")
    return float(num) / float(den or 1)


def nearest(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Distance in seconds from each target to the closest point."""
    return np.array([np.abs(points - t).min() for t in targets])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--split", default="software", help="software | website | multi")
    ap.add_argument("--quiet", action="store_true", help="totals only, no per-video rows")
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    rows = [json.loads(line) for line in annotations(args.split).open()]
    rng = np.random.default_rng(0)
    ours: list[float] = []
    rand: list[float] = []
    per_video: list[dict[str, float | str | int]] = []
    skipped: list[str] = []

    for row in rows:
        path = ROOT / row["video_path"]
        if not path.exists():
            continue
        if len(per_video) >= args.limit:
            break

        native = video_fps(path)
        if native is None or native <= 0:
            skipped.append(row["video_path"])
            continue
        gt = np.array(sorted(k["frame"] / native for k in row["keyframes"]))
        tiles = extract_tile_grid(path, fps=args.fps, width=GRID_W, height=GRID_H)
        duration = len(tiles) / args.fps
        # A keyframe past the decoded end is an annotation we cannot be scored
        # on; drop it rather than let it inflate both sides equally.
        gt = gt[gt <= duration]
        if len(gt) < 2 or len(tiles) < 4:
            continue

        scores = change_scores(tiles)
        marks = select_marks(scores, fps=args.fps, budget=len(gt), min_gap_s=0.0)
        if len(marks) < 1:
            continue
        mine = nearest(np.array([m.t for m in marks]), gt)

        # Same count of timestamps, drawn uniformly over the same span.
        draws = [
            nearest(np.sort(rng.uniform(0, duration, len(marks))), gt).mean()
            for _ in range(args.draws)
        ]
        ours.append(float(mine.mean()))
        rand.append(float(np.mean(draws)))
        per_video.append({
            "video": row["video_path"],
            "app": row.get("app", "?"),
            "duration_s": round(duration, 1),
            "gt": len(gt),
            "ours_s": round(float(mine.mean()), 2),
            "random_s": round(float(np.mean(draws)), 2),
        })

    o, r = np.array(ours), np.array(rand)
    if not len(o):
        print(f"{args.split}: no scorable videos on disk")
        return 0
    if not args.quiet:
        print(f"{'video':<18}{'app':<22}{'dur':>6}{'gt':>4}{'ours':>7}{'random':>8}  winner")
    for v in per_video if not args.quiet else []:
        w = "ours" if v["ours_s"] < v["random_s"] else "random"  # type: ignore[operator]
        print(f"{v['video']!s:<18}{str(v['app'])[:21]:<22}{v['duration_s']:>6}"
              f"{v['gt']:>4}{v['ours_s']:>7}{v['random_s']:>8}  {w}")

    wins = int((o < r).sum())
    if skipped:
        print(f"\nSKIPPED (unreadable file): {', '.join(skipped)}")
    print(f"\nsplit                    : {args.split}")
    print(f"videos scored            : {len(o)}")
    print(f"mean distance to a human keyframe -- ours {o.mean():.2f}s  random {r.mean():.2f}s")
    print(f"ours closer on           : {wins}/{len(o)} videos")
    # Sign test: how surprising is that win count if the two were equivalent?
    from math import comb
    n = len(o)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n
    print(f"one-sided sign test      : p = {p:.4f}")
    Path(f"scratch/eval/guiworld_{args.split}.json").write_text(
        json.dumps({"per_video": per_video, "wins": wins, "n": n,
                    "mean_ours": float(o.mean()), "mean_random": float(r.mean()),
                    "p": p, "fps": args.fps, "split": args.split}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
