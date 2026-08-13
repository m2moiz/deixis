"""Score dsj' marks against MaViLS slide changes, on 22-102 minute lectures.

The long-form benchmark. GUI-World validated the detector on 15-second GUI clips
with dense action; this is the other end -- hour-long recordings where the
screen is a slide deck and a change is rare. That shape is what dsj is
actually for, and until now nothing with ground truth covered it.

GROUND TRUTH IS AN INTERVAL, NOT AN INSTANT. MaViLS raters mapped each
transcribed sentence to a slide number, so a slide change is only localised
between the last sentence on the old slide (`lower_bound`) and the first
sentence on the new one (`time`). Scoring against either endpoint alone would
invent precision the labels do not have, so a mark landing anywhere inside the
interval scores 0 and outside it scores the distance to the nearer end. The
random baseline is scored identically, so neither side profits from the slack.

Verified before trusting the derivation: the frames either side of one interval
in computer_vision_2_2 are a Camera Obscura engraving and two photographs --
a real content change, correctly bracketed.

Usage: python scratch/mavils_eval.py [--budget-mode gt|default]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsj.dekho import GRID_H, GRID_W, change_scores, select_marks
from dsj.media import extract_tile_grid

DATA = Path(__file__).resolve().parent / "datasets"
GT = DATA / "mavils_changepoints.json"
VIDEOS = DATA / "mavils_youtube"

# The five lectures whose video was fetched, keyed to their ground-truth file.
PAIRS = {
    "computer_vision_2_2_high_res_NIaICLR7D0Q.mp4": "ground_truth_computer_vision_2_2.xlsx",
    "deeplearning_goodfellow_XlYD8jn1ayE.mp4": "ground_truth_deeplearning.xlsx",
    "numerics_hennig_2ETIOk_Sbhk.mp4": "ground_truth_numerics.xlsx",
    "psychology_MIT_syXplPKQb_o.mp4": "ground_truth_psychology.xlsx",
    "solar_resource_BcVzc6IGwS0.mp4": "ground_truth_solar_resource.xlsx",
}

TOLERANCE_S = 2.0  # a hit, at 1 fps sampling
RANDOM_DRAWS = 100


def interval_distance(marks: np.ndarray, lo: float, hi: float) -> float:
    """Distance from the nearest mark to the interval [lo, hi]; 0 if inside."""
    inside = (marks >= lo) & (marks <= hi)
    if inside.any():
        return 0.0
    return float(np.minimum(np.abs(marks - lo), np.abs(marks - hi)).min())


def score(marks: np.ndarray, intervals: list[tuple[float, float]]) -> tuple[float, float]:
    """Mean distance to ground truth, and the fraction hit within TOLERANCE_S."""
    d = np.array([interval_distance(marks, lo, hi) for lo, hi in intervals])
    return float(d.mean()), float((d <= TOLERANCE_S).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--min-gap", type=float, default=5.0)
    ap.add_argument(
        "--budget-mode",
        default="gt",
        choices=("gt", "default"),
        help="gt: one mark per ground-truth change (like-for-like). "
        "default: the shipped budget of 150, i.e. what a user actually gets.",
    )
    args = ap.parse_args()

    truth = json.loads(GT.read_text())
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | int | str]] = []

    for video_name, gt_key in PAIRS.items():
        path = VIDEOS / video_name
        if not path.exists() or gt_key not in truth:
            print(f"SKIP {video_name}: video or ground truth missing")
            continue

        t0 = time.monotonic()
        tiles = extract_tile_grid(path, fps=args.fps, width=GRID_W, height=GRID_H)
        decode_s = time.monotonic() - t0
        duration = len(tiles) / args.fps

        cps = truth[gt_key]["changepoints"]
        intervals = [
            (float(c["lower_bound"]), float(c["time"]))
            for c in cps
            if float(c["time"]) <= duration
        ]
        if len(intervals) < 5:
            print(f"SKIP {video_name}: only {len(intervals)} usable changepoints")
            continue

        budget = len(intervals) if args.budget_mode == "gt" else 150
        marks = select_marks(
            change_scores(tiles), fps=args.fps, budget=budget, min_gap_s=args.min_gap
        )
        mine = np.array([m.t for m in marks])
        ours_d, ours_hit = score(mine, intervals)

        draws = [
            score(np.sort(rng.uniform(0, duration, len(marks))), intervals)
            for _ in range(RANDOM_DRAWS)
        ]
        rand_d = float(np.mean([d for d, _ in draws]))
        rand_hit = float(np.mean([h for _, h in draws]))

        rows.append({
            "lecture": gt_key.replace("ground_truth_", "").replace(".xlsx", ""),
            "minutes": round(duration / 60, 1),
            "gt": len(intervals),
            "marks": len(marks),
            "decode_s": round(decode_s),
            "ours_s": round(ours_d, 2),
            "random_s": round(rand_d, 2),
            "ours_hit": round(ours_hit, 3),
            "random_hit": round(rand_hit, 3),
        })
        print(f"  scored {rows[-1]['lecture']}", flush=True)

    if not rows:
        print("nothing scored")
        return 1

    print(f"\nbudget mode: {args.budget_mode}   tolerance: {TOLERANCE_S}s   fps: {args.fps}\n")
    print(f"{'lecture':<24}{'min':>6}{'gt':>5}{'marks':>7}{'ours':>8}{'random':>8}"
          f"{'hit%':>7}{'rnd%':>7}")
    for r in rows:
        print(f"{str(r['lecture'])[:23]:<24}{r['minutes']:>6}{r['gt']:>5}{r['marks']:>7}"
              f"{r['ours_s']:>8}{r['random_s']:>8}"
              f"{100 * float(r['ours_hit']):>6.0f}%{100 * float(r['random_hit']):>6.0f}%")

    o = np.array([float(r["ours_s"]) for r in rows])
    rd = np.array([float(r["random_s"]) for r in rows])
    oh = np.array([float(r["ours_hit"]) for r in rows])
    rh = np.array([float(r["random_hit"]) for r in rows])
    print(f"\nmean distance to a slide change: ours {o.mean():.2f}s   random {rd.mean():.2f}s")
    print(f"within {TOLERANCE_S:.0f}s of a change:        ours {100 * oh.mean():.0f}%"
          f"      random {100 * rh.mean():.0f}%")
    print(f"ours closer on {int((o < rd).sum())}/{len(o)} lectures")

    out = Path("scratch/eval") / f"mavils_{args.budget_mode}.json"
    out.write_text(json.dumps({"rows": rows, "fps": args.fps,
                               "budget_mode": args.budget_mode}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
