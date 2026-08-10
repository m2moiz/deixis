#!/usr/bin/env python3
"""Turn MaViLS ground-truth spreadsheets into change-point timestamps in seconds.

Each MaViLS ground-truth .xlsx has one row per transcribed sentence:

    Time (or Key) | Slidenumber | Value
    127.70        | 3           | "OK, so now that we've situated us, ..."

`Time` is the sentence start in seconds from the beginning of the recording.
`Slidenumber` is the slide the human rater says is on screen at that moment;
-1 means "no slide visible" (camera on the speaker, whiteboard, demo).

Needs pandas and openpyxl, which are NOT project dependencies -- nothing
shipped reads spreadsheets. Run it without touching the lockfile:

    uv run --with pandas --with openpyxl python \
      scratch/datasets/mavils_gt_to_changepoints.py \
      scratch/datasets/mavils/data/ground_truth_files -o mavils_changepoints.json

Verified 2026-08-10: regenerates scratch/datasets/mavils_changepoints.json
byte-identically -- 24 lectures, 25.2 h, 1344 change points.

A change point is any row whose Slidenumber differs from the previous row's.
The emitted timestamp is that row's Time -- i.e. the first sentence spoken
under the new slide. The true pixel change happened somewhere in the silent
gap before it, so each mark is a late-biased estimate; see `lower_bound`.

Usage:
    uv run --with pandas --with openpyxl python mavils_gt_to_changepoints.py \
        mavils/data/ground_truth_files -o mavils_changepoints.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def changepoints(path):
    """Return (metadata, list-of-changepoint-dicts) for one ground-truth file."""
    df = pd.read_excel(path)

    # Some files name the time column "Time", others "Key". Same meaning.
    time_col = "Time" if "Time" in df.columns else "Key"
    if time_col not in df.columns or "Slidenumber" not in df.columns:
        raise ValueError(f"{path}: unexpected columns {list(df.columns)}")

    df = df.dropna(subset=[time_col, "Slidenumber"]).sort_values(time_col)
    times = df[time_col].astype(float).tolist()
    slides = df["Slidenumber"].astype(int).tolist()

    points = []
    for i in range(1, len(times)):
        if slides[i] == slides[i - 1]:
            continue
        points.append(
            {
                "time": times[i],
                # The change cannot have happened before the previous sentence
                # started, so [lower_bound, time] brackets the true instant.
                "lower_bound": times[i - 1],
                "from_slide": slides[i - 1],
                "to_slide": slides[i],
                # A -1 endpoint means the view left or entered "no slide" --
                # still a real visual change, but a cut rather than a slide flip.
                "kind": "slide_flip" if -1 not in (slides[i - 1], slides[i]) else "view_cut",
            }
        )

    meta = {
        "file": Path(path).name,
        "sentences": len(times),
        "duration_s": times[-1] if times else 0.0,
        "distinct_slides": len(set(slides)),
        "n_changes": len(points),
    }
    return meta, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_dir", help="directory of MaViLS ground_truth_*.xlsx files")
    ap.add_argument("-o", "--out", default="mavils_changepoints.json")
    args = ap.parse_args()

    result = {}
    for path in sorted(Path(args.gt_dir).glob("*.xlsx")):
        meta, points = changepoints(path)
        result[meta["file"]] = {"meta": meta, "changepoints": points}
        flips = sum(p["kind"] == "slide_flip" for p in points)
        print(
            f"{meta['file'][:50]:50s} {meta['duration_s']/60:6.1f} min "
            f"{meta['n_changes']:4d} changes ({flips} flips, "
            f"{meta['n_changes']-flips} cuts)"
        )

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    total = sum(r["meta"]["n_changes"] for r in result.values())
    hours = sum(r["meta"]["duration_s"] for r in result.values()) / 3600
    print(f"\n{len(result)} lectures, {hours:.1f} h, {total} change points -> {args.out}")


if __name__ == "__main__":
    main()
