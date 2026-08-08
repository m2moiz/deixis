"""Tile-grid change detection with a learned activity mask.

Reimplements SeeAction's change-region idea (SSIM map -> connected regions) on an
ffmpeg + numpy pipe: decode to a coarse greyscale grid, diff tiles, suppress tiles
that change chronically (webcam tiles, clocks, spinners), and mark a frame when a
large enough connected region of the remaining tiles changed.

Decode once at a fine grid; coarser grids are derived by block-averaging, so a
parameter sweep costs one decode.

Usage:
  python scratch/tile_probe.py <video> --cache scratch/tiles.npy          # decode
  python scratch/tile_probe.py <video> --cache scratch/tiles.npy --sweep  # analyse
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

FINE_W, FINE_H = 128, 84  # ~23x23 px tiles on a 2940x1912 source


def decode_tiles(path: str, fps: float, w: int, h: int) -> tuple[np.ndarray, float]:
    """Decode the video to an (N, h, w) uint8 array of greyscale tile means."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"fps={fps},scale={w}:{h},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    n_bytes = w * h
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frames: list[np.ndarray] = []
    try:
        while True:
            buf = proc.stdout.read(n_bytes)
            if len(buf) < n_bytes:
                break
            frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w))
    finally:
        proc.stdout.close()
        proc.wait()
    return np.stack(frames), time.monotonic() - start


def coarsen(fine: np.ndarray, factor: int) -> np.ndarray:
    """Block-average an (N, h, w) array down by an integer factor."""
    if factor == 1:
        return fine
    n, h, w = fine.shape
    h2, w2 = h // factor, w // factor
    trimmed = fine[:, : h2 * factor, : w2 * factor].astype(np.uint16)
    return trimmed.reshape(n, h2, factor, w2, factor).mean(axis=(2, 4)).astype(np.uint8)


def largest_component(mask: np.ndarray) -> int:
    """Size of the largest 4-connected True region in a 2-D boolean array."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best = 0
    for sy, sx in np.argwhere(mask):
        if seen[sy, sx]:
            continue
        size = 0
        q = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        while q:
            y, x = q.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        best = max(best, size)
    return best


Result = dict[str, Any]


def analyse(
    tiles: np.ndarray,
    delta: int,
    activity_max: float,
    min_cluster: int,
    mask_cap: float = 0.25,
) -> Result:
    """Mark frames whose unmasked changed tiles form a large enough region."""
    d = np.abs(tiles[1:].astype(np.int16) - tiles[:-1].astype(np.int16))
    changed = d > delta

    activity = changed.mean(axis=0)
    noisy = activity >= activity_max
    noisy_frac = float(noisy.mean())
    masked_off = noisy_frac <= mask_cap
    usable = ~noisy if masked_off else np.ones_like(noisy)

    marks: list[int] = []
    sizes: list[int] = []
    for i in range(changed.shape[0]):
        region = changed[i] & usable
        size = largest_component(region) if region.any() else 0
        sizes.append(size)
        if size >= min_cluster:
            marks.append(i + 1)  # index into the frame array

    return Result(
        marks=marks,
        n_frames=int(tiles.shape[0]),
        cluster_sizes=sizes,
        mask_applied=masked_off,
        masked_tile_fraction=noisy_frac,
        params={
            "grid": [int(tiles.shape[2]), int(tiles.shape[1])],
            "delta": delta,
            "activity_max": activity_max,
            "min_cluster": min_cluster,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dump", help="write marks for the best config as JSON")
    ap.add_argument("--probe", default="379,1195", help="frame indices to report on")
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.exists():
        fine = np.load(cache)
        print(f"loaded {cache} {fine.shape}")
    else:
        fine, secs = decode_tiles(args.video, args.fps, FINE_W, FINE_H)
        np.save(cache, fine)
        print(f"decoded {fine.shape} in {secs:.1f}s -> {cache}")

    if not args.sweep:
        return 0

    probes = [int(x) for x in args.probe.split(",")]
    minutes = fine.shape[0] / (args.fps * 60.0)
    print(
        f"\n{'grid':>8} {'delta':>6} {'act':>5} {'clust':>6} "
        f"{'marks':>6} {'/min':>6} {'masked%':>8} " + " ".join(f"{p:>7}" for p in probes)
    )

    best: Result | None = None
    t0 = time.perf_counter()
    for factor, label in ((1, "128x84"), (2, "64x42"), (4, "32x21")):
        tiles = coarsen(fine, factor)
        for delta in (8, 12, 20):
            for activity_max in (0.20, 0.35):
                for min_cluster in (2, 3, 5):
                    r = analyse(tiles, delta, activity_max, min_cluster)
                    marks = set(r["marks"])
                    hits = " ".join(f"{'MARK' if p in marks else '-':>7}" for p in probes)
                    print(
                        f"{label:>8} {delta:>6} {activity_max:>5.2f} {min_cluster:>6} "
                        f"{len(r['marks']):>6} {len(r['marks']) / minutes:>6.1f} "
                        f"{100 * r['masked_tile_fraction']:>7.1f}% {hits}"
                    )
                    # target: no mark at the cursor-only frame, a mark at the checkbox
                    hit = probes[0] not in marks and probes[1] in marks
                    if hit and (best is None or len(r["marks"]) < len(best["marks"])):
                        best = r
    print(f"\nsweep took {time.perf_counter() - t0:.1f}s")

    if best is None:
        print(f"NO CONFIG satisfies: no mark at {probes[0]} and a mark at {probes[1]}")
        return 1
    print(f"best: {best['params']}  marks={len(best['marks'])}")
    if args.dump:
        Path(args.dump).write_text(
            json.dumps({k: v for k, v in best.items() if k != "cluster_sizes"}, indent=1)
        )
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
