"""Measure change-point rates on a real video with a dHash over ffmpeg output.

Decodes at a fixed fps, downscales to 9x8 greyscale in ffmpeg, computes a 64-bit
difference hash per frame in numpy, and reports how many frames survive a
Hamming-distance threshold against the last *kept* frame.

Usage: python scratch/hash_probe.py <video> [--fps 1] [--thresholds 3,5,8,12]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

W, H = 9, 8
FRAME_BYTES = W * H


def decode_hashes(path: str, fps: float) -> tuple[np.ndarray, float]:
    """Return (uint64 hashes, wall seconds) for every sampled frame."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"fps={fps},scale={W}:{H},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    hashes: list[int] = []
    try:
        while True:
            buf = proc.stdout.read(FRAME_BYTES)
            if len(buf) < FRAME_BYTES:
                break
            f = np.frombuffer(buf, dtype=np.uint8).reshape(H, W)
            bits = f[:, 1:] > f[:, :-1]
            hashes.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    finally:
        proc.stdout.close()
        proc.wait()
    return np.array(hashes, dtype=np.uint64), time.monotonic() - start


def popcount(x: np.uint64) -> int:
    return int(x).bit_count()


def keep_count(hashes: np.ndarray, threshold: int) -> list[int]:
    """Indices kept: first frame, then any frame >threshold from the last kept."""
    if len(hashes) == 0:
        return []
    kept = [0]
    last = hashes[0]
    for i in range(1, len(hashes)):
        if popcount(last ^ hashes[i]) > threshold:
            kept.append(i)
            last = hashes[i]
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--thresholds", default="0,3,5,8,12,16")
    ap.add_argument("--cache", help="npy file to read hashes from / write them to")
    ap.add_argument("--dump", help="write kept indices per threshold as JSON here")
    args = ap.parse_args()

    if args.cache and Path(args.cache).exists():
        hashes = np.load(args.cache)
        secs = float("nan")
    else:
        hashes, secs = decode_hashes(args.video, args.fps)
        if args.cache:
            np.save(args.cache, hashes)
    n = len(hashes)
    print(f"frames sampled : {n}  (fps={args.fps})")
    print(f"decode+hash    : {secs:.1f}s  ({n / secs:.0f} frames/s)")

    if n > 1:
        consec = np.array([popcount(hashes[i - 1] ^ hashes[i]) for i in range(1, n)])
        pct = np.percentile(consec, [50, 75, 90, 99])
        print(
            "consecutive Hamming: "
            f"mean={consec.mean():.1f} p50={pct[0]:.0f} p75={pct[1]:.0f} "
            f"p90={pct[2]:.0f} p99={pct[3]:.0f} max={consec.max()}"
        )
        print(f"identical pairs    : {(consec == 0).sum()} / {n - 1}")

    print()
    print(f"{'thresh':>6} {'kept':>7} {'%kept':>7} {'per-min':>8}")
    minutes = n / (args.fps * 60.0)
    dumped: dict[str, list[int]] = {}
    for t in (int(x) for x in args.thresholds.split(",")):
        kept = keep_count(hashes, t)
        dumped[str(t)] = kept
        print(f"{t:>6} {len(kept):>7} {100 * len(kept) / n:>6.1f}% {len(kept) / minutes:>8.1f}")

    if args.dump:
        Path(args.dump).write_text(json.dumps({"fps": args.fps, "n": n, "kept": dumped}, indent=1))
        print(f"\nwrote {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
