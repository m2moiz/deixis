"""What a resume for the marks pass would save, and whether it could be exact.

Two questions, one script, because they are the two halves of the decision in
docs/marks-resume.md.

1. WHAT DOES THE SCAN COST. `extract_tile_grid` end to end on a real recording,
   wall clock and bytes. The answer sets the ceiling on what any resume can
   save.

2. COULD A RESUME BE EXACT, AND WOULD IT BE FAST. A resume restarts the decode
   part-way in, so it has to reproduce the tail of a full scan byte-for-byte --
   otherwise a resumed run yields different marks from an uninterrupted one,
   silently, since both look like plausible marks. And it only pays if the
   seeked decode is cheaper than the decode it replaces.

Three seek forms are compared because media.extract_frame already documents
that they are not interchangeable: `-ss` before `-i` seeks the container index
and can land on the wrong keyframe, `-ss` after `-i` is exact but decodes from
zero and so saves nothing, and the preroll pair is what extract_frame settled
on.

    uv run python scratch/marks_resume_probe.py <video> [resume_at_seconds]
    uv run python scratch/marks_resume_probe.py <video> [resume_at_seconds] --self-test

`--self-test` offsets the comparison by one frame, so every leg MUST report
DIFFERS. An all-IDENTICAL run is the answer that decided docs/marks-resume.md,
and a comparison that cannot say anything else would produce it whether or not
the seeks were exact -- practice #14 in docs/tooling-gaps.md.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dsj.dekho import DEFAULT_FPS, GRID_H, GRID_W
from dsj.media import PREROLL_S, extract_tile_grid

FRAME_BYTES = GRID_W * GRID_H


def _seeked_grid(
    media: Path, *, seek_before: float | None = None, seek_after: float | None = None
) -> tuple[NDArray[np.uint8], float]:
    """extract_tile_grid's ffmpeg call with seeks bolted on, plus its wall time.

    Spelled out rather than reusing extract_tile_grid because the seek is the
    whole subject: that function deliberately has no seek parameter, and adding
    one in order to ask whether it should exist would be assuming the answer.
    """
    pre = ["-ss", f"{seek_before:.6f}"] if seek_before is not None else []
    post = ["-ss", f"{seek_after:.6f}"] if seek_after is not None else []
    cmd = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-nostdin", "-hide_banner", "-loglevel", "error",
        *pre, "-i", str(media), *post,
        "-vf", f"fps={DEFAULT_FPS},scale={GRID_W}:{GRID_H},format=gray",
        "-an", "-f", "rawvideo", "-",
    ]
    started = time.monotonic()
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    elapsed = time.monotonic() - started
    n = len(out) // FRAME_BYTES
    return np.frombuffer(out[: n * FRAME_BYTES], dtype=np.uint8).reshape(n, GRID_H, GRID_W), elapsed


def _compare(
    name: str, full: NDArray[np.uint8], part: NDArray[np.uint8], k: int, secs: float
) -> bool:
    """Print the verdict for one seek form; return True when it read IDENTICAL."""
    tail = full[k:]
    overlap = min(len(tail), len(part))
    if overlap == 0:
        print(f"  {name:26s} no overlap: full tail {len(tail)}, seeked {len(part)}")
        return False
    # A count of differing frames, not a boolean. "Identical" and "one frame off
    # at the seam" are different findings, and the second is the interesting
    # failure -- a resume right everywhere except at the join is the shape that
    # survives a careless test.
    differing = [i for i in range(overlap) if not np.array_equal(tail[i], part[i])]
    verdict = "IDENTICAL" if not differing else f"DIFFERS at {len(differing)}/{overlap} frames"
    first = f", first at frame {differing[0]}" if differing else ""
    print(f"  {name:26s} {secs:6.1f}s  frames={len(part):5d}  {verdict}{first}")
    return not differing


def main(argv: list[str]) -> int:
    self_test = "--self-test" in argv
    argv = [a for a in argv if a != "--self-test"]
    media = Path(argv[0])
    started = time.monotonic()
    full = extract_tile_grid(media, fps=DEFAULT_FPS, width=GRID_W, height=GRID_H)
    scan_s = time.monotonic() - started
    duration_s = len(full) / DEFAULT_FPS
    print(f"{media.name}")
    print(
        f"  full scan                  {scan_s:6.1f}s  frames={len(full):5d}  "
        f"{full.nbytes / 1e6:.1f} MB  {duration_s / scan_s:.1f}x realtime"
    )

    # Half way in: the point a resume has the most to gain from, and the point
    # a kill is most likely to land if it lands uniformly.
    k = int(argv[1]) if len(argv) > 1 else len(full) // 2
    if self_test:
        print("  SELF TEST: comparing against a one-frame offset; all legs MUST differ")
    forms: list[tuple[str, float | None, float | None]] = [
        ("fast seek (-ss before -i)", float(k), None),
        ("exact seek (-ss after -i)", None, float(k)),
        ("preroll seek (both)", max(0.0, k - PREROLL_S), min(float(k), PREROLL_S)),
    ]
    # Offsetting the comparison, not the seek: the seeked decode stays exactly
    # what the real run does, so a leg that still reads IDENTICAL against a
    # deliberately misaligned tail proves the comparison is not looking.
    against = k + 1 if self_test else k
    identical: list[bool] = []
    for name, before, after in forms:
        part, secs = _seeked_grid(media, seek_before=before, seek_after=after)
        identical.append(_compare(name, full, part, against, secs))
    if self_test:
        if any(identical):
            print("SELF TEST FAILED: a misaligned tail still compared IDENTICAL")
            return 1
        print("SELF TEST PASSED: the comparison can report a difference")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
