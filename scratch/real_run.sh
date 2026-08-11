#!/usr/bin/env bash
# End-to-end on a real screen recording: transcript, then visual marks.
set -euo pipefail
V="/Users/moiz/Desktop/Screen Recording 2026-08-06 at 15.07.10.mov"
OUT=scratch/real_0806.json
cd /Users/moiz/Documents/code/jaano
echo "=== transcribe ==="
uv run python -m jaano.suno "$V" -o "$OUT" --no-diarize
echo "=== mark ==="
uv run python -m jaano.dekho "$V" -t "$OUT"
echo "=== done ==="
