#!/usr/bin/env bash
# Run the mark evaluation over every recording available on this machine.
set -u
cd /Users/moiz/Documents/code/deixis
OUT=scratch/eval/mark_eval.jsonl
for v in "/Users/moiz/Movies/2024-11-14 17-21-27.mkv" \
         "/Users/moiz/Movies/2024-11-14 17-23-06.mkv" \
         "/Users/moiz/Movies/2024-11-14 17-23-57.mkv" \
         "/Users/moiz/Movies/2024-11-14 17-33-37.mkv" \
         "/Users/moiz/Movies/2024-11-14 17-26-56.mkv" \
         "/Users/moiz/Documents/code/hack-pearl/Screen Recording 2026-06-06 at 22.19.45.mov" \
         "/Users/moiz/Desktop/Screen Recording 2026-08-06 at 15.07.10.mov" \
         "/Users/moiz/Desktop/Screen Recording 2026-07-31 at 09.35.23.mov"; do
  echo "### $(basename "$v")"
  uv run python scratch/mark_eval.py "$v" --json "$OUT" || echo "FAILED: $v"
done
echo "### sweep complete"
