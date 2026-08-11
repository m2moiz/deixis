#!/usr/bin/env bash
set -u
cd /Users/moiz/Documents/code/jaano
for split in software website multi; do
  echo "########## $split"
  uv run python scratch/guiworld_eval.py --split "$split" --limit 999 --fps 1 --quiet
done
echo "########## complete"
