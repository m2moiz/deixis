#!/usr/bin/env bash
# Sequenced, not concurrent: both jobs are decode-bound on the same few cores,
# and running them together would halve each and teach us nothing.
set -u
cd /Users/moiz/Documents/code/dsj
while pgrep -f guiworld_eval >/dev/null; do sleep 30; done
echo "########## mavils, like-for-like budget"
uv run python scratch/mavils_eval.py --budget-mode gt
echo "########## mavils, the shipped default budget"
uv run python scratch/mavils_eval.py --budget-mode default
echo "########## mavils complete"
