"""Score a raw_output.json against the hand-written ground truth.

The 2026-08-08 gemma run scored recall by hand-applying the rule written down in
scratch/vlm/RESULTS.md: case-insensitive substring match, whitespace collapsed,
markdown list markers stripped. Re-running that rule as code is the only way a
later run is comparable to it, so this script implements exactly that rule and
nothing more — it deliberately does no fuzzy matching, because a fuzzy match
would score `Your GPS aren't the problem` as a hit for `Your GPUs ...`, and that
substitution is the failure the measurement exists to catch.

Hallucinations are not scored here. They cannot be: counting them requires
comparing the output against the image, not against a string list.

Usage:
    python score.py <ground_truth.md> <raw_output.json>
"""

import json
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^## (\S+\.jpg)")
# Some ground-truth items carry a trailing parenthetical note outside the
# backticks (`... ` (browser tab title)), so match the first backticked span
# rather than requiring it to run to end of line.
ITEM = re.compile(r"^\d+\.\s+`([^`]+)`")


def normalize(text: str) -> str:
    """Lowercase, drop markdown list/table markers, collapse whitespace."""
    text = re.sub(r"[*|#`]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_ground_truth(path: Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        heading = HEADING.match(line)
        if heading:
            current = heading.group(1)
            truth[current] = []
            continue
        item = ITEM.match(line)
        if item and current is not None:
            truth[current].append(item.group(1))
    return truth


def main() -> None:
    truth = load_ground_truth(Path(sys.argv[1]))
    raw = json.loads(Path(sys.argv[2]).read_text())

    total_gt = 0
    total_hit = 0
    rows: list[tuple[str, int, int, float]] = []
    for entry in raw["results"]:
        name = Path(entry["image"]).name
        if name not in truth:
            continue
        output = normalize(entry["output"])
        strings = truth[name]
        hits = [s for s in strings if normalize(s) in output]
        misses = [s for s in strings if normalize(s) not in output]
        rows.append((name, len(hits), len(strings), float(entry["seconds"])))
        total_gt += len(strings)
        total_hit += len(hits)
        print(f"\n{name}: {len(hits)}/{len(strings)}  ({entry['seconds']}s)")
        for s in misses:
            print(f"    MISS  {s}")

    print(f"\nmodel: {raw.get('model')}  max_tokens: {raw.get('max_tokens')}")
    print(f"TOTAL recall: {total_hit}/{total_gt} " f"({100 * total_hit / total_gt:.0f}%)")
    for name, hit, n, secs in rows:
        print(f"  {name}: {hit}/{n} {secs}s")


if __name__ == "__main__":
    main()
