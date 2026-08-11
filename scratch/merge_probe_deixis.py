"""Throwaway: re-run the §2.4 measurement through jaano.merge itself.

The point is to prove the shipped implementation is the algorithm the plan
specified, not merely something that passes the unit tests. Run from the repo
root: uv run python scratch/merge_probe_deixis.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jaano.merge import Turn, TurnIndex, label_sentence

HERE = Path(__file__).parent
TR = json.loads((HERE / "meeting_deixis.json").read_text())
DI = json.loads((HERE / "senko_meeting.json").read_text())

# Exactly what jaano/diarize.py's _to_turns does, inlined so this probe does
# not need senko installed to run.
labels = sorted({s["speaker"] for s in DI["merged_segments"]})
idx = {name: i for i, name in enumerate(labels)}
turns = sorted(
    Turn(float(s["start"]), float(s["end"]), idx[s["speaker"]])
    for s in DI["merged_segments"]
)
index = TurnIndex(turns)

sents = TR["sentences"]
straddle = gap_only = unanimous = ties = 0
gap_tokens = 0
margins = []
assigned = []
for s in sents:
    votes = Counter()
    for tok in s["tokens"]:
        v = index.speaker_at(tok["t"])
        if v is None:
            gap_tokens += 1
        else:
            votes[v] += 1
    if not votes:
        gap_only += 1
    elif len(votes) > 1:
        straddle += 1
        top = votes.most_common()
        margins.append((top[0][1] - top[1][1]) / sum(votes.values()))
        if top[0][1] == top[1][1]:
            ties += 1
    else:
        unanimous += 1
    assigned.append(label_sentence(s, index))

print(
    f"sentences={len(sents)} unanimous={unanimous} straddling={straddle} "
    f"({straddle / len(sents):.1%}) all-tokens-in-gap={gap_only}"
)
margins.sort()
print(f"straddle margin: min={margins[0]:.2f}  median={margins[len(margins) // 2]:.2f}")
print(f"tie sentences: {ties}")
print(f"tokens falling in a VAD gap: {gap_tokens} of {sum(len(s['tokens']) for s in sents)}")

by_spk = Counter()
for s, a in zip(sents, assigned):
    by_spk[labels[a]] += len(s["tokens"])
print("words per speaker:", dict(by_spk))
won = Counter(labels[a] for a in assigned)
print("sentences per speaker:", dict(won))
runs = 1 + sum(1 for i in range(1, len(assigned)) if assigned[i] != assigned[i - 1])
print(f"speaker runs across the transcript: {runs}")
