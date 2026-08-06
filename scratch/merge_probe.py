"""Throwaway: measure the token-vote merge on the real 74-minute transcript."""
from __future__ import annotations

import bisect
import json
from collections import Counter

TR = json.load(open("meeting_deixis.json"))
DI = json.load(open("senko_meeting.json"))
turns = [(float(s["start"]), float(s["end"]), s["speaker"]) for s in DI["merged_segments"]]
turns.sort()
starts = [t[0] for t in turns]
labels = sorted({t[2] for t in turns})
idx = {name: i for i, name in enumerate(labels)}


def speaker_at(t: float) -> int | None:
    """Index of the turn covering t, or None if t falls in a gap."""
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return None
    s, e, spk = turns[i]
    return idx[spk] if s <= t <= e else None


def nearest(t: float) -> int:
    i = bisect.bisect_right(starts, t) - 1
    cands = [j for j in (i, i + 1) if 0 <= j < len(turns)]
    best = min(cands, key=lambda j: 0 if turns[j][0] <= t <= turns[j][1]
               else min(abs(t - turns[j][0]), abs(t - turns[j][1])))
    return idx[turns[best][2]]


sents = TR["sentences"]
straddle = 0
gap_only = 0
unanimous = 0
margins = []
assigned = []
for s in sents:
    votes = Counter()
    misses = 0
    for tok in s["tokens"]:
        v = speaker_at(tok["t"])
        if v is None:
            misses += 1
        else:
            votes[v] += 1
    if not votes:
        gap_only += 1
        assigned.append(nearest(s["start"]))
        continue
    if len(votes) > 1:
        straddle += 1
        top = votes.most_common()
        margins.append((top[0][1] - top[1][1]) / sum(votes.values()))
    else:
        unanimous += 1
    assigned.append(votes.most_common(1)[0][0])

print(f"sentences={len(sents)} unanimous={unanimous} straddling={straddle} "
      f"({straddle / len(sents):.1%}) all-tokens-in-gap={gap_only}")
if margins:
    margins.sort()
    print(f"straddle margin: min={margins[0]:.2f} median={margins[len(margins) // 2]:.2f}")

# How different would sentence-span max-overlap be?
def overlap_label(s):
    best, bestv = None, 0.0
    for a, b, spk in turns:
        ov = min(s["end"], b) - max(s["start"], a)
        if ov > bestv:
            best, bestv = idx[spk], ov
    return best if best is not None else nearest(s["start"])


diff = sum(1 for s, a in zip(sents, assigned) if overlap_label(s) != a)
print(f"token-vote vs sentence-span-overlap disagree on {diff} sentences ({diff / len(sents):.1%})")

# Schema cost
base = json.dumps(TR, separators=(",", ":"))
lab_int = json.dumps({**TR, "speakers": labels,
                      "sentences": [dict(s, speaker=a) for s, a in zip(sents, assigned)]},
                     separators=(",", ":"))
lab_str = json.dumps({**TR,
                      "sentences": [dict(s, speaker=labels[a]) for s, a in zip(sents, assigned)]},
                     separators=(",", ":"))
tok_int = json.dumps({**TR, "speakers": labels,
                      "sentences": [dict(s, speaker=a,
                                         tokens=[dict(t, s=speaker_at(t["t"])) for t in s["tokens"]])
                                    for s, a in zip(sents, assigned)]},
                     separators=(",", ":"))
print(f"compact base            {len(base):>9,} B")
print(f"+ sentence int speaker  {len(lab_int):>9,} B  (+{len(lab_int) - len(base):,}, "
      f"+{(len(lab_int) / len(base) - 1):.2%})")
print(f"+ sentence str speaker  {len(lab_str):>9,} B  (+{len(lab_str) - len(base):,}, "
      f"+{(len(lab_str) / len(base) - 1):.2%})")
print(f"+ per-token speaker     {len(tok_int):>9,} B  (+{len(tok_int) - len(base):,}, "
      f"+{(len(tok_int) / len(base) - 1):.2%})")
print(f"on-disk today (indent=none in repo): {len(json.dumps(TR)):,} B")

# Speaker share of the transcript by sentence and by word
by_spk = Counter()
for s, a in zip(sents, assigned):
    by_spk[labels[a]] += len(s["tokens"])
print("words per speaker:", dict(by_spk))
runs = 1 + sum(1 for i in range(1, len(assigned)) if assigned[i] != assigned[i - 1])
print(f"speaker runs across the transcript: {runs}")
