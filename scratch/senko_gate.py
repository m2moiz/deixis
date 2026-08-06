"""Throwaway gate: does senko build, load and produce sane turns on this machine?"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from senko import Diarizer

wav = Path(sys.argv[1])
out = Path(sys.argv[2])

t0 = time.perf_counter()
d = Diarizer(quiet=False)
t_load = time.perf_counter() - t0
print(f"device={d.device!r} vad={getattr(d, 'vad', None)!r} clustering={getattr(d, 'clustering', None)!r}")
print(f"LOAD {t_load:.2f}s")

t1 = time.perf_counter()
res = d.diarize(str(wav))
t_diar = time.perf_counter() - t1
print(f"DIARIZE {t_diar:.2f}s")
print("top-level keys:", list(res.keys()))

out.write_text(json.dumps(res, indent=2, default=str))
segs = res.get("segments") or res.get("merged_segments") or []
print(f"segments={len(segs)}")
if segs:
    print("first segment repr:", json.dumps(segs[0], default=str))
    spk = {}
    for s in segs:
        k = s.get("speaker")
        spk[k] = spk.get(k, 0.0) + (float(s["end"]) - float(s["start"]))
    print("speaker -> total seconds:", {k: round(v, 1) for k, v in sorted(spk.items(), key=lambda kv: -kv[1])})
    print("first 12 turns:")
    for s in segs[:12]:
        print(f"  {float(s['start']):8.2f} -> {float(s['end']):8.2f}  {s.get('speaker')}")
