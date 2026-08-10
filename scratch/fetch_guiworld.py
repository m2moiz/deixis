"""Fetch the GUI-World benchmark videos for the three desktop/browser splits.

Mobile (android/IOS) and XR are skipped: different modality, and their
annotations do not even use the same `keyframes` key.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path("scratch/guiworld")
SPLITS = ("software", "website", "multi")
CAP_GB = 6.0

def size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*.mp4")) / 1e9

want: list[str] = []
for split in SPLITS:
    ann = ROOT / "Annotation/benchmark" / f"{split}.jsonl"
    rows = [json.loads(line) for line in ann.open()]
    for r in rows:
        p = r["video_path"]
        if not (ROOT / p).exists():
            want.append(p)
print(f"{len(want)} videos to fetch across {SPLITS}", flush=True)
failed: list[str] = []

BATCH = 40
for i in range(0, len(want), BATCH):
    if size_gb(ROOT) > CAP_GB:
        print(f"CAP HIT at {size_gb(ROOT):.1f} GB -- stopping, {len(want)-i} videos left unfetched", flush=True)
        break
    chunk = want[i:i+BATCH]
    # Retry, and COUNT what still failed. The first version passed check=False
    # and printed only progress, so a transient hub error silently left 182 of
    # 246 website videos unfetched and the run still said "done".
    for _attempt in range(3):
        subprocess.run(["hf","download","shuaishuaicdp/GUI-World",*chunk,
                        "--repo-type","dataset","--local-dir",str(ROOT),"--quiet"], check=False)
        if all((ROOT / c).exists() for c in chunk):
            break
    missing = [c for c in chunk if not (ROOT / c).exists()]
    if missing:
        failed.extend(missing)
        print(f"  FAILED after 3 attempts: {len(missing)} of this batch", flush=True)
    print(f"  {i+len(chunk)}/{len(want)}  {size_gb(ROOT):.2f} GB", flush=True)

print(f"done {size_gb(ROOT):.2f} GB", flush=True)
if failed:
    print(f"UNFETCHED: {len(failed)} videos -- {failed[:10]}", flush=True)
