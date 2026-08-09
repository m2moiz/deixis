# Public datasets for evaluating screen-recording change-point detection

Investigated 2026-08-09. Every row below was checked by opening the dataset page,
README, or paper — not recalled. Where I could not confirm something, it says so.

Requirement recap: screen content (not natural film), ground-truth timestamps saying
"at time T something changed", downloadable today without an institutional agreement,
and a few-GB slice must be pullable.

## Ranked candidates

| # | Dataset | What the temporal annotations actually are | Distribution | Licence | Size | URL |
|---|---|---|---|---|---|---|
| 1 | **GUI-World** (ICLR 2025) | Per-video `keyframes` list: integer **frame index** + `sub_goal` + mouse/keyboard action. Human-annotated. Verified on `software.jsonl`: 496 benchmark videos, 2,401 keyframes, mean 4.8/video (min 2, max 17). Videos are 30 fps 1920×1080, so `frame/30` = seconds. | **Video files hosted directly on HF** as `.mp4`/`.mov`. Annotations as `.jsonl`. No YouTube step. | **CC BY 4.0** (stated in the HF README) | Whole repo ≈ 94 GB across 6 dirs (software 63.7 GB / 4,720 files, website 12.7 / 2,499, multi 8.6 / 475, IOS 7.3 / 492, android 0.9 / 3,800, XR 0.6 / 393). Individual videos 3–30 MB — a 50-video slice is ~1 GB. | https://huggingface.co/datasets/shuaishuaicdp/GUI-World · mirror `ONE-Lab/GUI-World` · code https://github.com/Dongping-Chen/GUI-World |
| 2 | **MaViLS** (Interspeech 2024) | Human raters map **each transcribed sentence (with a video timestamp) to a slide number**. Slide-change times are derivable wherever the slide index increments between consecutive sentences — so boundary precision is bounded by sentence granularity, not frame-exact. | Split: code + transcripts + ground-truth **Excel** files + slide PDFs on GitHub; **videos on Kaggle** (free account required). | Code Apache-2.0. **Dataset licence not stated** on the repo page — unresolved. | 20 lectures, >22 h total, so ~66 min average — this is the only candidate whose video length matches a 33-minute Teams recording. Byte size not stated on the Kaggle page I could load. | https://github.com/andererka/mavils · videos https://kaggle.com/datasets/e98bcdecedc67af45204338260556f932f8ec426b81caed0130d2cce80c4ea84 |
| 3 | **SeeAction** (ICSE 2025, distinguished paper) | 7,260 video–action pairs over 288 screencasts (12.8 h) of Word, Zoom, Firefox, Photoshop, Windows 10 Settings. Actions are `[command][widget][location]`. | Single **Google Drive zip**, linked from the README. | **Not stated** in the README. | 288 videos / 12.8 h; byte size not stated. | https://github.com/DehaiZhao/SeeAction · paper https://arxiv.org/abs/2503.12873 |
| 4 | **PsTuts** (CVPR 2020) | Adobe Photoshop screencast tutorials with low-level and high-level semantic labels and temporal segmentation into action clips. | **Google Drive** folder containing both the pre-segmented clips and the original full tutorial videos. | Code Apache-2.0; dataset licence not stated in README. | Not stated. | https://github.com/KunpengLi1994/PsTuts · https://sites.google.com/view/pstuts/ |
| 5 | **VidChapters-7M** (NeurIPS 2023 D&B) | User-written YouTube chapter titles + start timestamps. Real segment boundaries, but the corpus is general YouTube — screen content is a small unlabelled minority you would have to filter for yourself. | **Annotations only.** Video IDs; you download with yt-dlp. Link rot applies. | Code MIT; dataset licence stated as "its own licence" and I could not open a page giving its text. | 817K videos / 7M chapters. Any slice you want. | https://github.com/antoyang/VidChapters · https://antoyang.github.io/vidchapters.html |

## Checked and rejected

| Dataset | Why it does not work |
|---|---|
| **COIN** | 11,827 YouTube videos, 476 h, 46,354 step segments with time boundaries — the annotation shape is right, but the 180 tasks are nursing, vehicles, appliances, cooking, sports. **No software/computer tasks.** Also requires a **signed licence agreement emailed to the maintainer** — fails the "downloadable now" bar. https://coin-dataset.github.io/ |
| **Mind2Web** | Does contain "video recordings during annotation" in the raw dump, but the dump ships via **Globus at Ohio Supercomputer Center**, and the released annotations are per-step HTML/screenshot pairs, not timestamps into those videos. CC BY 4.0. Not usable as temporal ground truth without doing the alignment yourself. https://github.com/OSU-NLP-Group/Mind2Web |
| **VideoGUI** | Checked `VideoGUI/VideoGUI-Mid-Plan` on HF: 462 rows of **start/end screenshot pairs** plus text, parquet, no video and no timestamps. Full recordings live behind Google Drive links. Not a change-point benchmark. https://huggingface.co/datasets/VideoGUI/VideoGUI-Mid-Plan |
| **SliTraNet** | Method and pretrained weights are on GitHub; the FAU Erlangen lecture corpus (30 videos, frame-level hard/gradual transition GT) is **not linked anywhere in the repo** — the README only describes an expected folder layout. Would need to email the authors. https://github.com/asindel/SliTraNet |
| **"Slidin' Videos"** (ITU AI for Good) | ~3,000 slide transitions with start/end frames over 240 presentations — exactly the right annotation. The event page describes it but publishes **no download link, no licence, no repo**. Dead end without contacting ITU. https://aiforgood.itu.int/event/slidin-videos-slide-transition-detection-and-title-extraction-in-lecture-videos/ |
| **Sharingan** (arXiv 2411.08768, Microsoft) | Paper describes "a basic self-curated dataset and an advanced benchmark adapted from prior work". I could find **no release URL** on the arXiv page, the HF paper page, or the MSR publication page. Treat as unreleased. |
| **CodeSCAN** | 12,000 **static screenshots** of VS Code, not video. Annotations are IDE element boxes / OCR. Wrong modality. https://arxiv.org/abs/2409.18556 |
| **AITW / AndroidControl / GUI-Odyssey / ScreenSpot / Screen2Words / Rico** | All are screenshot-and-action-tuple corpora, not continuous video with time axes. Nothing to compute a change *time* against. (Not individually page-verified — rejected on modality, which is uncontested in their own titles.) |
| **WebArena / VisualWebArena** | Interactive environments, not recorded video corpora. |

## Recommendation

**Use GUI-World's `software` benchmark split.** It is the only thing I found that is
simultaneously (a) genuine desktop screen recordings, (b) hosted as real video files
needing no scraping, (c) carrying human-placed per-frame change timestamps, and (d) under
a clean CC BY 4.0 licence.

Mapping to our metric: for video `software/N.mp4`, ground-truth change times are
`[k["frame"]/30 for k in row["keyframes"]]` seconds. Score our top-N output against those
with a tolerance window (±1 s is the natural floor given we sample at 1 fps).

### The one real caveat

These clips are **short**. Estimating each video's length from its last keyframe index
across all 496 benchmark rows: median ≈ 15 s, p75 ≈ 21 s, max ≈ 110 s, ~2.3 h total.
A 15-second clip with 5 keyframes has a change roughly every 3 seconds — a much denser
change rate than a 33-minute meeting recording, and at 1 fps we only get ~15 samples per
clip. So GUI-World tests *precision of localisation on dense GUI activity*; it does not
test *sparse change detection over a long recording*. For that second axis, **MaViLS**
(20 lectures × ~66 min) is the complement, at the cost of a Kaggle account and
sentence-granular rather than frame-exact boundaries.

Recommended plan: GUI-World as the primary automated benchmark now; MaViLS as a
second, long-form check if the length mismatch turns out to matter.

### Exact commands for a small slice

Verified working on this machine (`hf` 1.26.0). Note the filenames must be **positional**
— passing them to `--include` alongside positionals makes `hf` warn and ignore the
`--include`.

```bash
# 1. Annotations only (2.7 MB) — inspect before pulling any video
hf download shuaishuaicdp/GUI-World Annotation/benchmark/software.jsonl \
  --repo-type dataset --local-dir ./data/gui-world

# 2. Pick the first 20 software videos referenced by the benchmark annotations
python3 - <<'PY'
import json, subprocess
rows = [json.loads(l) for l in open('./data/gui-world/Annotation/benchmark/software.jsonl')]
paths = [r['video_path'] for r in rows[:20]]
subprocess.run(['hf','download','shuaishuaicdp/GUI-World',*paths,
                '--repo-type','dataset','--local-dir','./data/gui-world'], check=True)
PY
# ~20 files x 3-30 MB = roughly 300-400 MB
```

Ground-truth extraction:

```python
import json
FPS = 30  # verified: software/0.mp4 is 1920x1080, avg_frame_rate=30/1, 21.37 s, 641 frames
for row in map(json.loads, open('Annotation/benchmark/software.jsonl')):
    gt_seconds = [k['frame'] / FPS for k in row['keyframes']]
    yield row['video_path'], gt_seconds
```

Confirm the fps per file rather than assuming 30 globally — I probed one video, not all of
them:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate,nb_frames -show_entries format=duration \
  -of default=noprint_wrappers=1 software/0.mp4
```

### What was actually observed vs. inferred

- Observed: `software/0.mp4` downloads (HTTP 200, 9,016,054 bytes), probes as
  1920×1080, `avg_frame_rate=30/1`, duration 21.367 s, 641 frames; its annotation's last
  keyframe is frame 621, consistent with 30 fps.
- Observed: `software.jsonl` parses to 496 rows, 2,401 keyframes, systems
  `{Windows, macOS}`.
- Observed: directory file counts and byte totals via the HF tree API.
- Observed: `hf download` with positional filenames fetches both an annotation file and a
  video into `--local-dir`.
- Inferred, not observed: the duration statistics (median 15 s etc.) are lower bounds
  computed from last-keyframe indices, not from probing all 496 files. The other five
  directories were not probed at all.
- Not confirmed: MaViLS dataset licence, MaViLS/SeeAction/PsTuts byte sizes, the
  VidChapters dataset licence text, and the SeeAction annotation file format (the README
  links a Drive zip but does not document its schema).
