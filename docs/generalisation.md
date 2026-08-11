# Does any of this hold on a video that is not the reference recording?

Every number in [visual-marks.md](visual-marks.md) and
[do-marks-help.md](do-marks-help.md) came from one 33-minute Microsoft Teams
recording on one laptop. That is n=1, and n=1 is a demo. This is the check.

Four experiments: eight local recordings, two external benchmarks with human
ground truth (834 short GUI clips and five hour-long lectures), and a
sampling-rate sweep. Three of the four changed what the project believes.

## 1. Eight local recordings — does the `look` idea generalise?

`scratch/mark_eval.py` runs the shipped code on any video and scores two things
against a random baseline of the same size. No ground truth needed; both metrics
are computed from the video's own pixels.

- **Instability** — how settled is the frame you would send to a vision model?
- **Coverage** — pick any second, find its segment by the mark boundaries, take
  that segment's `look` frame: does it show the same screen?

```
video                        dur  mks | inst@t  @look   @rnd    x | cov@look  @rnd  sigma
2024-11-14 17-21-27.mkv       40    3 |  0.542  0.157  0.126  3.5 |    0.885 0.664    1.5
2024-11-14 17-23-06.mkv       50    4 |  0.646  0.174  0.139  3.7 |    0.874 0.613    2.2
2024-11-14 17-23-57.mkv       51    4 |  0.491  0.069  0.189  7.2 |    0.796 0.541    2.6
2024-11-14 17-33-37.mkv      107    8 |  0.340  0.180  0.139  1.9 |    0.862 0.678    2.2
2024-11-14 17-26-56.mkv      401   30 |  0.256  0.135  0.067  1.9 |    0.937 0.832    3.6
hack-pearl 2026-06-06         16    2 |  0.093  0.052  0.060  1.8 |    0.928 0.915    0.8
Teams 2026-08-06            1997  150 |  0.099  0.022  0.028  4.4 |    0.975 0.916    9.7
Teams 2026-07-31            4428  332 |  0.183  0.049  0.057  3.7 |    0.944 0.833   18.8
```

Two resolutions (1920x1080 OBS captures, 2940x1912 macOS), 16 seconds to 74
minutes, clearly different content.

- **`look` is more stable than `t` on 8/8**, by 1.8x to 7.2x. The midpoint idea
  is not an artifact of the reference recording.
- **`look` beats random coverage on 8/8**, by 0.8 to 18.8 sigma. The one weak
  case is the 16-second clip with two marks, which is too short to say anything.

But one column does **not** generalise, and it is worth saying plainly:

- **`look` beats random on *instability* only 4/8.** On the short OBS clips a
  random frame is often calmer than a `look` frame (0.157 vs 0.126, 0.174 vs
  0.139, 0.180 vs 0.139, 0.135 vs 0.067). Marks cluster where the action is, so
  their midpoints inherit some of that; random spreads into the dead air.

The two metrics are answering different questions and the split is the honest
result. Random gets you a calmer frame; `look` gets you a frame that *represents
the second you asked about*. For enumerating a recording the second is what
matters, and `look` wins that 8/8.

## 2. GUI-World — 834 recordings with human-annotated keyframes

[GUI-World](https://huggingface.co/datasets/shuaishuaicdp/GUI-World) (ICLR 2025,
CC BY 4.0) is 496 desktop, 246 browser and 146 multi-app screen recordings whose
**keyframes were placed by human annotators** at the moments a user action
changed the screen. Nobody here produced that ground truth, which is the point.

For each video the budget is set to **exactly the number of human keyframes**,
so the comparison is like-for-like rather than "who emits more guesses", and the
baseline is 200 draws of the same number of uniformly random timestamps.

```
split       videos    ours   random   better       won          p
software       490   1.18s    1.77s      33%   423/490    1.6e-64
website        246   0.98s    1.31s      25%   202/246    1.1e-25
multi           98   2.60s    2.84s       8%    76/98     1.9e-08
TOTAL          834                             701/834    3.5e-94
```

Four `software` videos are truncated on the hub and 48 `multi` videos went
unfetched at the download cap; both are named in the run output rather than
quietly dropped.

**The `multi` split is the honest weak spot** — 8% better than random, against
33% on `software`. Those clips are the longest (27s median) with the most
annotated changes, so a fixed budget of marks crowds together and the margin
compresses.

### Why this does not contradict the deictic result

[do-marks-help.md](do-marks-help.md) found the marks *losing* to random at
pointing to the moments that needed a frame (median 8.9s vs 4.7s). Both results
are correct and they measure different targets:

| Target | Nature | Marks |
|---|---|---|
| GUI-World keyframes | *user acted, the screen changed* — transitions | **win, p = 3.5e-94** |
| Deictic sentences | *speaker explains what is already on screen* — plateaus | **lose to random** |

A change detector finds changes. It does not find explanations, and explanation
happens after the change, while the screen sits still.

## 2b. MaViLS — the long-form case, 34 to 85 minutes

GUI-World clips are 15 seconds. jaano is for hour-long recordings, and that
shape had no ground truth until now. [MaViLS](https://github.com/andererka/mavils)
(Interspeech 2024) is 24 lectures whose transcribed sentences were mapped to
slide numbers by human raters; five of them are public YouTube videos.

**Ground truth here is an INTERVAL, not an instant.** A slide change is only
localised between the last sentence on the old slide and the first on the new
one. Scoring against either endpoint would invent precision the labels do not
have, so a mark inside the interval scores 0 and outside scores the distance to
the nearer end — and the random baseline is scored identically, so neither side
profits from the slack. The derivation was checked by eye before use: the frames
either side of one interval in `computer_vision_2_2` are a Camera Obscura
engraving and two photographs, a real content change correctly bracketed.

At a like-for-like budget (one mark per annotated change):

```
lecture                min    gt   marks    ours   random   hit%   rnd%
computer_vision_2_2   34.8    14      14   24.22    64.78    79%    21%
deeplearning          54.3    51      51    9.08    28.40    76%    17%
numerics              85.2    73      73    2.87    30.82    85%    16%
psychology            71.2    55      55   16.67    35.87    47%    13%
solar_resource        75.2    90      90   37.80    21.41    29%    20%

mean distance   ours 18.13s   random 36.26s
within 2s       ours 63%      random 17%      ours closer on 4/5
```

At the **shipped default** budget of 150 marks — what a user actually gets:

```
mean distance   ours 5.15s    random 9.26s
within 2s       ours 81%      random 42%      ours closer on 4/5
```

**Four in five slide changes are caught within two seconds on an hour-long
lecture.** That is the first evidence for the workload this project exists for.

### The one lecture it fails, and why

`solar_resource` loses to random in both modes. The cause is visible in a single
frame: it is **a camera pointed at a lecture hall**, not a screen recording. A
presenter walks about in front of a projected slide, and his motion dominates
every frame. Measured against a true screen capture:

```
median per-second change score
  solar_resource   2187      <- camera in the room, presenter moving
  numerics          347      <- screen capture
```

Six times the noise floor, and the slide changes are buried under it. This is a
real limit worth stating plainly: **the detector assumes the frame IS the
screen.** Neither the budget nor any threshold rescues a recording where most of
the pixels are a person. Detecting that case — the score's own noise floor is
the obvious signal — is not implemented.

## 3. The sampling rate — 1 fps is the best setting, not a compromise

The default is 1 fps and it looked like a decode-cost concession. Measured on
GUI-World across four rates:

```
fps=1  ours 0.79s vs random 1.36s   37/38 videos   p < 0.0001
fps=2  ours 0.84s vs random 1.31s   34/39          p < 0.0001
fps=4  ours 1.00s vs random 1.31s   30/39          p = 0.0005
fps=8  ours 1.11s vs random 1.31s   28/39          p = 0.0047
```

Accuracy degrades **monotonically as sampling gets finer.** The first version of
the benchmark harness defaulted to 4 fps and its docstring argued that 1 fps
would be too coarse for 15-second clips — a confident claim, written before
measuring, and wrong. It has been corrected in place rather than quietly fixed.

Likely mechanism, **untested**: with more samples per transition, several marks
land inside one change event and a fixed budget gets spent twice on the same
moment; coarse sampling forces them apart.

## What this run establishes, and what it does not

**Established:**

- The `look` midpoint beats the mark itself on every video tried (8/8).
- Marks match human change annotations far better than chance on **834
  third-party GUI recordings**, p = 3.5e-94 pooled.
- On **hour-long lectures with slide-change ground truth**, 81% of changes are
  caught within two seconds at the shipped default budget, against 42% for
  random. This is the workload the project targets and it now has evidence.
- 1 fps is the right default, on evidence rather than convenience.
- The pipeline runs unmodified from 16-second clips to 85-minute lectures, on
  1920x1080 OBS captures, 2940x1912 macOS captures and YouTube lecture video.

**Not established, and one known failure:**

- **A camera pointed at a screen defeats it.** `solar_resource` is a lecture
  hall filmed from the back; the presenter's motion is 6x the change floor of a
  real screen capture and the slide changes vanish under it. Detecting this case
  automatically is not implemented.
- The `multi` GUI-World split wins by only 8%. Denser change compresses the
  margin.
- Five lectures is a small long-form sample, and all five are slide decks. No
  coding session, no browser-only long-form session.
- The eight local videos have no ground truth; their metrics can compare schemes
  but never confirm a mark is *interesting*.
- Nothing here re-tests whether marks help an *agent*. That remains
  [do-marks-help.md](do-marks-help.md), n=8 questions, one recording.

## Datasets that turned out unusable

Checked by inspecting what actually landed on disk, not by reading a page:

| Dataset | Why not |
|---|---|
| **SeeAction** | 2 GB fetched, and it is `dataset/Images/photoshop_001/03538.png` — extracted **frames**, not video. The archive is also truncated at exactly 2 GiB by the download cap. |
| **VidChapters-7M** | 1.9 GB is `chapters.pkl` — annotations only. Every video needs a separate yt-dlp fetch, and screen content is an unlabelled minority of general YouTube. |
| **PsTuts** | Google Drive download produced 0 bytes. |

[datasets-surveyed.md](datasets-surveyed.md) records the wider survey, including
eight further candidates rejected before download.
