# Does any of this hold on a video that is not the reference recording?

Every number in [visual-marks.md](visual-marks.md) and
[do-marks-help.md](do-marks-help.md) came from one 33-minute Microsoft Teams
recording on one laptop. That is n=1, and n=1 is a demo. This is the check.

Three experiments: eight local recordings, an external benchmark with human
ground truth, and a sampling-rate sweep. Two of the three changed what the
project believes.

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

## 2. GUI-World — an external benchmark with human ground truth

[GUI-World](https://huggingface.co/datasets/shuaishuaicdp/GUI-World) (ICLR 2025,
CC BY 4.0) is 496 desktop screen recordings whose **keyframes were placed by
human annotators** at the moments a user action changed the screen. Nobody here
produced that ground truth, which is the entire point of using it.

Method: 40 videos from the `software` benchmark split (one, `software/165.mp4`,
is truncated on the hub and is skipped and named rather than silently dropped).
For each video the budget is set to **exactly the number of human keyframes**,
so the comparison is like-for-like, and the baseline is 200 draws of the same
number of uniformly random timestamps.

```
39 videos scored
mean distance from a human keyframe to the nearest mark
    ours    0.79s
    random  1.36s
ours closer on 37 / 38 videos          one-sided sign test  p < 0.0001
```

**The marks find human-annotated change moments, and they beat chance
decisively.** This is an external, repeatable result on data from another
research group, and it is the strongest evidence the detector has.

### Why this does not contradict the deictic result

[do-marks-help.md](do-marks-help.md) found the marks *losing* to random at
pointing to the moments that needed a frame (median 8.9s vs 4.7s). Both results
are correct and they measure different targets:

| Target | Nature | Marks |
|---|---|---|
| GUI-World keyframes | *user acted, the screen changed* — transitions | **win, p < 0.0001** |
| Deictic sentences | *speaker explains what is already on screen* — plateaus | **lose to random** |

A change detector finds changes. It does not find explanations, and explanation
happens after the change, while the screen sits still. Neither result is a bug;
together they say precisely what the marks are for.

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
- Marks match human change annotations far better than chance on an external,
  third-party benchmark (p < 0.0001, 39 videos).
- 1 fps is the right default, on evidence rather than convenience.
- The pipeline runs unmodified on 1920x1080 OBS captures and 2940x1912 macOS
  captures, on clips from 16 seconds to 74 minutes.

**Not established:**

- GUI-World clips are short (median ~15s, dense actions). Long-form sparse
  change is only tested on the two local Teams recordings.
- The eight local videos have no ground truth; their metrics are self-supervised
  and can only compare schemes, never confirm that a mark is *interesting*.
- Nothing here re-tests whether marks help an *agent*. That measurement remains
  the one in [do-marks-help.md](do-marks-help.md), n=8 questions, one recording.
- No lecture, no coding session, no browser-only session.

The closest complement for long-form ground truth is
[MaViLS](https://github.com/andererka/mavils) — 20 lectures, ~66 minutes each,
sentence-to-slide alignment — which needs a Kaggle account and gives boundaries
only at sentence granularity. [datasets-surveyed.md](datasets-surveyed.md) records that survey,
including the eight candidates that were checked and rejected.
