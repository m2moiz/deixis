# Visual marks: what was measured, and what it killed

`deixis/frames.py` writes a bounded list of timestamps into the transcript at
the moments the picture changed most. This is the record of how that design was
arrived at, including the three approaches that were built, measured, and
thrown away. Every number here came from a command; nothing is estimated.

**The reference recording** throughout is
`Screen Recording 2026-08-06 at 15.07.10.mov` — 2940x1912, 33 minutes, 60,065
frames, a Microsoft Teams call in which a browser form, Outlook and Mission
Control are all visited. It is a real working session, not a fixture. Sampled
at 1 fps it is 1,997 frames.

## The premise that was wrong

The plan going in was: hash each sampled frame, mark the timestamps where the
hash moved more than a threshold, and merge those into the transcript. It rests
on an assumption nobody had checked — **that a screen recording is mostly
static, so a change is a rare and therefore informative event.**

On this recording the median one-second interval already moves 38 of 10,752
tiles. The screen changes *most* seconds. Everything below follows from that one
measurement.

## Three detectors, three failures

### 1. Perceptual hash — too coarse

A 9x8 dHash over an ffmpeg pipe, which is what every off-the-shelf tool for
this implements (`video-sampler`, `videostil`, `framewise`, `peepshow`,
PySceneDetect's `detect-hash`, `imagehash`).

```
frames sampled : 1997  (fps=1)
consecutive Hamming: mean=1.4 p50=0 p90=5 p99=13 max=46
identical pairs    : 1185 / 1996

thresh    kept   %kept  per-min
     5     203   10.2%      6.1
     8      74    3.7%      2.2
```

Those numbers look like a working detector. They are not. Extracting the frames
at the marks and at the un-marks and **looking at them** shows:

| Frame | Verdict | What actually changed |
|---|---|---|
| t=600, marked | true positive | Mission Control → Outlook inbox |
| t=995, marked | weak true positive | text typed in a field, speaker tile switched |
| **t=379, marked** | **false positive** | **nothing but the mouse pointer moving** |
| **t=1195, unmarked** | **miss** | **a checkbox ticked, "Saved ✓" → "Saving…"** |
| t=1812, unmarked | true negative | only the speaker-tile highlight |

A 9x8 grid over a 2940x1912 frame gives 327x239-pixel cells. That resolves an
application switch and nothing smaller — and because dHash compares *adjacent*
cell brightness, near-equal cells on a white form flip bits from almost any
perturbation, which is how a pointer trips it.

This also lines up with the published literature: screen-content images are
composed of text and graphics with different statistics from natural images,
and hashing methods designed for natural images are documented as unsuitable
for them.

### 2. ffmpeg `mpdecimate` — too sensitive

The industry-standard duplicate-frame dropper, 8x8-pixel block SAD with
`hi`/`lo`/`frac` thresholds.

```
fps=1, mpdecimate (defaults)                  kept 1997 / 1997
fps=1, mpdecimate=hi=1536:lo=640:frac=0.5     kept 1997 / 1997
fps=1, crop out the webcam strip, defaults    kept 1592 / 1997
```

Zero frames dropped at any setting. Cropping the video-call tiles out removes
20%, which proves they contribute — and leaves 80% still surviving, which
proves they are not the whole story. `mpdecimate` exists to remove *encoder*
duplicates; any real pixel motion defeats it.

The two off-the-shelf options therefore fail in opposite directions and neither
is tuneable into the middle, because both collapse a frame to a single decision
with no notion of how large the change was or where.

### 3. Tile grid with a learned activity mask — the right shape, wrong output

Reimplementing SeeAction's change-region idea (per-pixel SSIM map → connected
components → change-region bounding boxes; that paper samples screencasts at
5 fps) on an ffmpeg+numpy pipe: decode to a 128x84 grid, diff tiles, suppress
tiles that change chronically, mark when a large enough connected region moved.

A sweep of 54 configurations — 3 grid sizes x 3 deltas x 2 mask levels x 3
minimum cluster sizes — contains **no row** that marks t=1195 without also
marking t=379:

```
    grid  delta   act  clust  marks   /min  masked%     379    1195
  128x84     12  0.20      3   1659   49.8     0.0%    MARK    MARK
   64x42     12  0.20      5    845   25.4     0.0%    MARK       -
   32x21     12  0.20      3    574   17.2     0.0%       -       -
```

Two things fell out of that run, both of which contradicted the design:

**The activity mask never engaged** (`masked% ≈ 0.0`), and the reason inverts
the argument for having one. Per-row change frequency at delta=4:

```
row 20  y~455px  0.018  #      <- webcam tiles
row 24  y~546px  0.026  ##     <- webcam tiles
row 64  y~1456px 0.065  #####  <- browser content
row 68  y~1547px 0.059  ####   <- browser content
```

**The webcam is not the noisy region.** Mean-pooling a ~23x23-pixel tile already
destroys smooth low-contrast motion — a face, a room — while preserving the thin
high-contrast edges that text and UI chrome are made of. The noisiest tiles are
the *content*. The masking stage was designed to solve a problem the downscale
had already solved, and it was deleted. (This is also, incidentally, what the
screen-content hashing literature prescribes on purpose: hash where the
gradients are.)

**And no threshold can work**, because of the premise failure at the top: the
score at t=379 (pointer only) is 75 and at t=1195 (checkbox) is 129, against a
p90 of 1,092. Both are noise-floor events. There is no cut that separates them,
and tuning against them was tuning against the wrong pair.

## What shipped: rank under a budget

Score every interval by changed-tile count, sort descending, take the top
`budget` while keeping picks `min_gap_s` apart.

```
score distribution: p50=38 p90=1092 p99=2762 max=10235

top-150, min gap 5s:
gaps     : median 9s, p90 27s, max 79s
coverage : 31 / 30 / 25 / 25 / 14 / 11 marks per five-minute bucket
```

Well spread, no clumping, no dead stretches. The **weakest** of the 150 picks
(rank 150, score 492) was extracted and inspected by eye: a real form scroll
from "drafting/writing · meeting notes" to "research · data analysis · deck
building", with "Saving…" → "Saved ✓". The marginal pick is a real event, so
the budget is if anything conservative.

Why a budget beats a threshold, stated generally: **a threshold encodes an
assumption about how often the content changes, and that assumption is a
property of the recording, not of the detector.** A budget makes no such
assumption. A static lecture yields 150 well-separated marks at the slide
transitions; a frantic screen-share yields the 150 largest transitions in it.
Same code, same bounded output, no per-video tuning.

## Cost

| Step | Reference recording (33 min) |
|---|---|
| ffmpeg decode + downscale to 128x84 @ 1 fps | **518s** |
| decode + downscale to 9x8 @ 1 fps | 205s |
| `-hwaccel videotoolbox`, 9x8 | 437s — **worse**; readback dominates |
| tile diff + ranking, 1,997 frames | **33 ms** |

Decode is the entire cost and the analysis is a rounding error on it — four
orders of magnitude apart. If this ever needs to be faster the lever is the
decode strategy (lower `fps`, keyframe-only extraction), never the arithmetic.
Note that the finer grid costs 2.5x the decode of the coarse one: the scaler,
not the pixel count coming out.

## How the gate can fail

Per practice #14 in [tooling-gaps.md](tooling-gaps.md), a check that cannot fail
is not a check. Both halves of the failure space are pinned in
`tests/test_frames.py`:

- `test_static_video_yields_no_marks` — a video where nothing happens must
  produce zero marks. Without it, a full budget on a real recording would prove
  only that the budget was spent.
- `test_marks_land_on_the_real_transitions` — a fixture cutting colour at t=4
  and t=8 must be marked at exactly `[4.0, 8.0]`. A detector marking every
  second would satisfy "both are marked", so the assertion is equality, not
  membership.

Twelve mutants were injected by hand against this suite. Ten were killed on the
first pass; the three that survived each pointed at something real and are
worth recording, because two of them were defects in the *tests* and one was a
defect in the *code's justification*:

| Mutant | Outcome |
|---|---|
| Remove the per-frame `.copy()` in `extract_tile_grid` | **Survived — the code was wrong, not the test.** Each `read()` returns a fresh `bytes` the view keeps alive, so there was no aliasing to defend against. The copy and its confident comment were deleted. |
| `int16` → `uint8` subtraction (wrap) | Survived. The existing test used 250→10, which wraps to 16 — still over delta, so the count came out right for the wrong reason. Killed by 248→0, which wraps to exactly 8 and reads as *unchanged*: a screen going black would have scored zero. |
| `kind="stable"` → `kind="quicksort"` | Survived. numpy's introsort is stable by accident on small inputs, so the three-element tie test pinned nothing. Killed at 305 elements, where the two sorts pick index 5 and index 152. |
| Gap `>=` → `>`, and dropping the `1/fps` conversion | Survived initially — every gap test ran at 1 fps, where seconds and frame indices coincide. Killed by a 2 fps case and an exact-boundary case. |

## What this does not do

No frame is described. A mark says *something happened at t*, which is enough to
navigate by and costs no tokens. Attaching a description to each mark is the
next piece of work, and it is blocked on a measurement rather than on code — see
[vlm-legibility.md](vlm-legibility.md).
