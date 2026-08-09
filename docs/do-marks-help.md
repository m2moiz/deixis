# Do the marks help an agent? A three-arm measurement

[visual-marks.md](visual-marks.md) establishes that the marks land on the
biggest visual changes. That is a claim about the detector. It is not the claim
that matters, which is: **does a downstream agent answer better because the
marks are there?** This is the measurement of that, and the answer is mostly no
— with one specific, fixable reason why.

## Method

**Questions, chosen by rule rather than by taste.** Every sentence in the
reference recording's transcript containing two or more demonstratives ("this",
"here", "that", "these", "there") and 40–200 characters long: 45 candidates.
Then the densest one from each three-minute bucket: ten. Two were dropped as
non-visual (a scheduling remark, a vague aside), leaving **eight questions**
spanning 05:16 to 30:11. These are exactly the sentences the project is named
for — the ones that are worthless as text.

**Answer key** written by extracting the frame at each timestamp and reading it,
**before any arm was run**. Two points per question: one for naming the right
application and page, one for reproducing a specific item from it.

**Three arms**, each a separate agent with no shared context and no access to
the key:

| Arm | Given |
|---|---|
| A | the transcript, with `marks` stripped out |
| B | the transcript **+ marks** |
| C | the transcript + marks + the ability to extract and view any frame |

**Grading was blind.** The three answer sets were stripped of self-identifying
sections, shuffled under a seed, relabelled X/Y/Z, and handed to a fourth agent
with the key and the rubric. The mapping was withheld until after grading.

## Result

| Arm | Score | Correct | Honest "cannot tell" | Fabrications |
|---|---|---|---|---|
| A — transcript only | 5 / 16 | 4 | 3 | 1 |
| B — transcript + marks | 7 / 16 | 5 | 2 | 1 |
| **C — + frame access** | **16 / 16** | **8** | **0** | **0** |

**Marks alone bought two points out of sixteen on eight questions.** That is
inside the noise of an n=8 set, and both A and B fabricated on the same
question — the marks did not prevent the one confident wrong answer either arm
gave.

**Frame access is the whole effect.** Arm C answered every question correctly
with no hedging and no fabrication, from 15 extracted frames.

The blind grader, not knowing which arm was which, described the difference in
kind rather than degree: arm C "is answering a different question… it reports
what it looked at *and rejected*, which is the behaviour of something reading
pixels rather than reconstructing from words," while A and B "fail in the same
place and the same way: wherever the speaker's words describe the artefact he is
*talking about* rather than the one he is *showing*, they follow the words."

Arm B's own verdict, written before it knew its score:

> A mark is a timestamp and a tile count. There is no description, no text, no
> thumbnail. Every application name, page name, heading, label, and value in the
> eight answers above came from the spoken words. […] Not a single "cannot tell"
> became a nameable answer because of them.

## Two findings that change the design

### 1. The marks are worse than random at pointing you at the right moment

For each of the eight questions, how far is the nearest mark?

```
change-ranked marks : median 8.9s from the question    1/8 within 5s
150 random seconds  : median 4.7s (mean over 2000 draws)
random beats or ties the change-ranked marks in 1919/2000 draws  (p = 0.96)
```

This is the ICCV 2025 result — that random sampling matches or beats most
sophisticated frame selection — reproducing exactly, on this recording, for this
task. It was a known risk and it was not checked until after the feature shipped.

The reason is structural, and it is not a bug in the detector: **a large visual
change is a transition, and explanation happens on the plateau after it.** The
marks correctly identify the boundaries; the questions are about the middles.

Used the way the boundaries suggest — "seek to the last mark at or before t" —
the marks do beat random, but thinly:

```
frame at the preceding mark shows the same screen as t:
  change-ranked  97.3% on the eight questions, 94.8% across all 1996 seconds
  random         92.6%                          92.5%
```

### 2. A mark points at the least readable frame in its neighbourhood

Arm C reported that mark frames kept landing on mid-load skeletons and
mid-animation window switches — the highest-scoring mark it sampled (score
10,235) turned out to be a macOS Mission Control animation: maximum pixel churn,
zero content. Measuring that claim across all 150 marks:

```
local instability (tiles differing from the neighbouring frames)
  at a mark            mean 0.099
  midway between marks mean 0.020      <- 5.0x more stable
  a random second      mean 0.028

34 of 150 marks (23%) sit in the top 5% most unstable seconds of the recording
```

**The marks systematically point at the worst instants to look at.** They are
defined as the moments of maximum change, and a moment of maximum change is by
construction a moment when the screen is mid-way between two states.

## What followed: `look`

The fix falls out of the two findings together, and is now implemented. A mark
is a **segment boundary**, so every mark carries a second timestamp — `look`,
the **midpoint of the stretch during which that screen was up**. `t` answers
"when did this screen arrive"; `look` answers "where do I point a camera".

Measured on the reference recording after the change:

```
instability of the frame you would send to a vision model
  at a mark (t)       0.0989
  at its look         0.0223      <- 4.4x more stable, better on 141/150 marks
  at a random second  0.0313

same-screen coverage: find t's segment by its boundary, then take the frame at...
  the boundary itself   0.9476
  the segment midpoint  0.9753    <- what ships
  150 random timestamps 0.9161 +/- 0.0061   (~10 sigma below)
```

One methodological note, because it is the same trap this whole document is
about. The first attempt at that second table scored the `look` points *as if
they were boundaries* — searching for the last `look` at or before `t` — and
returned 0.9102, apparently **worse than random**. That number was an artifact
of the metric, not a property of the scheme: midpoints are not boundaries, so
the last `look` before `t` is frequently in the previous segment. Separating
"which segment is `t` in" (answered by marks) from "which frame represents that
segment" (answered by `look`) is what the code actually does, and measuring it
that way gives the table above. A plausible number from the wrong measurement
nearly buried a working change.

Two things this measurement does not establish:

- **n = 8.** The point estimates are directional, not precise. The 5/16 → 16/16
  gap is far too large to be noise; the 5/16 → 7/16 gap is well within it.
- It tests **point-in-time questions** ("what was on screen at t"), because the
  question set gives timestamps. Arm C noted the marks would earn their keep on
  a different shape — "walk me through what he demoed" — where segment
  boundaries are the answer rather than a means to it. That is untested.

The headline stands on its own, though: **for making a recording answerable, the
frames are the product and the marks are scaffolding.** Adding `extract_frame`
moved the score from 5/16 to 16/16. Everything else here is about pointing it at
better frames.
