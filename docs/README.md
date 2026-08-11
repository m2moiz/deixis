# docs

The reasoning that did not fit in the README, ordered by how likely it is to be
useful.

**If you read one, read [tooling-gaps.md](tooling-gaps.md)** — the standard this
project holds itself to, written out of an audit where five defects shipped past
a fully green test suite.

## What was measured

| | |
|---|---|
| [do-marks-help.md](do-marks-help.md) | **The result that shaped the product.** Three agents, blind-graded, on questions drawn by rule from the transcript's deictic sentences: transcript alone 5/16, plus change marks 7/16, plus frame access **16/16**. The frames are the product; the marks are scaffolding. |
| [generalisation.md](generalisation.md) | Does any of it hold off one recording? 834 third-party GUI recordings with human keyframes (p = 3.5e-94), five 34–85 minute lectures with slide-change ground truth (81% caught within 2s), and the sampling-rate sweep that proved a confident assumption wrong. |
| [visual-marks.md](visual-marks.md) | **Three change detectors built, measured on a real recording, and thrown away** — and the premise failure underneath all three. Why the shipped one has a budget and no threshold. |
| [vlm-legibility.md](vlm-legibility.md) | Can a small local vision model read dense on-screen text? Ground truth written *before* the model ran. The answer is no, and the reason is hallucination rather than resolution. |
| [datasets-surveyed.md](datasets-surveyed.md) | Every public dataset considered as external ground truth, what each one's annotations really are, and why eight were rejected. |

## How it is tested

| | |
|---|---|
| [tooling-gaps.md](tooling-gaps.md) | The 16 practices and 3 tool tiers this project is built to. A general Python checklist, not a jaano one. |
| [resume-gate-design.md](resume-gate-design.md) | How you test a resume that silently restarts, given it produces byte-identical output either way. |
| [mutmut-triage.md](mutmut-triage.md) | Every mutant the suite fails to kill, and why each one is accepted. Read this before adding a suppression. |

## The one idea they share

A check that passes because it is not looking is worse than no check, because
it also buys false confidence. Each document is an application of that:

- **tooling-gaps** is the general form — coverage proves a line ran, mutation
  testing proves something asserted on it.
- **resume-gate-design** is the sharpest instance — an output comparison cannot
  distinguish a working resume from a broken one, so the gate counts decoded
  chunks instead.
- **mutmut-triage** is the discipline turned on the tooling itself — a
  surviving mutant is either a missing test or an argued exception, never a
  number to drive down.
- **visual-marks** is the same idea aimed at a feature rather than a test: three
  detectors produced tidy, plausible summary statistics, and all three were
  wrong. Looking at the frames is what showed it.
- **vlm-legibility** is the ordering that makes a measurement honest — the
  ground truth was written down before the model was allowed to speak.
- **generalisation** is the same discipline turned on the project's own
  conclusions: every number came from one recording until it did not, and the
  external benchmark changed what the project believes about its own defaults.
- **do-marks-help** is the hardest version — grading was blind, the answer key
  was written first, and the feature that took the most work turned out to be
  the one contributing least.
