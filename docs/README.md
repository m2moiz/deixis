# docs

Reasoning that did not fit in the top-level README, in the order it is most
likely to be useful.

| | |
|---|---|
| [tooling-gaps.md](tooling-gaps.md) | **Start here.** The 16 practices and 3 tool tiers this project is built to, written out of an audit where five defects shipped past a fully green test suite. It is a general Python checklist, not a deixis one. |
| [resume-gate-design.md](resume-gate-design.md) | How you test a resume that silently restarts, given that it produces byte-identical output. The design behind `scratch/resume_gate.py` and `tests/test_resume_gate.py`. |
| [mutmut-triage.md](mutmut-triage.md) | Every mutant the test suite fails to kill, and why each one is accepted. Read this before adding a suppression. |
| [visual-marks.md](visual-marks.md) | Three change detectors built, measured on a real recording, and thrown away — and the premise failure underneath all three. Why the shipped one has a budget and no threshold. |
| [vlm-legibility.md](vlm-legibility.md) | Whether a small local VLM can read dense on-screen text. Ground truth written before the model ran; the answer is no, and the reason is hallucination rather than resolution. |

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
