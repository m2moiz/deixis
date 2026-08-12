# Can a small local VLM read a screen recording? One of them can, slowly

Describing the marked frames is the obvious next step after
[visual-marks.md](visual-marks.md), and the plan was to do it locally: free,
private, no API key, no uploads. Before building it, one question had to be
answered, because everything downstream depends on it — **can a 2–4B model
running on this machine actually read dense on-screen text?**

The first model tried could not, and the write-up below records that. A second
round on 2026-08-12 tested the three things the first round named as untested;
one of them works. **[Jump to the second round](#second-round-2026-08-12)** for
the answer and the numbers; the first-round detail stays because it is what the
second round is measured against.

Measured 2026-08-08 on the M2 MacBook Air (16 GB, fanless). Raw output,
driver and hand-written ground truth are under `scratch/vlm/`.

## Setup

- Model: `mlx-community/gemma-4-e2b-it-4bit` (4-bit, 3.3 GB, already cached)
- Runtime: `mlx-vlm` 0.6.10, in a venv separate from the project's
- Prompt: *"Transcribe all text visible in this screenshot, then describe what
  application is shown and what the user is doing. Be exhaustive about the
  text."*, `max_tokens=400`
- Decoding was **greedy** — `mlx_vlm.generate.ar.DEFAULT_TEMPERATURE` is `0.0`,
  verified rather than assumed. Every reading below is the model's argmax, so
  none of the errors are sampling noise.
- Five frames from the reference recording. **Ground truth was hand-written
  from the images before the model was run**, which is the only ordering that
  does not bias it toward whatever the model happened to say.

## Result

| Frame | Content | Ground truth | Recalled | Hallucinated | Seconds |
|---|---|---|---|---|---|
| `DET_00600` | Outlook inbox, dense FR/EN list | 15 | **4** | 7 | 41.4 |
| `DET_00995` | Web form, one question + answer | 13 | **8** | 2 | 19.9 |
| `UNDET_01195` | Web form, radio row + 7 checkboxes | 15 | **10** | 2 | 21.6 |
| `UNDET_01812` | Google Form, 6x5 radio matrix | 14 | **2** | 4 | 19.9 |
| `TOPN_01129` | Web form, usage-frequency matrix | 13 | **9** | 3 | 18.1 |
| **Total** | | **70** | **33 (47%)** | **18** | |

Resolution is not the binding constraint. The same frame re-extracted at three
widths:

| Width | Recalled / 15 | Hallucinated |
|---|---|---|
| 700 px | 5 | ~5, plus heavy structural fabrication |
| 1100 px | 4 | 7 |
| 1600 px | 6 | 5 |

Tripling the pixel count moved recall by one string. More pixels did let it read
some genuinely small text it had missed, but it lost others and invented at the
same rate.

## Why the verdict is "no", and it is not the recall number

47% recall on its own would be a starting point. The fabrications are the
problem, because they are **fluent, correctly formatted, plausible UI text** —
indistinguishable from a correct reading unless you have the image open beside
it:

> Image: `Your GPUs aren't the problem. Your storage …`
> Model: `Your GPS aren't the problem. Your storage …`

> Image: `Votre Kit AI Transformation est prêt — 8 livrat`
> Model: `Votre Kit Air Transformation est prêt - 8 livré`

Browser chrome present in every frame — identical pixels — came back as
`OpenCLI Browser` on three frames, `OpenGL Browser` on one, and `OpenAI Browser`
on another. Under greedy decoding, that spread means surrounding context alone
flips what the model reads off the same pixels.

An index built on this would silently contain a meeting where someone discussed
GPS and an OpenAI browser. A wrong description is worse than no description,
because a mark with no description is honestly empty and a mark with a
fabricated one is not.

Two structural failures matter as much as the string errors:

- On the checkbox frame it transcribed all seven options and **omitted the
  question above them** — `Verification, judgment & data safety`, with `Basic`
  selected. The answer, dropped; the menu, kept.
- On the 6x5 Google Form matrix it returned **none of the five column headers**
  (`<2hr`, `>2 & <4hr`, …). A matrix without its column axis is not a lossy
  record of the screen, it is an unusable one.

And the reading pane of the Outlook frame — `Hello Moiz`, `Here my feedback`,
`1 Entry point` — was recalled 0/4 at every resolution. That is the actual
message content, the part a describer exists to capture.

## Speed, also worse than assumed

18–22 s per image at 400 tokens, steady state, after a 10.5 s model load. The
planning estimate was ~5 s, extrapolated from a benchmark whose outputs were
23–26 tokens; at 10.6 tok/s the 400-token budget is most of the difference.
At 150 marks that is **45–55 minutes per 33 minutes of video**, not the
~1:1 the plan assumed.

## What this does not rule out

- **A larger local model.** Only one was tested. `Qwen3-VL-4B-Instruct-4bit` is
  a 2.5 GB download away and is the obvious next candidate; the project's own
  README has long named Qwen2.5-VL-7B, which was never actually run.
- **A terser prompt with a larger token budget.** The model spends its budget on
  markdown scaffolding (`**Main Content Area (List/Table — Continued):**`) and
  runs out mid-frame on dense screens. "Output only the text, no headings" with
  more tokens is untested and cheap to try.
- **A cloud VLM.** Not tested here at all.

What it *does* rule out is shipping a describer on this model as it was
configured, and it rules out temperature as the remedy — the run was already
greedy, so the garbling is the model's best guess and not noise.

## Second round, 2026-08-12

All three of the above were tested, on the **same five frames against the same
ground truth file, unedited**. Same machine, same runtime, same greedy decoding.

Recall is now scored by `scratch/vlm/score.py` rather than by hand. It implements
the rule the first round wrote down and nothing more — case-insensitive substring
match, whitespace collapsed, markdown and table markers stripped. Deliberately no
fuzzy matching: a fuzzy matcher would score `Your GPS aren't the problem` as a hit
for `Your GPUs …`, and that substitution is the failure this measurement exists to
catch. Run against the first round's raw output it reproduces its numbers exactly
— 33/70 overall, 4 / 8 / 10 / 2 / 9 per frame — so the scoring rule did not move
and the rows below are comparable. Hallucinations are still counted by hand,
because counting them needs the image, not a string list.

`run_config.py` takes model, prompt and token budget as arguments; `run.py` is
left pinned to the first round's configuration.

| # | Model | Prompt | `max_tokens` | Recalled / 70 | Hallucinated | Mean s/frame |
|---|---|---|---|---|---|---|
| 1 | `gemma-4-e2b-it-4bit` | descriptive | 400 | **33 (47%)** | 18 | 24 |
| 2 | `gemma-4-e2b-it-4bit` | descriptive | 1200 | **33 (47%)** | 18 | 45 |
| 3 | `gemma-4-e2b-it-4bit` | terse | 1200 | **17 (24%)** | 14 | 20 |
| 4 | `Qwen3-VL-4B-Instruct-4bit` | descriptive | 1200 | **60 (86%)** | 18 | 156 |
| 5 | `Qwen3-VL-4B-Instruct-4bit` | terse | 1200 | **56 (80%)** | 13 | 124 |

Row 1 is the first round, rescored. "Descriptive" is its prompt verbatim; "terse"
is *"Output only the text you can see in this screenshot, exactly as written. No
headings, no description, no commentary, no explanation. Text only."*

### The token budget was not the constraint

Row 2 tripled the budget and changed nothing: same 33/70, the same per-frame
4 / 8 / 10 / 2 / 9, the same fabrications, the same two structural failures — the
Google Form matrix still came back without a single column header, the radio
question `Verification, judgment & data safety` and its selected `Basic` still
absent. The extra 800 tokens bought more markdown scaffolding and 21 more seconds
per frame. The first round's guess that the model "runs out mid-frame" was wrong:
it was not running out, it had already said what it could read.

### The terse prompt is actively harmful, on both models

Rows 3 and 5 are the surprise. Told to emit text and nothing else, **both** models
degenerate into a repeat loop on the dense inbox frame — gemma emitted `Pascal
Nass / Votre Kit Transformation est prêt / PN / Journal fourni / Courriers du
dossier Autres (1)` forty times until the budget ran out (83 s); Qwen emitted
`pascal nass / @ : @ muhammadmoiz.work / Cci : pascal nass` about thirty times
(256 s, its slowest frame of any run).

On the other four frames gemma did the opposite and stopped almost immediately —
3.0 s and 133 characters on `UNDET_01195`, of which the entire content was the
macOS notification in the corner, scoring **0/15**. It also invented framing it
had not invented under the descriptive prompt: on the Google Form matrix it
produced `Welcome to the team.` / `Please select a topic from the list below.`,
neither of which appears anywhere on screen.

The descriptive prompt's headings are not waste. They are what gives the model a
structure to walk and a place to stop.

### The model was the constraint

Row 4 is the result. `Qwen3-VL-4B-Instruct-4bit` (2.5 GB download, 4-bit) on the
unchanged first-round prompt recovers **60 of 70** strings against gemma's 33, and
both of the structural failures reverse:

- All five matrix column headers — `<2hr`, `>2 & <4hr`, `>4 & <6hr`, `>6 & <8hr`,
  `>8hr` — plus all six row labels. That frame goes from 2/14 to **14/14**.
- The radio question and its selected value, `Verification, judgment & data
  safety` / `Basic`, both present. That frame goes from 10/15 to **14/15**.
- The Outlook reading pane — `Hello Moiz`, `Here my feedback`, `1 Entry point`,
  `Which role do you hold?` — the four strings gemma recalled **0/4 at every
  resolution**, all four read correctly. That is the actual message content, the
  part a describer exists to capture.

The hallucination count is unchanged at 18, and that number deserves care rather
than celebration. What changed is *where* they are and *what* they are. Eleven of
the 18 are on the single dense inbox frame, and they are proper nouns, URLs and
brand strings — `OpenCBI Browser` for `OpenCLI Browser`, `Merlyn from Packet` for
`Merlyn from Packt`, `Vushank Pandya` for `Vrushank Pandya`, and a merge of two
adjacent rows into `Hi Daniel Priestley`. Participant names garble on three
frames: `Moiz` came back as `Mat`, `Moaz` and `Mozi`. That is still the dangerous
class — fluent and plausible — but it is now concentrated in the metadata rather
than spread through the body text, which is a different failure to design around.

One defect is worth naming on its own: on the inbox frame the model finished its
transcription and then emitted the integers 1 through 100 as a bulleted list under
`**Other UI Elements:**`. Pure invention, and it is roughly a third of that
frame's 222 seconds.

### The constraint is now speed

156 s per frame, mean over the five, against gemma's 24. Per frame: 222, 97, 140,
193, 125 — the dense frames cost the most, which is also where the marks land.

At the 150 marks a 33-minute recording produces that is **about 6.5 hours of
description for 33 minutes of video**, roughly 12x realtime, on a fanless machine
that will be thermally throttled long before it finishes. The first round already
found gemma at 45–55 minutes per 33 minutes against a plan that assumed ~1:1;
this is eight times worse than that.

Row 5 was run to see whether the terse prompt could buy the time back by cutting
the description prose. It does not: 124 s mean, still 20x gemma, and it costs four
recalled strings. Not a trade worth making.

## Where this leaves the describer

**The capability question is answered, and the answer flipped.** A 4B model on
this machine can read a dense screen frame — 86% of hand-written ground truth,
including the matrix axes and message body that the first round called
"unusable". `Qwen3-VL-4B-Instruct-4bit` with the descriptive prompt is the
configuration to build on. The first round's verdict was correct about the model
it tested and wrong as a general claim about small local VLMs.

**The blocking question is now throughput, not legibility.** 12x realtime is not
a describer anybody runs on an hour of video. That is the next thing to decide,
and it is a different problem with different levers — fewer frames described,
cropping to the changed region rather than passing the full frame, batching, or
accepting an overnight batch job. None of those have been measured.

Two things were deliberately **not** run, with reasons:

- **`Qwen2.5-VL-7B`**, named in the plan since the start. The binding constraint
  is now seconds per frame, and a 7B model on 16 GB is strictly worse on exactly
  that axis. Testing it would answer a question that is no longer the one in the
  way.
- **A cloud VLM.** It was not called and not priced, so this document contains no
  cost figure for it. It stops being the obvious escape hatch the moment a local
  model clears the legibility bar, because the reason for local — no uploads of
  someone's inbox and video call — was never about capability.
