# Can a small local VLM read a screen recording? Measured: no

Describing the marked frames is the obvious next step after
[visual-marks.md](visual-marks.md), and the plan was to do it locally: free,
private, no API key, no uploads. Before building it, one question had to be
answered, because everything downstream depends on it — **can a 2–4B model
running on this machine actually read dense on-screen text?**

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
