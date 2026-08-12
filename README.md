<p align="center">
  <img src="assets/banner.png" alt="Dekho Suno Jaano" width="800">
</p>

<p align="center">
<a href="https://github.com/m2moiz/jaano/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/m2moiz/jaano/actions/workflows/ci.yml/badge.svg"></a>
<a href="pyproject.toml"><img alt="python" src="https://img.shields.io/badge/python-3.12+-blue"></a>
<a href="#requirements"><img alt="platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey"></a>
<a href="docs/generalisation.md"><img alt="validated" src="https://img.shields.io/badge/validated-834%20recordings%20%C2%B7%20p%3D3.5e--94-brightgreen"></a>
<a href="docs/do-marks-help.md"><img alt="blind graded" src="https://img.shields.io/badge/blind%20graded-16%2F16-brightgreen"></a>
</p>

---

**An hour of screen recording, made answerable.** `jaano` turns a recording into
a timestamped transcript that acts as an *index into the video* — so an agent
reads cheap text, notices a moment that only makes sense visually, and pulls
that exact frame.

```bash
jaano suno   meeting.mov -o transcript.json        # suno   -- listen
jaano dekho  meeting.mov -t transcript.json        # dekho  -- look
jaano dikhao meeting.mov 431.5 -o frame.jpg        # dikhao -- show me
```

Three Urdu imperatives, in the order the tool works:

| | | |
|---|---|---|
| <div dir="rtl">سنو</div> | `suno` | *listen* — what was said, and when |
| <div dir="rtl">دیکھو</div> | `dekho` | *look* — when the picture changed |
| <div dir="rtl">دکھاؤ</div> | `dikhao` | *show me* — the picture itself |
| <div dir="rtl">جانو</div> | `jaano` | *know* — what you get from all three, and so the command |

<p align="center">
  <img src="assets/pipeline.svg" alt="A recording becomes a transcript and a list of marks; frames are seeked out of the original on demand" width="900">
</p>

## The problem

> *"See **this** column **here**?"*

That is the most information-dense sentence in the meeting, and as text it is
worthless. The referent was on screen and the transcript threw it away. Words
that only mean something in context are the exact failure mode of an audio-only
transcript of a screen-share.

Feeding the whole video to a vision model is the other extreme: a 74-minute
recording is **444,000 tokens** on a native-video model against **10,000** for
its transcript — 44× the cost to answer a question that is usually about one
second of it.

`jaano` keeps the video on disk as a random-access resource and spends tokens
only where the transcript says something visual happened.

## Does it work? Measured, not asserted

| Question | Answer | Evidence |
|---|---|---|
| Does frame retrieval help an agent? | **5/16 → 16/16** on blind-graded questions | [do-marks-help.md](docs/do-marks-help.md) |
| Do the change marks beat chance? | **701/834** third-party recordings, p = 3.5e-94 | [generalisation.md](docs/generalisation.md) |
| On hour-long recordings? | **81%** of slide changes caught within 2s, vs 42% random | [generalisation.md](docs/generalisation.md) |
| Can a small local VLM read a screen? | **No** — 47% recall, 18 fabrications | [vlm-legibility.md](docs/vlm-legibility.md) |
| Where does it fail? | A camera pointed at a screen. Documented, not hidden | [generalisation.md](docs/generalisation.md) |

Every number came from a command. Three change detectors were built, measured
and thrown away before the one that shipped — [that history is written
down](docs/visual-marks.md), because the ones that looked like they worked are
the interesting part.

---

## Status

| | |
|---|---|
| **Suno — transcript** | working. Chunked ASR, resume, optional speaker labels, ~13× realtime |
| **Dekho — change marks** | working. [Validated on 834 recordings and five hour-long lectures](docs/generalisation.md) — though [worth little on their own](docs/do-marks-help.md) for answering questions |
| **Dikhao — frame retrieval** | working. The step that actually makes a recording answerable |
| **Frame description** | **not built**, blocked on a *measured* finding rather than a guess. See [Roadmap](#roadmap) |

---

## Requirements

- **macOS on Apple Silicon.** ASR runs through [`parakeet-mlx`][pmlx], and the
  whisper engine through `mlx-whisper`. Both are Metal-backed; there is no CPU
  or CUDA path.
- **Python 3.12+**
- **ffmpeg** on `PATH`, for anything that is not already a 16 kHz mono WAV.
- [`uv`][uv] for dependency management.

[pmlx]: https://github.com/senstella/parakeet-mlx
[uv]: https://docs.astral.sh/uv/

## Install

```bash
uv tool install git+https://github.com/m2moiz/jaano
```

That puts `jaano` on your `PATH`. With speaker labels (see the caveats below):

```bash
uv tool install "jaano[diarize] @ git+https://github.com/m2moiz/jaano"
```

Model weights (~2.4 GB) download on first run and are cached by
`huggingface_hub`.

**To work on it instead**, clone and sync — but note that `uv sync` installs the
command at `.venv/bin/jaano` and links it nowhere, so from a clone every
invocation is prefixed with `uv run`:

```bash
git clone https://github.com/m2moiz/jaano
cd jaano
uv sync                      # or: uv sync --extra diarize
uv run jaano suno recording.mov -o transcript.json
```

Every command below is written bare (`jaano ...`), which is what an installed
copy gives you. From a clone, prefix each one with `uv run`.

## Usage

`jaano dikhao` prints the path it wrote and nothing else, so it composes:

```bash
open "$(jaano dikhao recording.mov 431.5 -o /tmp/f.jpg)"
```

**If you point an agent at a recording, tell it about `jaano dikhao`.** That one
verb is the difference between an agent reconstructing the screen from what was
said about it and an agent *looking*: measured 5/16 against 16/16 on a
blind-graded question set ([docs/do-marks-help.md](docs/do-marks-help.md)).

`python -m jaano.suno` and `python -m jaano.dekho` still work and are the same
code.

### Suno — transcription

```bash
jaano suno recording.mov -o transcript.json
```

Progress renders live on stderr:

```
    running [##########--------------] 42%  31:12/74:07 audio  elapsed 1:29  eta 2:01  35.4x
```

| Flag | |
|---|---|
| `-o, --out PATH` | where the transcript JSON goes (required) |
| `--status PATH` | write a JSON heartbeat, for runs you detach from |
| `--no-resume` | ignore any checkpoint and start over |
| `--no-diarize` | skip speaker labelling |
| `--require-diarize` | fail rather than degrade if labelling cannot run |
| `--model ID` | override the ASR model; the default follows `--engine` |
| `--engine parakeet\|whisper` | which ASR backend (default `parakeet`) |
| `--language CODE` | whisper only: ISO code, e.g. `ur`. Detected if omitted |
| `--prompt TEXT` | whisper only: seeds the decoder; biases spelling and script |
| `--roman-urdu` | whisper, Urdu, written in Latin. Sets the two flags above |

**Urdu, and anything else parakeet cannot read.** `parakeet-tdt-0.6b-v3` covers
25 languages and they are all European — `ur` is not among them, so an Urdu
voice note comes back as nothing usable. `--engine whisper` swaps in
`whisper-large-v3-turbo`, which reads it:

```bash
jaano suno voice-note.m4a -o transcript.json --roman-urdu
```

Roman Urdu is a **prompt**, not a setting. whisper writes Urdu in Urdu script by
default; seeding the decoder with a Roman Urdu example makes it emit Latin, and
whisper's own condition-on-previous-text carries that across windows. Measured
on 116s of Urdu speech: **275 of 277 words came back in Latin**, English words
left in English where they were spoken in English — which is the point, for
speech that switches mid-sentence. `--roman-urdu` is that prompt plus
`--language ur`; `--prompt` takes your own.

The model matters more than it looks. The full `whisper-large-v3` ignores the
prompt outright — 280 of 280 words in Urdu script, and 218s rather than 85s for
the same clip. Turbo is the default here for that reason, and changing it means
re-measuring.

Two things the whisper engine does not do: it writes **no checkpoint**, so an
interrupted run starts over, and it reports **no progress** between start and
finish — it owns its own window loop and exposes no hook to bank or count one
from. It runs at ~1.4x realtime against parakeet's ~13x. All three are fine for
a voice note and wrong for an hour of lecture, which is why parakeet stays the
default.

It is an extra, because mlx-whisper pulls torch (~250 MB):

```bash
uv tool install "jaano[whisper] @ git+https://github.com/m2moiz/jaano"
```

**Long runs.** An hour of audio is not something you sit and watch, so detach it
and poll the heartbeat:

```bash
jaano suno meeting.mov -o out.json --status run.json &
jq -r '"\(.state) \(.fraction * 100 | floor)% eta \(.eta_s)s"' run.json
```

`state` moves `extracting → running → diarizing → done`, or `failed` with an
`error`. The file is written atomically, so a reader never sees half of one.

**Interruptions are cheap.** A checkpoint is written beside the output every
chunk. Re-running the same command resumes from it; a changed model, source
file, or chunk geometry invalidates it automatically and the run starts over
rather than reusing tokens that describe something else.

### Dekho — change marks

A second pass adds the timestamps where the picture changed most. It is
separate because the transcript arrives at ~13x realtime and this decodes every
sampled frame at ~10x, so coupling them would make the fast half wait:

```bash
jaano dekho recording.mov -t transcript.json
```

| Flag | |
|---|---|
| `-t, --transcript PATH` | the transcript to add marks to (required) |
| `-o, --out PATH` | write elsewhere instead of overwriting `--transcript` |
| `--budget N` | how many marks to keep (default 150) |
| `--min-gap SECONDS` | how far apart they must be (default 5) |
| `--fps N` | frames sampled per second of video (default 1) |
| `--delta N` | grey levels a tile must move to count (default 8) |

`--budget` rather than a sensitivity threshold is the one design decision worth
knowing about, and it is not a preference — three threshold-based detectors were
built and measured against a real recording before this one, and all three
failed. [docs/visual-marks.md](docs/visual-marks.md) has the numbers.

### Dikhao — retrieving a frame

```bash
jaano dikhao recording.mov 431.5 -o frame.jpg [--width 1500]
```

| Flag | |
|---|---|
| `-o, --out PATH` | where the image goes; `.jpg` is what a vision model wants |
| `--width N` | scale to N pixels wide, aspect preserved (default 1500; `0` keeps source) |

Nothing is precomputed and nothing is cached — the video is already on disk, and
seeking into it is cheap. The 1500px default is a measured ceiling rather than a
taste: a full 2940px frame is ~776 KB as a JPEG, more than most vision APIs
want, and legibility stopped improving well below that — 700px to 1600px moved
recall by one string in fifteen ([docs/vlm-legibility.md](docs/vlm-legibility.md)).

Also available as `jaano.media.extract_frame(video, t, dest, width=...)`.

## Output

```jsonc
{
  "audio": "/path/to/recording.mov",   // the SOURCE, not a temp wav
  "model": "mlx-community/parakeet-tdt-0.6b-v3",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],   // only when diarization ran
  "diarization": "senko 0.1.0",               // absent if it did not
  "text": "the whole transcript as one string",
  "sentences": [
    {
      "start": 12.34,
      "end": 15.02,
      "speaker": 0,                     // index into `speakers`
      "text": " See this column here.",
      "tokens": [{"t": 12.34, "w": " See"}, {"t": 12.51, "w": " this"}]
    }
  ],

  // added by `python -m jaano.dekho`; absent until that pass has run
  // t    = when the picture changed (the boundary)
  // look = the frame worth extracting: the middle of the stretch that screen was up
  "marks": [{"t": 417.0, "score": 2841, "look": 431.5}],
  "marks_meta": {"budget": 150, "min_gap_s": 5.0, "fps": 1.0, "delta": 8,
                 "grid": [128, 84], "frames_sampled": 1997, "source": "..."}
}
```

Two things about this shape are deliberate:

- **`speaker` is an integer index, not a name.** It costs 1.7% of file size
  where `"SPEAKER_01"` on every sentence costs 3.3% for no extra information —
  and an integer *looks* like the arbitrary cluster id it is, where a name
  invites you to treat it as an identity.
- **`diarization` is absent when the pass did not run.** Without that, "no
  diarization" and "diarization found one speaker" are the same document.
- **`marks` sits beside `sentences`, not inside them.** A mark is a fact about
  the video at a time, not about a sentence; nesting it in whichever sentence
  happens to span that second would invent a relationship nothing observed.
  `marks_meta` travels with it for the same reason `diarization` does — marks
  from a budget of 150 and marks from a budget of 20 are different documents.
- **`look` is not `t`, and a consumer wanting a frame should use `look`.** A
  mark is by construction the moment of maximum change, which is the moment the
  screen is halfway between two states — mid-load skeletons and half-drawn
  window switches. Measured: a frame at `t` differs from its neighbours in 9.9%
  of tiles against 2.2% at `look`. Use `t` to know *when*, `look` to know
  *where to point a camera*.

## Development

```bash
just test        # fast lane, ~10s
just typecheck   # pyright strict, package and tests
just check       # THE gate: types, lint, fast tests. What CI runs.
just verify      # everything incl. the end-to-end gates, with coverage
just mutate      # mutation testing over the pure modules
```

`just verify` is the session-close gate and takes ~20 minutes: it runs real ASR
and diarization, including two end-to-end gates that deliberately re-prove they
can fail. That cost is the point — see [docs/tooling-gaps.md](docs/tooling-gaps.md).

| Doc | |
|---|---|
| [docs/tooling-gaps.md](docs/tooling-gaps.md) | the practices and tools this project is built to, written out of an audit where five defects shipped past a green suite |
| [docs/resume-gate-design.md](docs/resume-gate-design.md) | how you test a resume that silently restarts, given it produces byte-identical output |
| [docs/mutmut-triage.md](docs/mutmut-triage.md) | every surviving mutant and why it is accepted |
| [docs/visual-marks.md](docs/visual-marks.md) | the three change detectors that were built and measured before this one, and why each failed |
| [docs/generalisation.md](docs/generalisation.md) | two external benchmarks with human ground truth — 834 GUI recordings (p = 3.5e-94) and five hour-long lectures (81% of slide changes caught within 2s) |
| [docs/do-marks-help.md](docs/do-marks-help.md) | do the marks actually help an agent? Three arms, blind-graded: 5/16 → 7/16 → 16/16 |
| [docs/vlm-legibility.md](docs/vlm-legibility.md) | whether a small local VLM can read a screen frame. Measured: not this one |

---

## Why it is built this way

### The idea

Not "extract everything from the video." The video stays on disk as a
random-access resource, and the transcript is its index.

```
video ──► audio ──► timestamped transcript      (cheap, ~10k tokens, read fully)
  │
  └────── frames on demand, at timestamps the transcript justifies
```

An agent reads the transcript, hits a deictic moment, and asks for that instant.
Nothing else is ever extracted. Cost scales with what is *interesting*, not with
video length.

This inverts the usual pipeline. Scene-detect-then-OCR-everything is a **push**
design: extract all visual content upfront, pay for it whether or not any of it
mattered, then try to compress.

The false-positive half of that argument turned out to be **measurably true** —
a perceptual-hash detector fired on cursor movement alone on a real recording,
which is exactly what was predicted here before anyone had run one. The
conclusion drawn from it was too strong, though. Visual marks *do* enumerate the
video, and cheaply: ranking every sampled frame costs 33 ms of arithmetic on top
of a decode, and produces 150 timestamps and no descriptions. Enumerating is not
what makes the push design expensive; **describing** everything you enumerated
is. Marks are a table of contents, and the transcript still decides which
entries are worth opening.

### Decisions made so far

**ASR: Parakeet TDT 0.6b v3 via `parakeet-mlx`.**
Parakeet over Whisper for two structural reasons: it does not hallucinate on
silence — it predicts token *duration* and skips gaps rather than grinding every
frame — and that same mechanism makes its timestamps **native** rather than
recovered after the fact by DTW alignment the way Whisper's are. Native
timestamps are the whole product here: the transcript is only an index if its
times point at the right instants.

The MLX build was chosen over two other local copies of the same model. Handy
ships a NeMo ONNX export and FluidAudio a CoreML `.mlmodelc` of v2; same
weights, three incompatible formats. MLX wins on being Metal-native on the
target hardware and on `parakeet-mlx` exposing per-token alignment directly.

**Chunking is not optional at meeting length.** `parakeet-mlx` defaults
`chunk_duration` to `None`, which feeds the whole file to Metal in one buffer —
an hour of audio asks for ~14.5 GB against a ~9.5 GB max buffer and dies. jaano
drives the chunk loop itself at 120s with 15s overlap, which is also what makes
per-chunk progress and resume possible.

**Not OCR.** OCR flattens the frame to text and throws away everything that
carried the meaning — layout, table structure, what is highlighted, what the
cursor is resting on, the shape of a chart. Frames go to a vision model.

**Execution provider: CPU, not CoreML.** Measured on a 45s clip:

| Provider | Speed |
|---|---|
| `CPUExecutionProvider` | 45s audio in 2.3s — **19.7x realtime**, 1.1s load |
| CoreML | 45s audio in 33.9s — 1.3x realtime |

CoreML fragmented the encoder into 319 partitions across 3,249 nodes, supporting
only 1,528 of them; the marshalling cost swamped the compute. On a quantised int8
model the plain ONNX Runtime CPU kernels win outright. General lesson: a
triple-digit partition count in an EP load warning means the accelerator is
hurting you — benchmark against CPU before assuming otherwise.

**Ingestion: jaano runs ffmpeg itself, though it does not have to.**
`parakeet-mlx` already shells out to ffmpeg for any input, so a `.mov` would
load without a line of code here. It is done in `jaano/media.py` anyway for
two things that call cannot give: progress during the tens of seconds an
hour-long 4 GB recording takes to demux (~1000x realtime, measured on a
74-minute wav), and an error you can act on — parakeet's own failure surfaces
as the ffmpeg build banner with the diagnosis buried in it.

The extracted wav lives in a temp directory for the length of the run and is
not cached. Extraction is a small fraction of wall time, and caching would mean
inventing invalidation and leaving 135 MB files next to the user's recordings.
Anyone who wants the wav can extract it by hand and pass it — the conversion
step is skipped when the input is already mono `pcm_s16le` at the model's rate.

**Vision, not OCR — but which vision model is now an open question.**
The plan named Qwen2.5-VL-7B-Instruct on `mlx-vlm`, on the strength of published
DocVQA and ChartQA scores. That model has still never been run here. What *has*
been run is `gemma-4-e2b-it-4bit`, on five frames from a real recording with
ground truth written down first: 47% recall and 18 fabricated strings, at
18-22 s per image. Benchmark leadership on document QA did not survive contact
with a screenshot of an inbox — see
[docs/vlm-legibility.md](docs/vlm-legibility.md) — so the model choice is
deliberately reopened rather than carried forward.

**Speaker labels: senko, optional, and wrong in ways worth naming.**
Diarization is an extra — the `[diarize]` install above — not a dependency. It pulls
29 packages (scikit-learn, scipy, umap-learn, hdbscan, coremltools, numba) for
one optional pass, it is a source build, and on Darwin it forces
`device='coreml'` unconditionally, so a core dependency here would be a core
dependency that breaks `uv sync` for anyone not on Apple Silicon. Without it the
transcript is written exactly as before and the run exits 0.

senko over pyannote for one measured reason: it imports no torch, and on the
74-minute reference recording it diarized in **8.0s — 554x realtime**, plus **12.4s**
of warm model load. Against 149s of ASR that is **+14%**. The first run on a
machine pays **~51.5s** instead of 12.4s while CoreML compiles and persists its
embeddings cache; that is a one-off, and it is not the steady state a first-time
user should be shown as one.

Labels land on the *sentence*, as an integer index into a top-level `speakers`
array. Measured on the real transcript, that costs **1.7%** of file size; the
string `"SPEAKER_01"` on every sentence costs 3.3% to carry no extra
information, and a per-token label costs **20.4%** — 95 KB, ~24,000 tokens of
agent context — and was rejected. An integer also *looks* like the arbitrary
cluster id it is, where a name invites a reader to treat it as an identity. A
`diarization` provenance string sits beside `speakers`, and is **absent** when
the pass did not run: without it, "diarization was not run" and "diarization ran
and found one speaker" are the same document.

Sentences are labelled by counting **token votes** against turns, not by
intersecting the sentence's span with them. A span includes its internal
silences, so a speaker talking during a pause mid-sentence can hold more of the
span than the speaker who said the words. On the reference recording the two
approaches disagree on 6 of 664 sentences — this is a small correctness win, not
a large one, and it is kept because interval overlap fails worse the more
interleaved the speech gets.

What it gets wrong, all of it measured on a 74-minute reference recording
whose ground truth is two speakers:

- **Over-clustering.** senko found **three** speakers where there are two. The
  phantom holds 58.5s across 13 turns and **wins zero of the 664 sentences**
  under the token vote — the sentence-level schema absorbs it, where a
  per-token one would have put a nonexistent third speaker in front of the
  agent 13 times. senko exposes **no `num_speakers` knob**; clustering is
  unsupervised and the only parameter is `mer_cos`. A file where a phantom
  cluster *does* win sentences would produce a transcript with a speaker who
  does not exist.
- **Interruptions are misattributed.** parakeet segments on punctuation and
  pauses, not on voice, so one sentence can span a speaker change — 69 of 664,
  **10.4%**, do. The vote hands the whole sentence to whoever contributed more
  tokens, so a short interjection vanishes into the surrounding speaker and a
  long one takes their words with it. Splitting a sentence at a speaker change
  is the honest fix and is not in this tool.
- **Back-channels are invisible.** No turn on the reference file is shorter than
  1.01s, and senko's merged segments are non-overlapping, so the format cannot
  represent a "yeah" spoken over someone. It is attributed to whoever holds the
  floor. For an index whose job is to find *"see this column here"* — a
  floor-holder utterance — that is the right failure, but it is a failure.
- **Numbering is arbitrary and per-file.** `SPEAKER_01` in one recording has
  nothing to do with `SPEAKER_01` in another. It is a clustering artifact, not a
  speaking order and not a person.

---

## Roadmap

Frame *selection* is answered. Frame *description* is not.

**What the selection work settled** (numbers and method in
[docs/visual-marks.md](docs/visual-marks.md)):

- The pHash worry recorded here previously was right about the symptom and
  wrong about the direction. A 9x8 dHash did not under-trigger on "same doc,
  different cell" — it fired on **pointer movement alone** and missed a ticked
  checkbox, because 327x239-pixel cells resolve nothing smaller than a window
  switch. ffmpeg's `mpdecimate`, the standard duplicate dropper, failed the
  opposite way and kept 1997 of 1997 frames.
- The deeper finding is that **no threshold can work**, on any detector: during
  an active session the screen changes most seconds, so "the screen changed" is
  not a rare event and cannot be an index. Ranking under a fixed budget replaces
  it and needs no per-video tuning.
- **No eval set was needed.** The plan called for 50–100 hand-labelled frame
  pairs before anything could be trusted. A budget has no threshold to
  calibrate, so the thing the eval set existed to justify does not exist. Two
  synthetic controls — a static video that must yield nothing, a two-cut video
  that must yield exactly its two cuts — pin the behaviour instead.
- **DINOv2 was not needed either**, and the reason is cheaper than expected:
  mean-pooling on the way down to a 128x84 grid already suppresses smooth motion
  (webcam tiles, faces) while keeping the high-contrast edges of text and UI.

**What blocks description**, now measured rather than assumed
([docs/vlm-legibility.md](docs/vlm-legibility.md)):

- `gemma-4-e2b-it-4bit` recovered 33 of 70 hand-written ground-truth strings
  across five real frames (47%) while fabricating 18 — and the fabrications are
  fluent, plausible UI text (`Your GPS aren't the problem`, `OpenAI Browser`),
  which is the kind that poisons an index silently. Resolution is not the fix:
  700 → 1600 px moved recall by one string.
- It is also 18–22 s per image, not the ~5 s assumed from an earlier benchmark
  whose outputs were 23–26 tokens long.
- Untested and worth trying before concluding anything general: a larger local
  model (`Qwen3-VL-4B`), a terse prompt with a larger token budget, and a cloud
  VLM — none of which have been run.
- **ColPali / ColQwen2** (late-interaction retrieval over page images, no OCR)
  remains the interesting alternative, but no MLX port exists and nobody reports
  running `colpali-engine` on Apple Silicon MPS. Experiment, not infrastructure.

### Prior art

Nothing does "long screen-recording in → deduped, VLM-described,
timestamp-queryable index out". `HKUDS/VideoRAG` is active and closest in spirit
but shaped for general long-video QA. The video-analysis MCP servers that exist
are OCR-based, which is the approach this tool rejects.

## Layout

```
jaano/            the package
  cli.py           the `jaano` command: suno | dekho | dikhao
  suno.py          suno   -- ASR orchestration, chunking, resume
  dekho.py         dekho  -- the moments the picture changed, ranked under a budget
  media.py         ffmpeg: audio out, tile grids out, dikhao frames out
  chunking.py      the chunk loop parakeet-mlx does not provide
  checkpoint.py    resume, and the validated boundary that reads it
  merge.py         token-vote speaker labelling
  asr.py           the one shape both ASR engines return
  whisper.py       the whisper engine, and why Roman Urdu is a prompt
  diarize.py       the fail-soft senko boundary
  atomic.py        write-or-do-not-write, for files a reader may be watching
tests/             fast unit tests, plus the two slow end-to-end gates
scratch/           working probes; the data beside them is gitignored
docs/              the reasoning that did not fit here
```

## License

[GNU AGPL-3.0-only](LICENSE). Use it, change it, run it — but if you run a
modified version as a network service, its users get the source.
