# deixis

Makes a long screen recording answerable.

*Deixis* is the linguistic term for words that only mean something in context —
*this*, *here*, *that one*. It is exactly what breaks an audio-only transcript of
a screen-share: **"see this column here"** is the most information-dense sentence
in the meeting and, as text, it is worthless. The referent was on screen.

## The idea

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
mattered, then try to compress. It also drowns in false positives — cursor
movement and scrolling trip content-based scene detection constantly on a
screen-share. Transcript-driven retrieval never enumerates scenes at all, and the
"which of these 60 frames mattered?" judgement is answered by the speaker, out
loud, at a known timestamp.

## Decisions made so far

**ASR: Parakeet TDT 0.6b v3, reusing weights already on this machine.**
Handy ships `parakeet-tdt-0.6b-v3-int8` as a NeMo ONNX export at
`~/Library/Application Support/com.pais.handy/models/`. `onnx-asr` loads that
directory directly — **verified, no extra model download**:

```python
import onnx_asr
p = "/Users/moiz/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8"
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", p, quantization="int8").with_timestamps()
result = model.recognize("audio.wav")
result.tokens, result.timestamps   # sub-second token alignment
```

Parakeet over Whisper for two reasons, both structural: it does not hallucinate
on silence (it predicts token *duration* and skips gaps, rather than grinding
every frame), and that same mechanism makes its timestamps **native** rather than
recovered after the fact by DTW alignment the way Whisper's are.

Other local copies exist but are not reusable: FluidAudio keeps a CoreML
`.mlmodelc` build of v2, and `parakeet-mlx` wants MLX safetensors. Same model,
three incompatible export formats.

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

**Vision, not OCR — `mlx-vlm` + Qwen2.5-VL-7B-Instruct (4-bit, ~5-6GB RAM).**
Qwen2.5-VL leads DocVQA (96.4% at 72B, near the ~98.1% human ceiling) and scores
89.5% on ChartQA. `mlx-vlm` is very actively maintained. Fallback for missed
chart/table detail is Qwen3-VL-30B-A3B-Thinking at ~18-20GB, which needs a 64GB
machine.

## Open

- **Frame dedup has no validated method for screen content.** Searching
  "screencast keyframe extraction", "slide change detection", "lecture video
  segmentation", "screen recording deduplication" turns up only generic
  luminance-shift academic work and abandoned repos. Confirmed gap.
  - Planned shape is two-stage: cheap pHash pre-filter, then a DINOv2 cosine
    gate for "is this actually new information".
  - The pHash threshold everyone cites (Hamming ≤8) is calibrated on natural
    photos, never on text-heavy UI. Worse, pHash discards *high-frequency*
    detail — precisely what separates two states of the same spreadsheet. Expect
    it to under-trigger on "same doc, different cell selected".
  - DINOv2 over CLIP because it is image-only self-supervised, so there is no
    text-conditioning to collapse every "spreadsheet-shaped" image together. One
    report puts DINOv2-ViT-B/14 at ~93% on document/screenshot duplicate
    detection, ahead of CLIP. **But no published benchmark tests discrimination
    between two different spreadsheets.** Nothing here is trustworthy until
    measured on our own data.
- **Therefore the first build task is an eval set**, not a feature: 50-100 hand-
  labelled frame pairs from real recordings ("same information" / "new
  information"). No threshold means anything without it.
- **ColPali / ColQwen2** (late-interaction retrieval over page images, no OCR)
  is the interesting phase-2 option, but no MLX port exists and nobody reports
  running `colpali-engine` on Apple Silicon MPS. Experiment, not infrastructure.

## Prior art

Nothing does "long screen-recording in → deduped, VLM-described,
timestamp-queryable index out". `HKUDS/VideoRAG` is active and closest in spirit
but shaped for general long-video QA. The video-analysis MCP servers that exist
are OCR-based, which is the approach this tool rejects.

## Layout

```
deixis/     package
scratch/    working files, not committed
```
