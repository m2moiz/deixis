# Acceleration on the OnePlus 15 — four research lanes, reconciled

Date: 2026-08-14. Target: OnePlus 15, Snapdragon 8 Elite Gen 5 (SM8850,
2×4.61 GHz + 6×3.63 GHz Oryon, Adreno 840, Hexagon HTP v81), 15 GB RAM,
Termux + Tasker. Four independent research agents (whisper, parakeet, NPU
reality, thermals), each citing sources; contradictions between lanes were
resolved in favour of the lane holding primary evidence, and are noted.

## The three findings that change dsj's plan

**1. The model matters more than the accelerator.** Measured on this exact
SoC: parakeet-tdt-0.6b-v3 through sherpa-onnx's native QNN runtime runs at
**RTF 0.056** (a 45-minute recording in ~2.5 minutes), while whisper
large-v3-turbo q8_0 on CPU with beam 3 measured **RTF ~1.8** (108 s per 60 s
of audio) and gains at most 13% from the NPU. The NPU's ASR wins live in
encoders and non-autoregressive models — CTC and transducers, exactly the
parakeet family — not in whisper's token-by-token decoder.
(sherpa-onnx PR #3720, benchmarked on a Xiaomi 17 Pro = same SM8850;
llama.cpp issue #25876, same chip, from Termux.)

**2. The Hexagon NPU is reachable from Termux, unrooted, no APK.** A
documented July 2026 run of `whisper-cli` with the ggml Hexagon backend on
SM8850 settles what was previously assumed impossible. The barrier is
unsigned-PD access on the cDSP, and Termux (untrusted_app) demonstrably gets
it on SM8850 builds. The one remaining unknown is whether sherpa-onnx's QNN
binaries — documented via `adb shell` — inherit this; the ggml precedent says
probably, and 30 minutes on the device settles it. (llama.cpp #25876.)

**3. OxygenOS downclocks non-whitelisted apps BEFORE any heat exists.**
GSMArena measured the OnePlus 15 failing to reach proper clocks from test
start in both Balanced and Performance modes, and sustaining full clocks for
a full hour once forced via Game Assistant. Whether Termux can join that
whitelist is the highest-value unknown in the whole stack — worth testing
before any code is written, because it bounds everything else.

## Verdict table

| Path | Verdict | Key evidence |
|---|---|---|
| parakeet v3, sherpa-onnx **QNN/NPU** (prebuilt SM8850 binaries) | **USABLE NOW** — C++ binary only, not pip | RTF 0.056, 8 s window, PR #3720; per-token `timestamps` + `durations` confirmed in the run log |
| parakeet v3, sherpa-onnx **ONNX int8 CPU** | **USABLE NOW** — pure Python, unmeasured on phones | RK3588 A76 calibration: RTF 0.088 @ 4 threads; Oryon should beat it, no phone number published |
| whisper turbo q8_0, whisper.cpp **CPU** | **USABLE** — the only word-timestamp path | 108 s / 60 s audio with beam 3 on SM8850 (#25876); greedy + VAD unmeasured, likely ~real-time |
| whisper, ggml **Hexagon** backend | **MARGINAL** | +13% over CPU at best (turbo, with `OPFILTER="ADD" OPPOLL=1`); large-v3 is 1.5× SLOWER than CPU; blocked on the v81 ADD-after-HMX bug (#25876) |
| whisper via **Qualcomm QNN compiled** (AI Hub) | **CAGED** | encoder 279 ms/window (~100× the ggml path) but: no turbo context binary shipped, no word timestamps, 200-token decoder cap, unreproduced outside Qualcomm's farm |
| whisper.cpp **OpenCL** on Adreno | **DEAD END today** | open assert failure on Adreno 8xx (whisper.cpp #3708) |
| whisper.cpp **Vulkan** on Adreno 8xx | **DEAD END today** | driver chaos on Adreno 830+; gibberish output reports (llama.cpp #16881) |
| sherpa-onnx on Adreno **GPU** | **DEAD END** | no OpenCL/Vulkan provider exists in sherpa-onnx; NNAPI is deprecated and falls back to CPU |
| ORT **QNN EP** on the ONNX transducer | **DEAD END** | no dynamic shapes, no Loop/If — moot; sherpa's native QNN runtime bypasses ORT entirely |
| **distil-whisper** for Urdu | **DEAD END** | English-only by construction |

## What this means for the port plan

- **Step 5 (sherpa engine) changes shape.** The QNN path is C++-only — the
  Python bindings do not expose `QnnConfig` at all — so the fast sherpa
  engine is a **subprocess engine** (like whispercpp) driving the
  `sherpa-onnx-offline` binary, not a pip import. The pip/CPU path remains
  the pure-Python fallback, with its own Termux risk (onnxruntime wheels are
  glibc; Termux is bionic — ORT issue #16514). Ship the subprocess engine
  first; it is also the one with the measured RTF.
- **Fixed windows change the chunking contract.** QNN context binaries are
  compiled per window length (3–30 s; audio beyond the window is silently
  truncated). dsj's 120 s chunks cannot be handed to it directly — the
  sherpa-QNN engine must sub-window (VAD-segment) inside `decode()` and
  offset timestamps itself. The ChunkEngine protocol survives; the engine
  gets thicker.
- **Greedy decoding only** for parakeet TDT everywhere: beam search on TDT
  is reported to hallucinate or return empty ~20% of the time
  (sherpa-onnx #3267). The QNN path is greedy-only anyway.
- **Whisper stays CPU** for Urdu, with VAD (`--vad`) and greedy (`-bs 1`)
  as the levers, and `--dtw large.v3.turbo` for word times (the turbo
  alignment-heads bug is fixed on master). Re-evaluate the Hexagon path
  when llama.cpp #25876's ADD bug is fixed — that is the tracked upstream
  blocker.
- **Model-quality landmine, independent of speed:** whisper large-v3 is
  documented to transliterate/translate English into Urdu script on
  code-switched audio (arXiv 2605.17846) — dsj's exact use case. Benchmark
  turbo vs full large-v3 on a real 5-minute code-switched sample on the Mac
  before committing the phone default.

## The 60-minute-run checklist (thermals lane, ranked)

1. **Try to whitelist Termux in Game Assistant / force performance mode**
   (`adb shell cmd game mode performance com.termux` as fallback). Detect
   the downclock by benchmarking a fixed clip whitelisted vs not.
2. `termux-wake-lock` — binary: job survives screen-off or dies.
3. **Disable the phantom-process killer** via adb (`settings put global
   settings_enable_monitor_phantom_procs false` + max_phantom_processes).
   Known Android-15 report where wake-lock alone was insufficient.
4. Battery → Unrestricted for **both** Termux and Tasker; Termux:Tasker
   timeout Never; `nohup` the real work.
5. Screen off; don't charge during runs — or use OxygenOS 16.0.2.401+
   **bypass charging**.
6. Ambient/conduction matter measurably: a 4 °F swing flipped a stress test
   from crash to pass on this phone. Face-down on metal, out of the case.
7. Threads: start 6 pinned to the perf cores (`taskset`, verify indices via
   cpufreq); benchmark 6 vs 8 vs 4 at steady state, not cold.
8. ggml runtimes: `--poll 0` — spin-waiting converts thermal budget to heat.
9. Run continuous; duty-cycling is folklore (sprinting literature says it
   buys latency, not sustained throughput).

Benchmark discipline for all of it: fixed 5-minute clip, 10 minutes of
warm-up first — this SoC's cold-start numbers lie.

## Unverified, carried forward honestly

- sherpa-onnx QNN from Termux specifically (adb-shell-documented only) — the
  gating smoke test for the fast path.
- RTF of the 30 s QNN window (only 8 s was benchmarked); QNN-vs-CPU WER for
  the quantised binaries (nobody measured accuracy).
- Whether the OnePlus 15's SM8850 variant (-AC/-AD/…) matters to the
  context binaries.
- Whether Termux can join OnePlus's performance whitelist at all.
- Whether Adreno **840** behaves like the 830 every GPU data point comes
  from (moot while both GPU paths are dead ends).
- whisper.cpp `--dtw` alignment *quality* on turbo (the flag works; nobody
  published alignment accuracy).

Sources are inline above; the four full lane reports (with complete URL
lists) live in the session transcript of 2026-08-14.
