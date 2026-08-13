# Android port — portable core, engines at the leaves

Target: the same package runs on macOS (as today, nothing lost) and on a
OnePlus 15 under Termux (aarch64 Linux, 15 GB RAM, no Metal, no CoreML).
Produced from two independent design passes — a system-design lane and a
code-grounded blueprint lane that read every import site — then reconciled.
Where they disagreed, the resolution and the reason are recorded inline.

Everything below is an **extraction**: modules move imports, not behavior.
The only new code is one vendored file, the engine registry, and (later) two
new engine modules.

## Why jaano cannot run on the phone today

`parakeet-mlx` is a hard dependency (`pyproject.toml`), and MLX ships no
aarch64-Linux wheels, so `uv tool install jaano` fails at resolve time. Worse,
the coupling is not confined to the parakeet engine: `checkpoint.py:26` uses
parakeet's `AlignedToken` as its on-disk token type, and `chunking.py:27-47`
imports parakeet's merge/sentence helpers and log-mel frontend. The class that
describes a word with a timestamp lives inside an Apple-only package — that,
not the ASR engine, is the actual blocker.

The full verified import surface: `checkpoint.py:26`, `chunking.py:27-47`,
`suno.py:49-50` (type-only) and `suno.py:304-309` (function-local), plus
`checkpoint.py:87`'s `version("parakeet-mlx")` metadata lookup. `media.py` and
`asr.py` mention parakeet only in prose. No other module touches it.

## Module layout — flat, 12 modules become 16

```
jaano/
  asr.py          contract + registry + EngineUnavailable      (grows)
  alignment.py    NEW: vendored token type + 4 merge helpers   (core)
  chunking.py     resumable chunk loop, engine-free            (loses parakeet imports)
  checkpoint.py   resume state                                 (import swap only)
  merge.py media.py atomic.py cli.py dekho.py                  (untouched)
  suno.py         orchestration, zero backend imports          (loses parakeet imports)

  parakeet.py     NEW: extracted from suno.py    chunk engine, Apple
  whisper.py      unchanged                      file engine,  Apple
  sherpa.py       LATER                          chunk engine, portable
  whispercpp.py   LATER                          file engine,  portable
```

**Flat, not `jaano/engines/`.** The blueprint lane proposed moving the chunk
loop into `jaano/engines/parakeet.py`; the design lane kept the layout flat
and made `chunking.py` engine-generic instead. Flat wins on both counts: a
subpackage renames `jaano.whisper` and churns imports and monkeypatch targets
for zero behavior change, and the deeper point is that the chunk loop is NOT
parakeet-specific once it is handed tokens instead of a model — it is resume
machinery, and sherpa needs it too. The boundary is enforced by a test, not a
directory (see CI).

## `jaano/alignment.py` — the vendored token vocabulary

parakeet-mlx 0.5.2's `alignment.py` is 287 lines of pure Python + numpy.
Nothing in it touches MLX. License: Apache-2.0 (dist-info
`License-Expression`; the source file itself carries no header). Vendor it
verbatim with a one-line attribution comment, keep every name.

Moves unchanged: `AlignedToken` (`{id, text, start, duration, confidence,
end}`, `__post_init__` sets `end = start + duration`), `AlignedSentence`,
`AlignedResult`, `SentenceConfig`, `tokens_to_sentences`,
`sentences_to_result`, `merge_longest_contiguous`,
`merge_longest_common_subsequence`. All four functions are pure list/index
arithmetic over token fields — verified by reading the installed source; zero
logic changes needed.

Three rules for the copy:

1. **Annotate the two merge returns** as `list[AlignedToken]` — upstream
   builds a bare `list`, which is the entire reason `chunking.py:131-147`
   carries four `cast`s. The casts and the `_Generates` scaffolding delete
   themselves.
2. **Keep `merge_longest_contiguous` raising bare `RuntimeError`**
   (upstream `alignment.py:194`). `chunking.py:141` catches exactly that type
   to fall back to the LCS merge — the exception type is load-bearing control
   flow. Do not "improve" it during the port.
3. **Keep `id: int` mandatory.** It is the merge key.

`chunking.py`'s docstring currently argues "the hard part stays upstream."
That premise dies with the vendoring; rewrite it. `test_chunking.py`'s
equivalence test against upstream becomes the proof the copy is faithful —
keep it, `skipif` parakeet-mlx absent. This is the one legitimate
self-disabling skip in the suite: it compares us *to* upstream, so upstream's
absence genuinely leaves nothing to compare.

## Engine contract — two protocols, and only ever two

```python
# jaano/asr.py
class ChunkEngine(Protocol):
    """An engine jaano drives chunk by chunk, so a run can resume.

    parakeet and sherpa both decode a chunk from fresh decoder state --
    that statelessness is what makes resume exact. An engine that carries
    state across windows cannot implement this and must be a FileEngine.
    """
    sample_rate: int
    min_chunk_samples: int
    def decode(self, samples: Any) -> list[AlignedToken]: ...

class FileEngine(Protocol):
    """An engine that owns its whole loop and returns a finished transcript.

    whisper threads each window's text into the next as a prompt; the Roman
    Urdu bias rides on that continuity. Cutting the file up to buy a resume
    would break the feature the engine was chosen for.
    """
    sample_rate: int
    def transcribe_file(self, audio, *, model_id, language, prompt) -> Transcription: ...
```

Engine modules are modules of plain functions, mirroring `diarize.py`:
`DEFAULT_MODEL`, `SAMPLE_RATE`, `INSTALL_HINT`, `available() -> str | None`
(None = usable; a string = the reason not), `fingerprint_fields()`, and
`load()` (chunk) or `transcribe_file()` (file). `load()` returns a small
module-private dataclass closing over the model handle and the feature step —
that is where `get_logmel` goes for parakeet.

### `chunking.py` after the extraction

`transcribe_chunked(engine: ChunkEngine, audio_data, *, chunk_s, overlap_s,
start_tokens, skip_before, on_chunk, sentence)`. Three substitutions in the
body, nothing else: `model.preprocessor_config.sample_rate` →
`engine.sample_rate`; the hop-length floor → `engine.min_chunk_samples`; the
logmel+generate+walk → `engine.decode(chunk)` then the same `+= offset` walk.
The offset walk stays byte-identical, including re-assigning `token.end`
after shifting `token.start` — `__post_init__` does not re-run on assignment.

`DecodingConfig` leaves the signature. Verified: jaano only ever reads
`cfg.sentence`, which defaults to `SentenceConfig()`; the decoding half is
engine business and moves into `parakeet.py`'s `load()`.

### `suno.py` after the extraction

`transcribe()` swaps its function-local parakeet imports for
`spec = get_engine(engine)` and branches on `spec.kind` (`"chunk"`/`"file"`).
This preserves the invariant documented at `suno.py:378` — exactly one thing
decides which branch runs. Temp dir, conversion probe, clocks, checkpoint
read/write, `_label_speakers`, atomic writes: untouched.

`CHUNK_S = 120.0` / `OVERLAP_S = 15.0` stay core constants. They were tuned
against a ~9.5 GB Metal budget and may be wrong for the phone, but changing
them changes the Fingerprint and orphans every checkpoint — "tune chunk
geometry for Android" is a separate, measured change.

## Registry — static tuple, availability at runtime

```python
ENGINES = ("parakeet", "whisper", "sherpa", "whispercpp")
_REGISTRY = {name: EngineSpec(name, module, kind, platform), ...}
```

`ENGINES` is the same four strings on every platform, so `--help` and
`--engine` validation are platform-independent; a script written on the Mac
fails on the phone with a specific remedy, not "unknown engine".
`get_engine(name)` imports the module, calls `available()`, and raises
`EngineUnavailable` with the reason. **`sys.platform` appears nowhere in
control flow** — "does the import resolve" is the question that actually
matters, and it answers correctly for free on a Rosetta Mac, an x86 CI
runner, and the phone.

Rejected: entry points (solves third-party registration nobody asked for),
platform-conditional `ENGINES` (makes `--help` platform-dependent), and
`if sys.platform` branches (the spaghetti this section exists to prevent).

## Dependencies

```toml
dependencies = ["numpy>=2.0", "pydantic>=2.13.4", "typer>=0.20"]

[project.optional-dependencies]
# Apple
parakeet = ["parakeet-mlx>=0.5.2"]
whisper  = ["mlx-whisper>=0.4.3"]
diarize  = ["senko>=0.1.0,<0.2"]
# Portable
sherpa   = ["sherpa-onnx>=1.10"]         # pin verified at implementation time
# Convenience (PEP 685 self-referencing)
apple    = ["jaano[parakeet,diarize]"]
android  = ["jaano[sherpa]"]
```

`whispercpp` gets **no extra** — it shells out to a binary. Discovery is
`shutil.which("whisper-cli")` plus a `JAANO_WHISPER_CPP` override; its
install hint names the package manager, never a uv command.

**No environment markers on parakeet-mlx.** `sys_platform == 'darwin'` would
make `jaano[parakeet]` on Android succeed while installing nothing — the
silent-degradation class that already cost this project a 45-minute recording.
The wrong extra must fail loudly at resolve time.

`requires-python = ">=3.12,<3.14"` stays (the cap is coremltools', reachable
only via `diarize`). Verify Termux's Python is in range on the device; if it
has moved past 3.13, widen the cap explicitly and move the constraint onto
the extra — never quietly.

Lazy-import pattern: `diarize.py:126-153` is the reference and every engine
copies it. `available()` never imports the backend (`find_spec` / `which`);
the real import is function-local; `ModuleNotFoundError` with
`exc.name != <package>` gets the "installed but will not load" message.
`_will_not_load` moves to `asr.py` and is shared — `whisper.py` currently
lacks this distinction (bare `ImportError` → always "not installed"), the bug
`diarize.py` already fixed; the move fixes it there too.

## Checkpoint compatibility

### On-disk bytes do not change

`checkpoint._to_json` omits `end` because `AlignedToken.__post_init__`
recomputes it. The vendored class copies that behavior verbatim; the document
stays `{id, text, start, duration, confidence}`; `SCHEMA` stays 1.

**Golden fixture, written first.** Commit a literal checkpoint produced by
today's code (a static file, NOT regenerated by the new code — that would
only prove the new code agrees with itself) and a test that `read_checkpoint`
returns identical tokens. Land the test before the vendoring commit, so
byte-compatibility is observed, not predicted.

### Engine identity joins the Fingerprint without breaking it

Two engines can now produce tokens for the same media at the same geometry,
and cross-engine token lists are not mergeable. But adding a tenth field to
`Fingerprint` invalidates every checkpoint on disk.

Resolution (design lane's, over the blueprint lane's rename-and-discard):
drop `parakeet_version` from the dataclass, add `engine_fields: dict[str,
str]`, and serialize with a **flat merge** —

```python
def to_dict(self):
    return {k: v for k, v in asdict(self).items() if k != "engine_fields"} \
        | self.engine_fields
```

parakeet contributes `{"parakeet_version": version("parakeet-mlx")}` —
**byte-identical to today's document**, so existing checkpoints keep
resuming. sherpa contributes `{"engine": "sherpa-onnx", "sherpa_version":
...}`; the differing key sets mean a sherpa checkpoint can never equal a
parakeet one, which is stronger than a value comparison and costs no schema
bump. Fallback if the golden test cannot be made to pass: bump `SCHEMA` to 2
and say so in the changelog — fail-safe, but announced.

This also removes `checkpoint.py:87`'s `version("parakeet-mlx")` call, which
would crash a base install where the extra is absent.

`test_checkpoint.py:91-112` enumerates fingerprint field names exactly and
fails on drift, so the restructure is self-checking: forgetting to update the
test fails loudly.

### sherpa token ids must be deterministic across processes

`merge_longest_contiguous` compares `token.id`. sherpa returns token strings,
not vocabulary ids, and stored ids are compared against freshly-decoded ones
on resume — so an insertion-order interning table breaks resume (different
process, different integers for the same strings). Fix in `sherpa.py`, not in
core: derive the id from the token string (`zlib.crc32(text.encode())`, or
sherpa's own symbol-table index if exposed). Collisions are harmless — the
merge also gates on timestamp proximity; non-determinism is not. Test:
decode the same chunk in two subprocesses, assert identical ids.

Also from sherpa's result shape: `durations` is optional. When absent, derive
`duration = next.timestamp - this.timestamp` (frame stride for the last
token). **Never default to 0.0** — the merge reads `a[-1].end`, and a
zero-duration tail collapses the overlap window and silently degrades every
chunk seam.

## Failure modes

`asr.py` gains `EngineUnavailable(RuntimeError)`, sibling of
`DiarizationUnavailable` — with the asymmetry in its docstring:
**diarization degrades, ASR does not.** `_label_speakers` can fall back to an
unlabelled transcript because ASR was already paid for; an unavailable engine
means no transcript exists, so the error propagates and the CLI exits
non-zero. **No automatic engine fallback** — silently transcribing an hour of
Urdu with a different model, at different quality, under a different `model`
key in the payload, is worse than failing in one second.

The phone, asked for parakeet, gets all three properties `diarize.py`
established: name the machine (not just the missing package), name the remedy
that exists (`--engine sherpa` runs the same parakeet-tdt-0.6b-v3 through
ONNX), and distinguish absent from broken (`_will_not_load`). A missing
whisper.cpp *binary* and a missing GGML *model file* are different failures
and get different messages.

`suno.py:292`'s prompt/language guard generalizes from "parakeet takes
neither" to a per-engine frozenset — `--prompt`/`--language` are meaningful
for the two whisper engines only. A frozenset per engine; not a capability
system.

## CI

| Job | Runner | Proves |
|---|---|---|
| `install-gate` | macos-14 | the documented Apple install, updated to the extras form |
| `check` | macos-14 | full suite with all extras synced |
| `install-gate-linux` | `ubuntu-24.04-arm` (verify availability first) | the documented Android install line resolves, imports, and decodes on CPU |
| **AST boundary test** | in `check` | core modules import no backend |

The boundary test AST-parses every non-engine `jaano/*.py` and asserts no
import of `{parakeet_mlx, mlx_whisper, mlx, senko, sherpa_onnx, coremltools}`.
AST, not grep — prose mentions must not fail it. This is what makes "core is
portable" a fact rather than a claim, on a platform where nothing else would
observe it.

Two honesty constraints: if the arm runner is unavailable, fall back to x86
`ubuntu-latest` and **say in the workflow comment that the target arch is
unexercised** — an aarch64-labelled job silently running x86 is a skip that
looks like a pass. And CI cannot run Termux: the phone install remains a
manual smoke, recorded in the README with a date and version, labelled as
not-CI-observable.

`test_install_gate.py`'s `DOCUMENTED` regex pins the README's literal
command; it changes in the same commit as the README (its own drift test
enforces this). New gate assertion: `jaano --help` exits 0 without
`[parakeet]` installed, and a real `jaano suno` without the extra fails with
the message naming the extra. `pyproject`'s mutmut `only_mutate` gains
`jaano/alignment.py` — pure, fast-testable, exactly the shape that list
exists for; today the four helpers have zero fast-lane direct tests.

## What NOT to build

1. No plugin framework — four engines, one dict, all in-tree.
2. No ABC tower — two `Protocol`s, never inherited from.
3. No config system — CLI flags only; `JAANO_WHISPER_CPP` is the lone env
   override because binary location is genuinely environmental.
4. No capability negotiation — `spec.kind` and one frozenset are the whole of it.
5. No auto-selection of "best available engine" — an unreproducible run is
   worse than an error.
6. No engine-specific payload fields — `{audio, model, text, sentences}` is
   pinned by tests for downstream readers; whisper.cpp's `p`/`t_dtw` stop at
   the engine boundary. `dekho.py` must not learn which engine made a file.
7. No chunked-resume for whisper.cpp — it is a `FileEngine` like mlx-whisper;
   resume there is a future measurement, not a design input.
8. No rewrite of `media.py`, `merge.py`, `dekho.py`, `atomic.py` — already
   portable; the AST test proves it.

## Sequencing — each step independently green

| # | Commit | Gate |
|---|---|---|
| 1 | golden checkpoint fixture + resume test, against current code | new test green before anything moves |
| 2 | vendor `alignment.py`; `checkpoint.py`+`chunking.py` import it; delete casts; fast unit tests for the four helpers | full suite + golden + upstream-equivalence |
| 3 | `Fingerprint.to_dict()` + `fingerprint_fields()`; extract `parakeet.py`; registry; AST boundary test | golden still green = byte-compat observed |
| 4 | extras split; README both install lines; install-gate command | install-gate green on the new documented command |
| 5 | `sherpa.py` + `install-gate-linux` | Linux job green; arch caveat if x86 |
| 6 | `whispercpp.py` | mac-side stub-binary test; real run on the phone |
| 7 | Termux manual smoke, dated in README | not CI-observable — labelled as such |

Steps 1–3 ship no new capability and must not change any output byte. Step 4
is the only breaking install change and gets its own commit so a bisect lands
on it cleanly. Step 3 is the highest-risk commit and needs `just verify`
(slow equivalence tests), not just `just check`.

## Open questions (decide before step 4; steps 1–3 proceed regardless)

1. **Product decision:** what does bare `uv tool install jaano` mean once
   parakeet is an extra? Either the README's headline line becomes
   `jaano[apple]` (recommended: macOS is the primary platform and a default
   install with no working engine is a trap), or bare `jaano` stays and is
   documented as the minimal/Termux form. This changes the install-gate regex
   and the "Requirements: macOS on Apple Silicon" README section, which
   becomes false once the base package is portable.
2. **Unverified, each gating its own step:** sherpa-onnx wheel availability
   and pin for aarch64/Termux (step 5); whether sherpa exposes stable token
   ids (assumed not; worked around); `ubuntu-24.04-arm` runner availability
   (step 5); Termux Python vs `<3.14` (step 7).
