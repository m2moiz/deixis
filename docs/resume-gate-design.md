# A resume gate that cannot silently pass

Design for the end-to-end verification of `--resume` in `jaano/suno.py`
(HEAD `f081231`, branch `feat/transcription`).

Prototype: `scratch/resume_gate.py` — written, executed, output pasted in
[§6](#6-what-it-actually-printed).

The whole document is organised around one property: **every leg carries a
positive observation that the thing under test happened.** A gate that only
checks "the run finished and the output is correct" is satisfied by a resume
feature that does nothing at all.

---

## 1. Failure modes a naive harness has

The harness under review:

```bash
uv run python -m jaano.suno scratch/meeting.wav -o scratch/gate_resumed.json --no-resume >log 2>&1 &
PID=$!
sleep 45
kill -9 $PID
```

Eight defects. Two were already known; the rest are new. **Every one of them
makes the gate pass while testing nothing** — none of them make it fail.

### F1 — `$!` is the `uv` wrapper, not the worker (confirmed empirically)

```
$ uv run python -c "import os; print('py pid', os.getpid(), 'ppid', os.getppid(), 'pgid', os.getpgid(0))"
py pid 46858 ppid 46855 pgid 46852
$ echo $$
46852
```

`uv 0.11.26` **forks** a child interpreter and waits on it; it does not `exec`
into it. The shell's `$!` is pid 46855 (uv). `kill -9 46855` reaps the wrapper
and leaves 46858 holding the GPU. That is the observed 17-minute orphan.

Two compounding effects worth naming separately: `wait $PID` then returns
*immediately* (the wrapper really is dead), so the script has no signal that
anything is wrong; and the orphan keeps writing to `gate_resumed.json` and its
checkpoint *while the next leg is reading them*, which makes every downstream
assertion a race against a process the gate does not know exists.

### F2 — a resume that silently restarts passes every output check

`skip_before` is threaded from `read_checkpoint` → `transcribe_chunked`. If
`read_checkpoint` returns `None` for any reason (stale fingerprint, missing
file, wrong shape — `checkpoint.py:127-148` returns `None` for all of them, by
design, *silently*), the run starts from zero and produces a **byte-identical,
completely correct** transcript. `shasum` matches. `text ==` matches.
`sentences ==` matches. The feature is inert and the gate is green.

This is the single most important thing the gate must discriminate, and no
comparison of outputs can ever do it.

### F3 — `sleep 45` is a race, not a synchronisation

45 s must cover: interpreter start, `from_pretrained` (~2.4 GB of weights, and
a **cold HF cache is a multi-minute download**), ffmpeg extraction on a `.mov`,
`load_audio`, and then at least one chunk decode. On the measured machine chunk 1
of `clip360.wav` landed at elapsed 0:07 — but chunk 2 took 63 s and chunk 3 took
91 s (§5). Slide any of those and 45 s fires before a checkpoint exists.

And the script's own handling of that case is `... || echo "NO CHECKPOINT
WRITTEN"` — which returns 0, so `set -e` does not trip. The gate then "resumes"
a checkpoint that is not there, i.e. runs a **full fresh transcription**, which
compares equal to the baseline. Green. Nothing tested.

### F4 — `set -e` does not see through the pipes

`uv run ... 2>&1 | tail -1` exits with `tail`'s status. Without `set -o
pipefail` a crashed baseline is a passing step. Same for step 3.

### F5 — nothing establishes that the kill worked

`kill -9 $PID` is a *send*, not an observation. Nothing checks the exit status
of the thing killed, nothing checks that the process is gone, nothing checks
that transcription stopped. Under rule 09.3 this is the whole failure: "signal
sent" is not "process dead", and "process dead" is not "work stopped".

### F6 — no assertion that the checkpoint is *mid-run*

The script checks `ls` on the checkpoint. `next_start` can legitimately equal
`total_samples` — that is what the **last** chunk writes (`chunking.py:117`).
A checkpoint at `next_start == total` makes the resumed run skip every chunk,
merge nothing, and emit the banked tokens: correct output, zero resumed work.
Symmetrically `next_start == 0` cannot happen but an empty/torn file can, and
`ls` is satisfied by a zero-byte file.

### F7 — the `.mov` path is never exercised, and it is the one that can regress silently

`fingerprint()` keys on `media` — the source the user handed us — precisely
because a `.mov` extracts to a **fresh temp wav with a new path and mtime on
every run** (`checkpoint.py:52-62`, `transcribe.py:188-191`). If that keying
ever regresses to the decoded audio, resume never matches for the *normal*
input, and it fails **silently**: it just starts over, and every `.wav` test in
the suite stays green, because for a conforming wav `media` and `audio` are the
same file. A wav-only gate is structurally blind to this.

Second-order: on the `.mov` path the worker spawns **ffmpeg as a child**.
Killing only the Python process leaves ffmpeg running.

### F8 — no timeout anywhere

Every wait in the script is either unbounded (`wait`, and the foreground runs)
or a fixed sleep. The stated requirement is "fail loudly rather than hang", and
this is the shape that produced a 17-minute hang.

### F9 (minor) — the comparison runs under a different interpreter

Step 4 uses bare `python3`, not the project venv. It happens to work because it
only touches `json`, but the gate's assertions should run under the same
interpreter as the code under test.

---

## 2. How to kill a transcription and prove it died (macOS)

### Recommendation

**Launch `sys.executable` directly (no `uv run` wrapper) with
`start_new_session=True`, kill with `os.killpg(pgid, SIGKILL)`, and prove death
with four independent observations.**

```python
proc = subprocess.Popen(
    [sys.executable, "-m", "jaano.suno", str(media), "-o", str(out),
     "--status", str(status_marked)],
    cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
    start_new_session=True,
)
pgid = os.getpgid(proc.pid)
...
os.killpg(pgid, signal.SIGKILL)
```

Two independent problems, two independent fixes:

- **No wrapper** → `proc.pid` *is* the worker. This is strictly better than
  making the wrapper transparent, because there is nothing to be transparent
  about. When the harness itself runs under `uv` (as a pytest test does),
  `sys.executable` is already `.venv/bin/python`; `uv run` buys nothing and
  costs the fork.
- **`start_new_session=True`** → the child leads a fresh process group *and*
  session, so one `killpg` reaches ffmpeg and any other descendant, and that
  group is disjoint from the harness's own group so `killpg` cannot signal the
  harness. (In a **non-interactive** shell there is no job control, so a
  background job stays in the *script's* process group — `kill -- -$PGID` there
  kills the script. `setsid` is what a shell version would need, and macOS ships
  no `setsid(1)`; `python -c 'os.setsid()'` or `script`-style wrappers are the
  workarounds. This is a good reason not to write it in shell.)

### Proving it died — four observations, in order of strength

| # | Check | What it rules out |
|---|---|---|
| 1 | `proc.wait(timeout=30)`; assert `rc == -signal.SIGKILL` | The worker is reaped, and reaped *by our signal* — not by finishing normally (`rc == 0`) and not still running (`TimeoutExpired`). |
| 2 | poll `os.killpg(pgid, 0)` until `ProcessLookupError` | Any *descendant* still alive. **This is the check that catches F1 directly**: with `uv run` under `start_new_session`, the orphaned interpreter stays in the group and signal-0 succeeds instead of raising. |
| 3 | `pgrep -f <uuid marker>` returns non-zero | An escapee that changed its own process group. The marker is a uuid injected into the `--status` path, so it is on the child's argv and nowhere else. |
| 4 | checkpoint bytes unchanged across a 5 s window | Progress, as opposed to liveness. |

Check 3 uses `pgrep` as an **auditor, never as the killer** — the distinction
matters. Pattern-matching is sound for "prove this set is empty" and unsound for
"identify the one process to signal".

Check 4 is deliberately weak and labelled as such: the checkpoint only advances
once per chunk (60–90 s on this machine, §5), so 5 s of stillness proves little
on its own. It is kept because it is the only check that speaks to *work
stopping* rather than *pids vanishing*, and it costs 5 s. **Do not promote it to
the primary evidence** — the honest primary is #1 + #2.

### The alternatives, and why each is worse

| Approach | Verdict |
|---|---|
| `bash -c 'exec uv run …'` | **Does not help.** `exec` replaces the shell with `uv`, and `uv` still forks. You'd get uv's pid. The wrapper is the problem, not the shell. |
| `bash -c 'exec .venv/bin/python -m …'` | Correct for the pid, but strictly worse than Popen-direct: adds a shell to quote through, and still misses ffmpeg grandchildren. Use it only if the harness must be shell. |
| `pkill -f <marker>` as the killer | Racy and unverifiable. `pkill` can match the shell that launched it, an editor with the path open, or a `grep` in the same pipeline; there is no way to know *which* pids it hit, no exit status to reap, and no way to distinguish "killed the worker" from "killed nothing". Fine as an auditor (#3). |
| `multiprocessing` in-process | On macOS the start method is **spawn**, so you re-import and reload the 2.4 GB model anyway — no saving over `Popen`. And it *loses* the coverage that motivates a subprocess gate at all: CLI wiring, `--status`, the piped-stderr bar branch, and a real SIGKILL that skips every `finally`, `atexit` and `BaseException` handler. The exception-based interrupt in `tests/test_resume_cli.py` already covers the in-process case, and it unwinds cleanly — which is exactly what a hard kill does not. |

### One real finding this exposes

`atomic_write_text` cleans up its temp file in a `except BaseException` handler
(`atomic.py:41-46`). **SIGKILL runs no handler.** A kill landing inside the
checkpoint write window strands a `.resumed.json.ckpt.<pid>.tmp` beside the
output. It is harmless (nothing reads it, and `read_checkpoint` would reject
it), but it accumulates. The gate *reports* it rather than asserting on it,
because it is timing-dependent — asserting would make the gate flaky, and
asserting its *absence* would be asserting a race. Worth a bead.

---

## 3. Proving the resumed run actually used the checkpoint

This is F2, and it is the assertion the reviewed harness lacked entirely.
Four candidate observables, ranked:

| Observable | Strength | Use |
|---|---|---|
| **`--status` JSON `resumed_from_s`** | **Strongest, machine-readable, exact.** `Progress.resumed_from_s` is `skip_before / rate` (`transcribe.py:207`), it is in `asdict(p)` in every heartbeat (`transcribe.py:149`), and it *survives to the final `"done"` payload* (`transcribe.py:267`). Assert `abs(resumed_from_s - next_start/RATE) < 1e-6` against the value read out of the checkpoint before the kill. | **Primary.** |
| **Chunk-line count in piped stderr** | **Strong and fully independent** of both the checkpoint and the status file. stderr is a pipe, so `main()` prints one bar line per chunk rather than rewriting one line (`transcribe.py:296-300`). A run that consumed the checkpoint emits `len(starts) - skipped` lines; one that restarted emits `len(starts)`. Different code path, different file, same conclusion. | **Primary, corroborating.** |
| `"resuming from" on stderr` | Real but partial: it proves `read_checkpoint` returned non-`None` (`transcribe.py:199-203`), not that the loop skipped anything. Cheap, so keep it. | Secondary. |
| Elapsed wall time | **Reject.** It is exactly the flaky assertion the measured 7 s / 63 s / 91 s per-chunk spread (§5) would make useless, and it is the assumption class that produced F3. | Never. |

The progress bar's *starting position* is a fourth possibility and is
redundant with the chunk-line count, which is easier to parse exactly.

**The negative control is what makes these trustworthy.** `--self-test` reruns
leg 3 with `--no-resume` and requires the gate to **fail**. A harness with no
mutation test is a harness whose green means nothing; this is the direct answer
to "a harness that can silently not test the thing it claims to test." It costs
one extra run and should be part of the routine invocation.

---

## 4. Input length: the concrete answer

Geometry, from `transcribe.py:32-33` and `chunking.py:73-78`:

```
chunk_samples   = int(120.0 × 16000) = 1_920_000
overlap_samples = int( 15.0 × 16000) =   240_000
stride          = 1_680_000 samples  = 105.0 s exactly
starts          = range(0, total_samples, 1_680_000)
n_chunks        = ceil(total_samples / 1_680_000)
```

Resume needs a kill point that is **neither the first boundary nor the last** —
otherwise `next_start` is either 0 (nothing banked) or `total` (nothing left),
and leg 3 is a no-op that still passes (F6). That requires **3 chunks**:

> **Hard minimum: 211 s of audio** (`total_samples > 2 × 1_680_000`).
> **Recommended routine input: 225 s** — 3 chunks, `starts = [0, 1_680_000, 3_360_000]`, verified below.
> **`scratch/clip45.wav` is unusable**: 45 s is one chunk, `next_start == total` on the only `on_chunk`, and there is nothing to resume.

`tests/conftest.py` uses 360 s (4 chunks, `starts = [0, 105, 210, 315]`). That
is the better choice when you want a kill point with a *whole chunk on either
side*, and the safer choice if the kill trigger is ever loosened. 225 s is the
right routine default **because of the performance regression**: cost per chunk
is not flat (§5), so dropping the 4th chunk saves ~40 % of every run, and the
observable-triggered kill (rather than a sleep) makes the extra margin
unnecessary.

Scaling to the real 74-minute file is a flag change (`--media`), not a design
change: the kill trigger is `next_start >= starts[k]`, so it fires at the same
chunk index regardless of length.

---

## 5. Measured cost (this is the budget the design has to live inside)

Real run, `scratch/clip360.wav`, warm model cache:

```
$ time uv run python -m jaano.suno scratch/clip360.wav -o /tmp/probe.json --no-resume
   running [########----------------]  33%  2:00/6:00 audio  elapsed 0:07  eta 0:14  16.0x
   running [###############---------]  62%  3:45/6:00 audio  elapsed 1:10  eta 0:42  3.2x
   running [######################--]  92%  5:30/6:00 audio  elapsed 2:41  eta 0:14  2.0x
   running [########################] 100%  6:00/6:00 audio  elapsed 3:09  eta 0:00  1.9x
done: 5:59 audio in 3:12 -> /tmp/probe.json
uv run …  3.95s user  32.29s system  18% cpu  3:14.46 total
```

Per-chunk wall time: **7 s, 63 s, 91 s, 28 s**. Three inferences, flagged as
such:

- **Inferred:** the near-flat 18 % CPU with 32 s system time and 4 s user time
  says the process is not CPU-bound — consistent with GPU/Metal work plus
  memory pressure, not with a Python hot loop.
- **Inferred:** chunk 1's 7 s is probably an artefact of **MLX lazy
  evaluation** — `model.generate` returns unevaluated arrays and the cost is
  realised when the tokens are first read, which for chunk 1 happens inside
  chunk 2's merge. Per-chunk timings from this bar are therefore not a sound
  basis for the perf investigation. Normalised, chunks 2–4 sit at ~0.6–0.8 s of
  wall clock per second of audio, i.e. **sub-realtime**.
- **Consequence for the harness:** any assertion involving elapsed time is
  unsound here. That is why §3 rejects wall-clock as an observable and why §7
  puts a large explicit ceiling on every wait rather than a tight one.

Measured again on `clip225.wav` during the gate runs: an identical 3-chunk
baseline took **74 s** on one invocation and **19 s** on another, minutes apart,
same input, same machine (§6). Whatever the regression is, its variance is
larger than its mean. This is the strongest single argument against any
timing-based assertion or `sleep`-based synchronisation anywhere in the gate.

Budget for the full gate on 225 s: baseline + resumed + 3 stale-rejection runs
≈ 5 full transcriptions ≈ **4–8 minutes observed**. On 360 s, roughly triple —
the 4-chunk run measured 3:12 against 225 s's 74 s, i.e. the 4th chunk alone
cost more than the first three.

---

## 6. What it actually printed

`scratch/resume_gate.py`, executed on `scratch/clip225.wav`:

```
$ uv run python scratch/resume_gate.py --media scratch/clip225.wav
media   : clip225.wav (225s, 3 chunks, starts=[0, 1680000, 3360000])
[1] baseline (--no-resume)
    74s, 3 chunk lines, sha b33fa4ec4893
[2] interrupted (SIGKILL the process group after chunk 1)
    pid 60135 pgid 60135 marker 369613c2
    killed with next_start=1680000 (105.0s), 575 tokens banked
[3] resume
    38s, 2 chunk lines, resumed_from_s=105.0
[4] equivalence
    byte-identical: b33fa4ec4893
[5] stale-checkpoint rejection
    model_id: refused, full transcript, byte-identical to baseline
    chunk_s: refused, full transcript, byte-identical to baseline
    media_mtime_ns: refused, full transcript, byte-identical to baseline

PASS
EXIT=0
```

The three numbers that carry the verdict: **`2 chunk lines`** where the baseline
had 3, **`resumed_from_s=105.0`** matching the banked `next_start=1680000`
exactly (1 680 000 / 16 000 = 105.0), and **`b33fa4ec4893` == `b33fa4ec4893`**.
The first two are what the reviewed harness had no equivalent of; the third is
all it had.

No stranded `.tmp` note fired, i.e. the SIGKILL did not land inside a
`write_checkpoint` window on this run. That is luck, not a property — see §2.

### The negative control

```
$ uv run python scratch/resume_gate.py --media scratch/clip225.wav --work scratch/gate_selftest --self-test
media   : clip225.wav (225s, 3 chunks, starts=[0, 1680000, 3360000])
[1] baseline (--no-resume)
    19s, 3 chunk lines, sha b33fa4ec4893
[2] interrupted (SIGKILL the process group after chunk 1)
    pid 62769 pgid 62769 marker 243594c5
    killed with next_start=1680000 (105.0s), 575 tokens banked
[3] resume -- SELF TEST: --no-resume, the gate MUST fail below
    46s, 3 chunk lines, resumed_from_s=0.0

FAIL: the resumed run decoded 3 chunks; a run that consumed the checkpoint
decodes 2 (3 - 1 banked). It started over.

SELF TEST PASSED: the gate detects a run that ignored its checkpoint.
EXIT=0
```

This is the run the reviewed harness would have called **green**: leg 3 produced
`sha b33fa4ec4893`, byte-identical to the baseline, with the resume feature
explicitly disabled. F2, demonstrated rather than argued.

Note also `19s` here against `74s` for the *identical* baseline command on the
*identical* input minutes earlier — a **4× swing**, same file, same chunk count,
same machine. Anything the gate asserted about wall clock would be flaky on
that alone.

---

## 7. The implementation

Full source at `scratch/resume_gate.py`. The load-bearing parts:

### Every wait is a bounded poll on an observable

```python
def wait_until(pred, timeout_s, what, worker=None):
    deadline = time.monotonic() + timeout_s
    while True:
        if worker is not None and worker.proc.poll() is not None:
            raise GateFailure(
                f"worker exited (rc={worker.proc.returncode}) before {what}\n"
                f"--- tail of {worker.log} ---\n{worker.stderr_text()[-2000:]}")
        got = pred()
        if got:
            return got
        if time.monotonic() > deadline:
            raise GateFailure(f"timed out after {timeout_s:.0f}s waiting for {what}" + …)
        time.sleep(POLL_S)
```

The `worker.proc.poll()` branch is what turns F3's silent-restart into a loud
failure: if the run dies or finishes before the condition, we say so with the
log attached instead of proceeding.

### The kill trigger is a checkpoint boundary, not a clock

```python
def banked():
    c = read_ckpt(ckpt)
    return c if c and c["next_start"] >= starts[kill_after] else None

banked_ckpt = wait_until(banked, FIRST_CHUNK_TIMEOUT_S,
                         f"checkpoint to reach next_start >= {starts[kill_after]}", worker=w)
kill_and_prove_dead(w, ckpt)
```

This makes "the checkpoint is mid-run" true **by construction** rather than by
assertion, and it is machine-speed-independent: cold cache, warm cache,
75-minute file, 225-second file, same code.

### The assertions, and what each one catches

| Leg | Assertion | Catches |
|---|---|---|
| 1 | `base_chunks == len(starts)` | The baseline decoded a different number of chunks than the geometry predicts — i.e. our model of `chunk_starts` is wrong, so every count below is wrong. **This is the gate's own self-check.** |
| 1 | `status.resumed_from_s == 0.0` | `--no-resume` silently consuming a checkpoint. |
| 1 | `not baseline.json.ckpt.exists()` | The checkpoint outliving a completed run (`transcribe.py:263`). |
| 2 | `rc == -SIGKILL` | F5. A run that finished normally, or one still alive, being called "interrupted". |
| 2 | `killpg(pgid, 0)` → `ProcessLookupError` | **F1.** An orphan surviving the kill. |
| 2 | `pgrep -f marker` → empty | An orphan that escaped the process group. |
| 2 | checkpoint bytes frozen over 5 s | Progress continuing after the kill (weak, see §2). |
| 2 | `0 < next_start < total` | **F6.** A checkpoint that is empty or complete, either of which makes leg 3 vacuous. |
| 2 | `ckpt.fingerprint.media == media.resolve()` | **F7.** The fingerprint regressing onto the decoded temp wav — the failure mode that is invisible on `.wav` input. |
| 2 | `not out.exists()` | A partial transcript written as if complete. |
| 3 | `res_chunks == len(starts) - skipped` | **F2.** A resume that restarted from zero. |
| 3 | `resumed_from_s == next_start / RATE` | **F2**, independently and exactly. |
| 3 | `"resuming from" in stderr` | `read_checkpoint` returning `None`. |
| 4 | `sha256(baseline) == sha256(resumed)` | Resume producing a *different* transcript — the correctness claim proper. |
| 4 | both files parse as JSON, non-empty | A truncated transcript, which "does not announce itself — it merely looks short". |
| 4 | `not ckpt.exists()` | Checkpoint not cleaned up. |
| 5 | per-field: no `"resuming from"`, `resumed_from_s == 0`, no sentinel token in output, output `==` baseline | **A stale checkpoint being accepted, and — separately — being *rejected into a broken run*.** The last clause is the one that matters: rejection must yield the *correct full* transcript, not merely "not the corrupt one". |
| self-test | the gate fails when leg 3 runs `--no-resume` | **The harness testing nothing.** |

### Stale-checkpoint legs without a 2.4 GB download

`read_checkpoint` compares the *stored* fingerprint dict against the one the run
computes, for exact equality (`checkpoint.py:140`). Poisoning one field of the
stored copy is therefore equivalent to changing that property of the run, and
costs nothing:

```python
poisoned = json.loads(json.dumps(banked_ckpt))
poisoned["fingerprint"][field] = value          # model_id / chunk_s / media_mtime_ns
poisoned["tokens"][0]["text"] = " XYZZYSENTINEL"
stale_ckpt.write_text(json.dumps(poisoned))
```

**Flagged as an inference:** this tests the *rejection* path end-to-end but not
the *fingerprint construction* path — it assumes `fingerprint()` would actually
produce a different `model_id` if `--model` changed. `tests/test_checkpoint.py::
test_every_fingerprint_field_invalidates` covers construction at unit level, so
the pair is complete; neither alone is. The sentinel token is what proves
rejection rather than a coincidentally-matching transcript.

`chunk_s`/`overlap_s` are module constants with no CLI flag, so this poisoning
approach is the *only* way to exercise the geometry-change case end-to-end
short of monkeypatching the module. That is a design smell worth a bead:
**the fingerprint covers a property the CLI cannot vary.**

---

## 8. Shell script or pytest?

**Recommendation: pytest, `@pytest.mark.slow`, in the repo.** Reasons, in order:

1. The kill mechanism *needs* `start_new_session` + `killpg` + `waitpid`, and
   macOS ships no `setsid(1)`. The Python version is shorter and correct; the
   shell version is longer and approximate.
2. `set -e` semantics are the source of F3, F4 and F9. An `assert` that raises
   is not negotiable the way `|| true` is.
3. `conftest.py` already has `chunked_audio_path` (session-scoped, cached
   clip, skips cleanly without ffmpeg or `scratch/meeting.wav`) and `model_id`.
   The gate is ~40 lines of test on top of the helpers.
4. `pyproject.toml` registers `slow` **without deselecting it** — deliberately:
   "a test carved out of the normal invocation is a test nobody observes". The
   gate inherits that posture for free.

**Fragility, honestly:** this test spawns processes, sends SIGKILL, and polls
the filesystem. It is more fragile than the rest of the suite. Three
concessions keep it from destabilising `pytest`:

- Split the file: `tests/test_resume_gate.py`, `pytestmark = pytest.mark.slow`,
  and a module-level `skipif` on `scratch/meeting.wav` + ffmpeg, matching
  `chunked_audio_path`'s existing skip conditions.
- Use a **225 s** fixture clip (a `CLIP_SECONDS`-parameterised sibling of the
  existing 360 s one), not the 74-minute file. The full-file run stays a
  manual/CI-nightly invocation of the same test via an env var or `--media`.
- Keep the `.mov` leg and the three stale legs as separate test functions so a
  failure names the property, not "the gate".

**Keep `scratch/resume_gate.py` as well.** It is the form you want when
debugging the feature by hand or running the 74-minute gate detached, and it is
where `--self-test` lives most naturally. The two share the same assertions;
the script is the reference and the pytest module is the routine gate.

---

## 9. Residual gaps — what this gate still does *not* establish

Stated explicitly so nobody reads a green run as more than it is.

1. **Byte-identity is asserted on one clip.** It holds because `generate()`
   decodes each chunk from a fresh decoder state (`chunking.py:92-95`), so the
   claim is structural — but it is verified at one length, on one file, with
   one model.
2. **The kill lands between chunks, in practice.** The trigger fires when the
   checkpoint reaches a boundary, so the kill arrives just after a write.
   A kill landing *inside* `write_checkpoint`'s window is the interesting
   durability case and this gate does not target it deliberately. Fault
   injection into `atomic_write_text` is the right tool, and is a separate
   piece of work.
3. **The perf regression is unaddressed and shapes the design.** Every timeout
   here is generous specifically because per-chunk cost is unstable (§5). If
   the regression is fixed, the timeouts stay correct; they are ceilings, not
   assertions.
4. **The `.mov` leg exercises a 320×240 `testsrc2`**, not a real screen
   recording. It proves the fingerprint survives temp-wav churn, which is the
   point; it proves nothing about real screen-recorder container quirks.
5. **`--model` is never actually varied.** See §7's flag.
