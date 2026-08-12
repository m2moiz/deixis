# Why the marks pass has no resume

`jaano suno` checkpoints and resumes; `jaano dekho` does not, and a kill
part-way through a scan throws the whole decode away. This is the record of why
that asymmetry is deliberate — including the measurement that came out the
opposite way from what the argument against a resume expected.

Bead `jaano-yny` asked for either a checkpoint or this document.

Every timing and every comparison below is verbatim output of
[`scratch/marks_resume_probe.py`](../scratch/marks_resume_probe.py), on an M2
with ffmpeg 8.1.2; the line counts in §5 are `wc -l`. Nothing here is
estimated. Two real screen recordings, both 2940x1912: the 33-minute
reference `visual-marks.md` uses throughout, and a 74-minute one — the length
the bead was actually worried about, measured rather than extrapolated.

## The decision

**No checkpoint.** A killed scan is re-run by retyping the same command, and
what that costs is bounded by nine minutes on the longest recording anyone has
put through it. The machinery that would save part of those nine minutes is the
~1,300 lines the ASR half needed, plus a second trust boundary in a pass that
currently has none.

**And if the cost is ever paid, it should buy a cache, not a checkpoint** — see
[§4](#4-the-artifact-worth-writing-is-not-a-checkpoint). That is the finding
that decided this, not the size of the saving.

## 1. What a kill costs

```
Screen Recording 2026-07-31 at 09.35.23.mov   (73.8 min, 4.2 GB)
  full scan                   536.6s  frames= 4428  47.6 MB   8.3x realtime

Screen Recording 2026-08-06 at 15.07.10.mov   (33.3 min, 1.25 GB)
  full scan                   189.5s  frames= 1997  21.5 MB  10.5x realtime   (cold)
  full scan                   124.8s  frames= 1997  21.5 MB  16.0x realtime   (warm)
```

The 33-minute file was scanned twice, once on first touch and once after the
probe had already read it through: 189.5 s against 124.8 s, a 1.5x spread from
page cache alone. Both are quoted because neither is *the* number, and a resume
argued from a single timing would be argued from whichever of the two happened
to be taken.

**Nine minutes is the whole exposure**, on the longest real recording available
and at the default `fps=1` / 128x84 geometry. Redoing it is one command, it
needs no model, no network and no GPU, and `mark_video` writes its output
through `atomic_write_text` at the very end — so an interrupted scan leaves
nothing behind, nothing half-written, and nothing to clean up. The failure mode
this bead protects against is "wait nine minutes again", not "lose work" and not
"recover from a corrupt state".

Set that beside what the ASR resume protects. On the same class of file
`docs/resume-gate-design.md` §5 measures per-chunk cost at 60–90 s across ~43
chunks, with a variance larger than its mean (the *identical* baseline command
on the *identical* input took 74 s on one invocation and 19 s on another). That
pass loads 2.4 GB of weights before it starts and holds the GPU throughout. The
two are not the same problem at the same scale.

> Both figures above disagree with `visual-marks.md`'s cost table, which records
> 518 s for the 33-minute file, and with `dekho.py`'s "runs at ~4x" — those two
> agree with each other and with nothing measured here. Two fresh runs at
> different lengths agree instead on 8–10x. Something under this call has moved
> since that table was written and this document does not know what, so both
> figures are left standing where they are rather than overwritten. Filed as
> `jaano-875`.

## 2. The objection that turned out to be wrong

The argument against a resume was expected to be a correctness one: a resume has
to restart the decode part-way in, `extract_tile_grid` has no seek, and
`media.extract_frame` already documents at length that a container seek can
"snap to the next keyframe and return a frame from the WRONG MOMENT with exit 0
and no warning". If a seeked scan samples even slightly different instants than
an uninterrupted one, a resumed run produces *different marks* — and, since both
lists look like plausible marks, it does so silently.

That was checked rather than asserted, and it is false on these files:

```
Screen Recording 2026-07-31 at 09.35.23.mov, resuming at 2214s of 4428
  fast seek (-ss before -i)   277.3s  frames= 2214  IDENTICAL
  exact seek (-ss after -i)   431.6s  frames= 2214  IDENTICAL
  preroll seek (both)         197.2s  frames= 2214  IDENTICAL
```

All three seek forms reproduce the tail of the full scan **byte for byte**, over
2,214 frames. The same three came back IDENTICAL over 999 frames on the
33-minute file, and over 131 frames on the 320x240 `.mov` fixture in
`scratch/gate_work/`. A resumable scan is therefore *feasible* and could even be
exact. The case against it is a cost case, not an impossibility case, and this
document says so rather than reaching for the tidier argument.

An all-IDENTICAL run is exactly the answer a comparison that is not looking
would also produce, so the probe ships `--self-test`, which offsets the
comparison by one frame and requires every leg to go red. It was run:

```
$ uv run python scratch/marks_resume_probe.py scratch/gate_work/recording.mov 100 --self-test
  SELF TEST: comparing against a one-frame offset; all legs MUST differ
  fast seek (-ss before -i)     0.1s  frames=  131  DIFFERS at 130/130 frames, first at frame 0
  exact seek (-ss after -i)     0.2s  frames=  131  DIFFERS at 130/130 frames, first at frame 0
  preroll seek (both)           0.2s  frames=  131  DIFFERS at 130/130 frames, first at frame 0
SELF TEST PASSED: the comparison can report a difference
```

Two caveats on that result, because it is one container format: these are
QuickTime H.264 files with a reliable index. `media.py` records that MPEG-TS has
no global index and that its fast seek returns nothing at a timestamp the file
plainly contains. A real implementation would owe the preroll form (which is what
`extract_frame` settled on for exactly this reason) and a test on a container
without an index.

## 3. What the saving would actually be

The preroll seek decoded the second half of the 74-minute file in 197.2 s where
the full scan took 536.6 s, so a resume landing at the midpoint saves on the
order of **five minutes** on the worst file in hand. On the 33-minute file the
same pair is 65.9 s against 124.8 s — 53%, which is what a resume at the
halfway mark should cost if decode were uniform.

Flagged as an inference rather than a measurement: 197.2 s is *below* half of
536.6 s, which it should not be. The probe read the same 4.2 GB file four times
in a row and §1 shows page cache alone worth 1.5x, so warmth is the likely
confound, and the two halves of a screen recording are in any case not equally
busy. The honest form of this number is "roughly half a scan", not 63%.

That saving is real and larger than the argument against a resume assumed. It
is not what decides this — §4 is.

## 4. The artifact worth writing is not a checkpoint

`extract_tile_grid` returns the whole file as one array on purpose. Its
docstring says why: 21 MB for 33 minutes "is why the whole thing is returned as
one array rather than streamed — and why the caller can afford to sweep
parameters over it without decoding twice". The 54-configuration sweep in
`visual-marks.md` is that affordance being used.

A checkpoint has to spill partial tile grids to disk, keyed to a fingerprint of
the source and the geometry. But *that is most of a tile-grid cache already* —
and the cache is strictly the better feature, because the same 47.6 MB on disk
would serve three things instead of one:

| Want | Cache | Checkpoint |
|---|---|---|
| Re-run with a different `--budget` or `--min-gap` without re-decoding | yes | no |
| Sweep `--delta` over one decode | yes | no |
| Continue after a kill | yes | yes |

Only the third row is what the bead asked for, and it is the rarest of the
three: nobody has yet lost a scan, whereas re-running with a different budget is
the ordinary way the parameters were chosen in the first place. Building the
checkpoint would spend the fingerprinting and the on-disk format — the expensive,
error-prone parts — on the least useful of the three.

So the decision is not only "not now". It is "not this shape". If a future run
is long enough to justify writing tiles to disk, the thing to write is
`--tiles <path>`: a whole-file grid cache, from which resume falls out as the
degenerate case of a cache that happens to be short.

## 5. What the machinery costs, counted

The ASR resume is the only comparable thing in this repo, and it is not 209
lines of `checkpoint.py`:

| | lines |
|---|---|
| `jaano/checkpoint.py` | 209 |
| `tests/test_checkpoint.py` | 296 |
| `tests/test_resume_cli.py` | 198 |
| `tests/test_resume_gate.py` | 77 |
| `tests/gate_helpers.py` | 70 |
| `scratch/resume_gate.py` | 463 |
| **total, code and gate** | **1,313** |
| `docs/resume-gate-design.md`, the design that made the gate trustworthy | 513 |

The gate is not optional padding on that bill, and `resume-gate-design.md` is
the reason: F2 in that document is that **a resume which silently restarts
produces a byte-identical, completely correct output**. `shasum` matches,
`sentences ==` matches, the feature is inert and every check is green. A marks
resume has the identical property — `select_marks` is a pure function of the
tile array, so a resume that silently rescanned from zero would emit exactly the
right marks. Anything less than a kill-and-count gate would be a check that
cannot fail, which `tooling-gaps.md` #14 names as worse than no check at all.

There is a second cost that does not show up as lines. `checkpoint.py` calls
itself "the one trust boundary in jaano that reads bytes it did not produce in
this run". The marks pass has no such boundary today; the only file it reads
that it did not make is the transcript, and that already has a guard. A
checkpoint adds a second one, and with it a pydantic document model, an
invalidation fingerprint, and the question of what happens when the fingerprint
matches but should not.

That last question is worse here than for ASR. ASR's fingerprint covers the
model, the source and the chunk geometry — all things a user chooses. A tile
grid's correctness depends on the *decoder*: ffmpeg's version, the swscale
kernel, any hardware acceleration path. A point release that shifted one tile
mean by one grey level would splice two incompatible halves into a mark list
that looks entirely reasonable. §1's unexplained 2.7x timing gap is not evidence
that pixels have changed, but it is evidence that this stack moves underneath a
recorded measurement without anyone noticing.

## 6. What would change the answer

Written down so this is a decision with an expiry rather than a permanent
refusal:

- **The scan stops being one ffmpeg call.** Attaching a description to each mark
  is the next piece of work (`vlm-legibility.md`), and that is per-mark model
  inference — expensive, restartable at a natural boundary, and the shape a
  checkpoint is actually for. Revisit then, and checkpoint *that*, not the
  decode.
- **A scan exceeds roughly half an hour.** A finer grid or a higher `--fps`
  buys that quickly; `visual-marks.md` measures the scaler, not the pixel count,
  as the cost driver.
- **Somebody actually loses a scan and minds.** No instance so far. One would be
  worth more than this whole analysis.

In all three cases, build [§4](#4-the-artifact-worth-writing-is-not-a-checkpoint)'s
cache and take the resume as a consequence of it.
