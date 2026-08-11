# Surviving mutants, and why each one is accepted

`just mutate` mutates `jaano/merge.py`, `jaano/checkpoint.py` and
`jaano/atomic.py` and reports which mutations the test suite fails to notice.
Coverage proves a line executed. This proves something asserted on what it did.

```
192 mutants   179 killed   13 survived
```

All 13 survivors are accepted. Each is either **provably equivalent** — the
mutated program cannot behave differently from the original — or **untestable
on this platform** without machinery that costs more than it protects. A
survivor with neither justification is a missing test, and the first run had
16 of those. They are now tests.

Re-read this file before adding a suppression. "Chasing 100%" is how mutation
testing turns into busywork; the point is the triage, not the number.

---

## Provably equivalent

**`merge._distance` mutants 1 and 2** — `turn.start <= t <= turn.end` weakened
to `<` at either end.

At `t == turn.start` the original returns `0.0` early. The mutant falls through
to `min(abs(t - start), abs(t - end))` = `min(0, duration)` = `0.0`. Identical.
Same argument at the other bound. The early return is a shortcut, not a
decision, so removing its edge cases changes nothing.

**`merge.label_sentence` mutant 13** — `votes[speaker] += 1` becomes `+= 2`.

Every vote doubles, so every comparison is between `2a` and `2b` where it was
between `a` and `b`. Ordering is preserved and ties stay ties, so `max()` and
the tie-break pick the same speaker for every possible input. Unkillable by
construction.

**`checkpoint.read_checkpoint` mutants 5, 9, 10, 11** — the first argument of
`cast()` mangled to `None`, `"XXdict[str, Any]XX"`, `"dict[str, any]"`,
`"DICT[STR, ANY]"`.

`typing.cast` returns its second argument unchanged and never evaluates the
first at runtime. These are edits to a type annotation expressed as a call.
Nothing at runtime can observe them; pyright would reject the malformed ones,
which is the check that actually applies here.

---

## Untestable without machinery that costs more than it protects

**`checkpoint.fingerprint` mutant 23** — `version("parakeet-mlx")` becomes
`version("PARAKEET-MLX")`.

Distribution names are normalised case-insensitively by `importlib.metadata`,
so both calls return the same string. Equivalent in practice; killing it would
mean asserting on a lookup that PEP 503 defines as case-insensitive.

**`atomic.atomic_write_text` mutants 5, 7, 11** — `encoding="utf-8"` becomes
`None`, is dropped, or becomes `"UTF-8"`.

`"UTF-8"` is a codec alias for `"utf-8"`; those are the same encoder. The other
two fall back to the locale default, which on this machine and on the CI runner
IS utf-8, so the bytes on disk are byte-identical. `test_the_file_is_written_as
_utf8_whatever_the_locale_is` asserts on raw bytes and still cannot separate
them — the difference only appears under a non-UTF-8 locale, which would mean
running the suite under one.

**Kept anyway** because the explicit argument is the thing that makes the
mutant equivalent. Delete it and the file becomes locale-dependent on a machine
we do not test on.

**`atomic.atomic_write_text` mutants 18 and 19** — `tmp.unlink(missing_ok=True)`
becomes `missing_ok=False` or `None`.

That line runs only in the `except BaseException` cleanup, and reaching it with
the temp file already gone requires a failure between `tmp.open()` succeeding
and the write completing, plus an external deletion in that window. Fault
injection at that precision costs more than the bug it would catch: the worst
case is a `FileNotFoundError` masking the original exception during cleanup of
a write that already failed.

---

## What the first run actually bought

Sixteen mutants that were **not** equivalent, on lines that were already at 98%
coverage. Each is now a test:

| Mutant | The gap |
|---|---|
| `_distance` `t + start` / `t + end` | sign slip invisible unless the corrupted term is the one `min` picks |
| `_distance` `return 1.0` | zero-distance never asserted against a nearer neighbour |
| `nearest_speaker` `j <= len` | off-the-end index never reached; no test asked for a time past the last turn |
| `fingerprint` six fields → `None` | the existing test compares two fingerprints from the SAME function, so a field nulled on both sides still differs where it differed before |
| `write_checkpoint` `fsync=True` → `False`/`None`/dropped | durability is invisible to a process reading its own page cache |
| `TurnIndex.__init__` message text | `pytest.raises(ValueError)` accepted any message |

The `fsync` one is the sharpest illustration. Three mutants removed the
durability guarantee that exists because losing a checkpoint costs minutes of
GPU time, and the suite was completely silent. Coverage said 98%.
