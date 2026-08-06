# Python project tooling: practices and tools that raise the odds it works

A standalone checklist. Ordered by what catches real failures, not by what looks
tidy.

The grading standard throughout: **does this catch a bug that ordinary linting
and a green `pytest` run would miss?** That bar comes from a build audit where
five separate defects shipped past a fully green suite — a resume feature that
silently restarted, a factory function nothing called, a diarization pass that
never ran, an error path never exercised, and a checkpoint format assumed rather
than parsed. Every one was "covered". None were asserted on.

---

## Part 1 — Practices

Cheap to apply, no judgement required, each one closes a class of failure.

### Enforcement

| # | Practice | Catches | Effort |
|---|---|---|---|
| 1 | Explicit `[tool.ruff.lint] select` — not bare defaults | Bare defaults are roughly `E4,E7,E9,F`. Everything `I` (imports), `UP` (upgrades), `B` (bugbear), `SIM`, `ANN` (annotations), `D` (docstrings), `PTH`, `ARG` would flag goes unseen | 10 min |
| 2 | Type checker in `strict`, not `basic` | Annotation coverage held by habit rather than enforced; `Any` leaking in from untyped third-party edges | 5 min + fixes |
| 3 | `py.typed` marker (PEP 561) | Without it your annotations are invisible to every consumer | 1 min |
| 4 | A real `[build-system]` | Its absence forces a `pythonpath` hack so tests can import the package | 5 min |
| 5 | `__all__` per module | Public surface is otherwise implicit; refactors silently break consumers | 10 min |

### Documentation

| # | Practice | Catches | Effort |
|---|---|---|---|
| 6 | Docstrings on every public function, with `Raises:` | Callers cannot know which exceptions to handle without reading the source | 30 min |
| 7 | A stated docstring convention (Google/NumPy/reST), enforced by ruff `D` | Ad-hoc styles drift and tooling cannot check them | 5 min |

### Runtime and observability

| # | Practice | Catches | Effort |
|---|---|---|---|
| 8 | `logging`, not `print` | Fine for a CLI; wrong the moment anything imports the module | 20 min |
| 9 | **Test timeouts** | A hung test is indistinguishable from a slow one. This cost ~2 hours of manual `ps` forensics in the audit | 2 min |
| 10 | **Warning filters** | Dependency noise buried the pass count on three consecutive runs, so the result could not be read | 2 min |

### The suite itself

| # | Practice | Catches | Effort |
|---|---|---|---|
| 11 | **Measure coverage** | "We believe it is tested" is not a measurement | 10 min |
| 12 | **Split fast/slow, fast by default** | A 4½-minute suite gets skipped, and a skipped gate is not a gate | 5 min |
| 13 | **End-to-end gates inside the suite, not in scratch** | Gates that run only when someone remembers are the ones that catch silently-skipped features | varies |
| 14 | **Every gate must be provably able to fail** | Ship a self-test that disables the feature and asserts the gate goes red. A gate never seen failing is not evidence | 30 min |

### Automation

| # | Practice | Catches | Effort |
|---|---|---|---|
| 15 | pre-commit hooks | Config from #1 and #2 is advisory until something enforces it on every commit | 15 min |
| 16 | CI on push | Local green is one machine's opinion | 20 min |

### The pattern behind #9–#14

All six concern **observability of the verification itself** rather than of the
code. It is common to gate runtime behaviour rigorously and never ask whether
the checks can be trusted to run, finish, and be read. That gap is where silent
failure lives.

---

## Part 2 — Tools

### Tier 1 — catches what a green suite hides

| Tool | Buys | Why it earns its place |
|---|---|---|
| **mutmut** | Mutation testing | **Highest-value single addition.** Mutates your code and asserts the tests notice. Coverage proves a line *executed*; mutation testing proves it was *asserted on*. Every silent-green defect in the audit was covered-but-unasserted |
| **pytest-timeout** | Aborts hanging tests | Converts a stall into a failure with a stack trace instead of a process to diagnose by hand |
| **pytest-cov** / **coverage** | Which lines never execute | The cheapest possible answer to "is this actually tested" |
| **hypothesis** | Property-based testing | Pays off wherever an invariant is easier to state than an example — parsers, merge algorithms, anything with a "for all inputs" property |
| **syrupy** | Snapshot testing | For code emitting large structured artifacts, a schema change becomes a reviewable diff rather than a hand-written assertion nobody updates |
| **pytest-randomly** | Shuffles test order | Catches inter-test dependence, which hides in any suite with shared fixtures |

### Tier 2 — correctness and friction

| Tool | Buys |
|---|---|
| **pydantic** | Runtime validation **at trust boundaries** — external JSON, config, LLM output, anything a previous process wrote |
| **pydantic-settings** | Typed configuration from env/files, validated at startup rather than on first use |
| **pytest-xdist** | Parallel execution. A slow suite is a skipped suite |
| **deptry** | Unused, missing, and transitively-relied-upon dependencies |
| **pip-audit** | CVE scan of the resolved tree |
| **typer** | Derives a CLI from type hints — near-free in an annotated codebase, deletes hand-rolled `argparse` |
| **structlog** | Structured logs. Worth it wherever the program already emits machine-readable state |
| **pytest-benchmark** | Timing regressions caught as test failures |

### Tier 3 — the current baseline, for reference

`uv` · `ruff` · `pyright` · `pytest` · `pre-commit`

- **uv** replaced pip, pip-tools, virtualenv and pipx; Rye was archived Feb 2026
- **ruff** replaced black + flake8 + isort outright
- **pyright** remains the safest type checker on conformance grounds
- **pytest** is uncontested

### Watch, do not adopt yet

| Tool | Status |
|---|---|
| **ty** (Astral) | 10–60× faster than pyright, but **53% typing-spec conformance vs pyright's 98%** as of 2026-04. Likely to win; not yet |
| **pyrefly** (Meta) | Same class, same caveat |
| **codspeed** | CI performance regressions. Only meaningful on a machine quiet enough to measure — a saturated host produces false positives costlier than the tool saves |

---

## What pydantic is and is not

Worth stating because it is commonly misunderstood as a type-checking
replacement.

Pydantic **consumes** type hints to build runtime validators. It never inspects
function bodies:

```python
def midpoint(bucket: str) -> int:
    return bucket + 1     # static checker: error. pydantic: never sees this.
```

Two further properties that surprise people:

- It **coerces by default** (`"1"` → `1`). Strict mode exists but is opt-in, so
  out of the box it is *looser* than expected
- It validates at model boundaries only, and costs CPU per validation — so it
  does not belong on a hot internal path

Use it where untrusted bytes enter. Use a static checker everywhere else. They
are complementary, not alternatives.

---

## Adoption order

1. **#9 timeouts and #10 warning filters** — two minutes, and they make every
   subsequent run readable and non-hanging
2. **#1, #2 ruff select and strict types** — expect a violation batch; that
   batch *is* the deliverable, it shows where habit lapsed
3. **#15, #16 pre-commit and CI** — makes 1 and 2 binding rather than advisory
4. **#11, #12 coverage and the fast/slow split** — measure before optimising
5. **mutmut** — last, because it produces the most uncomfortable report and is
   only worth reading once the cheap gaps are closed

---

## The one-line version

Static checking proves the code is consistent. Tests prove it did something.
**Mutation testing proves the tests would notice if it stopped.** Most projects
buy the first two and assume the third.
