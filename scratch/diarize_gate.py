#!/usr/bin/env python
"""An end-to-end diarization gate that cannot silently pass.

Run:
    uv run python scratch/diarize_gate.py --media scratch/meeting.wav
    uv run python scratch/diarize_gate.py --media scratch/meeting.wav --self-test

Same two rules as scratch/resume_gate.py, and for the same reason:

  1. Nothing waits on a clock. Every wait polls an observable with an explicit
     timeout, so the gate cannot become a race that happens to pass on a fast
     machine.
  2. Every leg carries a positive assertion that the thing under test actually
     HAPPENED. "It produced a labelled transcript" is not one of those: a run
     that skipped diarization also finishes, also exits 0, and also writes a
     perfectly good transcript. The phase has to be caught while it is running.

`--self-test` runs leg 1 with --no-diarize. The gate MUST then fail, which is
what makes a pass on the real run mean something.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_TIMEOUT_S = 3600.0
POLL_S = 0.2


class GateFailure(AssertionError):
    """A gate assertion failed. The message is the diagnosis."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise GateFailure(msg)


def say(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# a run we can watch while it happens
# --------------------------------------------------------------------------


def run_and_watch(media: Path, out: Path, status: Path, log: Path,
                  extra: list[str]) -> tuple[list[str], float]:
    """Transcribe `media`, polling the heartbeat throughout.

    Returns every distinct state observed, in order, and the wall clock.

    `sys.executable` rather than `uv run`, as in resume_gate: uv forks a child
    and waits, so the pid we hold would be the wrapper's, not the worker's.

    The states are collected DURING the run and not reconstructed afterwards.
    "diarizing" leaves nothing behind in the final status file -- it is
    overwritten by "done" seconds later -- so a gate that only inspects the end
    state cannot tell a diarized run from a skipped one.
    """
    marker = uuid.uuid4().hex
    status = status.with_name(f"{status.name}.{marker}")
    argv = [
        sys.executable, "-m", "jaano.suno",
        str(media), "-o", str(out), "--status", str(status), *extra,
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("wb")
    started = time.monotonic()
    proc = subprocess.Popen(
        argv, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
    )
    fh.close()

    states: list[str] = []
    while proc.poll() is None:
        try:
            state = json.loads(status.read_text())["state"]
        except (OSError, KeyError, json.JSONDecodeError):
            state = None
        if state and (not states or states[-1] != state):
            states.append(state)
        if time.monotonic() - started > RUN_TIMEOUT_S:
            os.killpg(os.getpgid(proc.pid), 9)
            raise GateFailure(f"run exceeded {RUN_TIMEOUT_S:.0f}s")
        time.sleep(POLL_S)

    elapsed = time.monotonic() - started
    tail = log.read_text(errors="replace")[-3000:]
    check(proc.returncode == 0, f"run exited rc={proc.returncode}\n--- {log} ---\n{tail}")

    # The final write can land between our last poll and the process exiting.
    try:
        final = json.loads(status.read_text())["state"]
        if not states or states[-1] != final:
            states.append(final)
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    return states, elapsed


# --------------------------------------------------------------------------
# what the labelled transcript says
# --------------------------------------------------------------------------


def report_speakers(payload: dict) -> Counter:
    """Print the histogram a human has to eyeball, and return it.

    Words as well as sentences: a cluster can hold a handful of sentences and
    almost no speech, which is what over-clustering looks like from here.
    """
    labels = payload["speakers"]
    sentences: Counter = Counter()
    words: Counter = Counter()
    for s in payload["sentences"]:
        sentences[s["speaker"]] += 1
        words[s["speaker"]] += len(s["tokens"])

    say(f"    speakers   : {labels}")
    for i, label in enumerate(labels):
        say(f"      [{i}] {label:<12} {sentences[i]:>4} sentences  {words[i]:>6} words")

    runs = 1
    previous = None
    for s in payload["sentences"]:
        if previous is not None and s["speaker"] != previous:
            runs += 1
        previous = s["speaker"]
    say(f"    speaker runs across the transcript: {runs}")
    return sentences


def report_straddles(payload: dict, media: Path) -> None:
    """Re-derive §2.4's counts from the same turns the run used.

    A second diarization pass of the same file, in-process. It costs ~21s and
    it is the only way to see INSIDE the vote from out here: the transcript
    records who won each sentence, never how close it was.
    """
    from jaano.diarize import speaker_turns
    from jaano.merge import TurnIndex

    result = speaker_turns(media)
    index = TurnIndex(result.turns)

    unanimous = straddling = all_in_gap = 0
    gap_tokens = 0
    total_tokens = 0
    for s in payload["sentences"]:
        votes: Counter = Counter()
        for token in s["tokens"]:
            total_tokens += 1
            who = index.speaker_at(token["t"])
            if who is None:
                gap_tokens += 1
            else:
                votes[who] += 1
        if not votes:
            all_in_gap += 1
        elif len(votes) == 1:
            unanimous += 1
        else:
            straddling += 1

    n = len(payload["sentences"])
    say(f"    turns      : {len(result.turns)}  labels {result.labels}")
    say(f"    sentences  : {n}  unanimous={unanimous}  "
        f"straddling={straddling} ({straddling / n * 100:.1f}%)  "
        f"all-tokens-in-gap={all_in_gap}")
    say(f"    tokens in a VAD gap: {gap_tokens} of {total_tokens}")


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def gate(media: Path, work: Path, self_test: bool, probe: bool) -> None:
    work.mkdir(parents=True, exist_ok=True)

    # -- 1. a labelled run, watched -----------------------------------------
    extra = ["--no-resume"] + (["--no-diarize"] if self_test else [])
    say(f"[1] labelled run{'  -- SELF TEST: --no-diarize, the gate MUST fail below' if self_test else ''}")
    labelled_out = work / "labelled.json"
    states, elapsed = run_and_watch(
        media, labelled_out, work / "lab.status", work / "lab.log", extra
    )
    say(f"    {elapsed:.0f}s wall clock, states seen: {states}")

    # THE assertion. Everything else in this file would also pass on a run that
    # never diarized at all.
    check(
        "diarizing" in states,
        f"the heartbeat never reported 'diarizing'; states were {states}. "
        f"The pass did not run, or ran too fast to observe -- either way this "
        f"gate has not seen what it claims to test.",
    )
    check(
        states.index("running") < states.index("diarizing") < states.index("done"),
        f"'diarizing' is out of order in {states}",
    )

    payload = json.loads(labelled_out.read_text())
    check("diarization" in payload, "the labelled transcript carries no provenance key")
    check("speakers" in payload, "the labelled transcript carries no speakers list")
    say(f"    provenance : {payload['diarization']}")

    # -- 2. additive against a --no-diarize run -----------------------------
    say("[2] additive against --no-diarize")
    plain_out = work / "plain.json"
    plain_states, _ = run_and_watch(
        media, plain_out, work / "plain.status", work / "plain.log",
        ["--no-resume", "--no-diarize"],
    )
    check("diarizing" not in plain_states,
          f"--no-diarize still ran the pass: {plain_states}")
    plain = json.loads(plain_out.read_text())
    check(set(plain) == {"audio", "model", "text", "sentences"},
          f"--no-diarize emitted {sorted(plain)}, not today's schema")
    check(plain["text"] == payload["text"], "the two runs disagree on the transcript text")
    check(len(plain["sentences"]) == len(payload["sentences"]),
          f"{len(plain['sentences'])} sentences unlabelled vs "
          f"{len(payload['sentences'])} labelled")

    for i, (before, after) in enumerate(zip(plain["sentences"], payload["sentences"])):
        check(set(after) - set(before) == {"speaker"},
              f"sentence #{i} gained {sorted(set(after) - set(before))}, not just 'speaker'")
        check({k: after[k] for k in before} == before,
              f"sentence #{i} was CHANGED by diarization, not merely annotated")
    say(f"    {len(plain['sentences'])} sentences: identical but for 'speaker'")

    # -- 3. the histogram ---------------------------------------------------
    say("[3] who spoke")
    per_speaker = report_speakers(payload)

    labels = payload["speakers"]
    silent = [labels[i] for i in range(len(labels)) if per_speaker[i] == 0]
    if silent:
        say(f"    absorbed by the vote (0 sentences won): {silent}")
    check(len(labels) >= 2,
          f"one speaker on a call known to have two: {labels}. Land the code, "
          f"leave the bead open -- choosing a diarizer is an operator decision.")
    if len(labels) > 4:
        raise GateFailure(
            f"{len(labels)} speakers on a call known to have two: {labels}. "
            f"Land the code, leave the bead open."
        )
    if len(labels) > 2:
        say(f"    NOTE: {len(labels)} clusters for a 2-person call. Known "
            f"over-clustering (plan §1.5). Not a kill IF the extras win nothing.")
        winners = [labels[i] for i in range(2, len(labels)) if per_speaker[i] > 0]
        check(
            not winners,
            f"a phantom cluster WON sentences: {winners}. The schema decision in "
            f"§3 rests on the sentence-level vote absorbing over-clustering; it "
            f"did not. STOP and re-decide.",
        )

    # -- 4. inside the vote -------------------------------------------------
    if probe:
        say("[4] straddle probe (a second diarization pass, ~21s)")
        report_straddles(payload, media)
    else:
        say("[4] straddle probe -- SKIPPED (--no-probe)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--media", type=Path, default=REPO / "scratch" / "meeting.wav")
    ap.add_argument("--work", type=Path, default=REPO / "scratch" / "diarize_gate_work")
    ap.add_argument("--self-test", action="store_true",
                    help="run leg 1 with --no-diarize; the gate MUST report FAIL")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    try:
        gate(args.media, args.work, args.self_test, not args.no_probe)
    except GateFailure as exc:
        say(f"\nFAIL: {exc}")
        if args.self_test:
            say("\nSELF TEST PASSED: the gate detects a run that never diarized.")
            return 0
        return 1
    if args.self_test:
        say("\nSELF TEST FAILED: the gate passed a run that never diarized. "
            "It is not testing what it claims to test.")
        return 1
    say("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
