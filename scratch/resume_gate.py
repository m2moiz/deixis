#!/usr/bin/env python
"""An end-to-end resume gate that cannot silently pass.

Run:
    uv run python scratch/resume_gate.py --media scratch/clip225.wav
    uv run python scratch/resume_gate.py --media scratch/clip225.wav --self-test
    uv run python scratch/resume_gate.py --media scratch/clip225.wav --mov

Design notes live in docs/resume-gate-design.md. The two rules the whole file is
built around:

  1. Nothing waits on a clock. Every wait is a poll on an observable with an
     explicit timeout and a message on expiry, so the gate cannot become a race
     that happens to pass on a fast machine.
  2. Every leg carries a positive assertion that the thing under test actually
     happened -- not just that the run finished. "It produced a correct
     transcript" is exactly what a resume that silently restarts also produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RATE = 16_000
CHUNK_S, OVERLAP_S = 120.0, 15.0

# Nothing here waits on a clock, but every poll needs a ceiling. These are
# generous: they exist to turn a hang into a loud failure, not to assert speed.
FIRST_CHUNK_TIMEOUT_S = 600.0
RUN_TIMEOUT_S = 1800.0
REAP_TIMEOUT_S = 30.0
GROUP_EMPTY_TIMEOUT_S = 30.0
FREEZE_WINDOW_S = 5.0
POLL_S = 0.25


class GateFailure(AssertionError):
    """A gate assertion failed. The message is the diagnosis."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise GateFailure(msg)


def say(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def chunk_starts(total_samples: int) -> list[int]:
    return list(range(0, total_samples, int(CHUNK_S * RATE) - int(OVERLAP_S * RATE)))


def wav_samples(path: Path) -> int:
    with wave.open(str(path)) as w:
        check(w.getframerate() == RATE, f"{path} is {w.getframerate()} Hz, expected {RATE}")
        return w.getnframes()


# --------------------------------------------------------------------------
# the worker: a real CLI process we can prove we killed
# --------------------------------------------------------------------------


@dataclass
class Worker:
    proc: subprocess.Popen
    pgid: int
    marker: str
    log: Path

    def stderr_text(self) -> str:
        return self.log.read_text(errors="replace")


def launch(media: Path, out: Path, status: Path, log: Path, extra: list[str]) -> Worker:
    """Start a transcription we own outright.

    `sys.executable` rather than `uv run`: `uv run` execs nothing -- it FORKS a
    child interpreter and waits, so Popen.pid (and shell `$!`) is uv's pid, not
    the worker's. SIGKILL to uv reaps the wrapper and orphans the process that
    is actually holding the GPU. That is the bug this whole file exists to make
    impossible, and the fix is to not have a wrapper.

    `start_new_session=True` puts the child in a fresh process group and
    session, so (a) one killpg reaches every descendant -- ffmpeg during .mov
    extraction is a real one -- and (b) that group is disjoint from the gate's
    own, so killpg cannot signal the gate itself. A plain background job in a
    non-interactive shell shares the script's process group; `kill -- -$PGID`
    there is suicide.

    The marker is a uuid on the argv. It is never used to kill anything -- it
    is how we audit, afterwards, that nothing carrying it survived.
    """
    marker = uuid.uuid4().hex
    status_marked = status.with_name(f"{status.name}.{marker}")
    argv = [
        sys.executable, "-m", "deixis.transcribe",
        str(media), "-o", str(out), "--status", str(status_marked), *extra,
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("wb")
    proc = subprocess.Popen(
        argv, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    fh.close()
    return Worker(proc=proc, pgid=os.getpgid(proc.pid), marker=marker, log=log)


def wait_until(pred, timeout_s: float, what: str, worker: Worker | None = None):
    """Poll `pred` until it returns something truthy. Never sleeps blindly."""
    deadline = time.monotonic() + timeout_s
    while True:
        if worker is not None and worker.proc.poll() is not None:
            raise GateFailure(
                f"worker exited (rc={worker.proc.returncode}) before {what}\n"
                f"--- tail of {worker.log} ---\n{worker.stderr_text()[-2000:]}"
            )
        got = pred()
        if got:
            return got
        if time.monotonic() > deadline:
            raise GateFailure(
                f"timed out after {timeout_s:.0f}s waiting for {what}"
                + (f"\n--- tail of {worker.log} ---\n{worker.stderr_text()[-2000:]}"
                   if worker else "")
            )
        time.sleep(POLL_S)


def read_ckpt(path: Path) -> dict | None:
    """Read a checkpoint, tolerating the instant it is being replaced."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def kill_and_prove_dead(w: Worker, ckpt: Path) -> None:
    """SIGKILL the group, then establish that nothing survived it.

    Four independent pieces of evidence, because the failure being designed out
    is "we sent a signal and assumed". Sending a signal is not an observation.
    """
    os.killpg(w.pgid, signal.SIGKILL)

    # 1. The worker itself is reaped, and reaped BY the signal we sent.
    try:
        rc = w.proc.wait(timeout=REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise GateFailure(
            f"pid {w.proc.pid} did not die within {REAP_TIMEOUT_S}s of SIGKILL"
        ) from exc
    check(rc == -signal.SIGKILL, f"worker exited rc={rc}, expected -{signal.SIGKILL} (SIGKILL)")

    # 2. No process remains in the group. This is the check that would have
    #    caught the original bug on the first run: signal 0 to a group with a
    #    live orphan in it succeeds instead of raising.
    def group_empty() -> bool:
        try:
            os.killpg(w.pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    wait_until(group_empty, GROUP_EMPTY_TIMEOUT_S, f"process group {w.pgid} to empty")

    # 3. Nothing ANYWHERE carries the marker -- catches an escapee that changed
    #    its own process group. pgrep as an auditor, never as the killer:
    #    pattern-matching is fine for "prove the set is empty" and hopeless for
    #    "identify the one process to signal".
    if shutil.which("pgrep"):
        found = subprocess.run(
            ["pgrep", "-f", w.marker], capture_output=True, text=True, check=False
        )
        check(
            found.returncode != 0,
            f"a process carrying marker {w.marker} outlived the kill: pids {found.stdout.split()}",
        )

    # 4. The checkpoint stops advancing. Weak on its own -- it only advances
    #    once per chunk, so a short window of stillness proves little -- and
    #    kept because it costs 5s and is the only check that speaks to
    #    "made no further progress" rather than "the pid is gone".
    before = read_ckpt(ckpt)
    time.sleep(FREEZE_WINDOW_S)
    after = read_ckpt(ckpt)
    check(
        before == after,
        f"the checkpoint advanced after the kill "
        f"({before and before['next_start']} -> {after and after['next_start']}): "
        f"something is still transcribing",
    )

    # 5. No atomic-write temp file stranded beside the output. SIGKILL skips
    #    atomic_write_text's BaseException cleanup, so a kill landing inside the
    #    write window leaves one. Reported, not asserted -- see the design doc.
    strays = sorted(ckpt.parent.glob(f".{ckpt.name}.*.tmp"))
    if strays:
        say(f"  note: stranded atomic-write temp files: {[p.name for p in strays]}")


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------


def run_to_completion(media: Path, out: Path, status: Path, log: Path,
                      extra: list[str]) -> Worker:
    w = launch(media, out, status, log, extra)
    try:
        rc = w.proc.wait(timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(w.pgid, signal.SIGKILL)
        raise GateFailure(
            f"run did not finish within {RUN_TIMEOUT_S}s\n{w.stderr_text()[-2000:]}"
        ) from None
    check(rc == 0, f"run exited rc={rc}\n--- {log} ---\n{w.stderr_text()[-3000:]}")
    return w


def final_status(w: Worker, status: Path) -> dict:
    p = status.with_name(f"{status.name}.{w.marker}")
    check(p.exists(), f"no status heartbeat at {p}")
    return json.loads(p.read_text())


def chunk_lines(w: Worker) -> int:
    """How many chunks this run actually decoded.

    stderr is a pipe, not a tty, so transcribe.py prints one bar line per chunk
    instead of rewriting one line. Counting them is an observable of work done
    that is completely independent of the checkpoint and the status file.
    """
    return sum(1 for line in w.stderr_text().splitlines() if line.strip().startswith("running"))


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def structured_diff(a: Path, b: Path) -> str:
    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    out = [f"text equal: {ja['text'] == jb['text']}",
           f"sentences: {len(ja['sentences'])} vs {len(jb['sentences'])}"]
    for i, (x, y) in enumerate(zip(ja["sentences"], jb["sentences"])):
        if x != y:
            out.append(f"first differing sentence #{i}:\n  A {x['start']:.3f} {x['text']!r}"
                       f"\n  B {y['start']:.3f} {y['text']!r}")
            break
    return "\n".join(out)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def make_mov(wav: Path, dest: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
         "-i", str(wav), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", str(dest)],
        check=True,
    )
    return dest


def gate(source_wav: Path, work: Path, as_mov: bool, self_test: bool,
         skip_stale: bool = False) -> None:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    total = wav_samples(source_wav)
    starts = chunk_starts(total)
    check(
        len(starts) >= 3,
        f"{source_wav} is {total / RATE:.0f}s -> {len(starts)} chunk(s). Resume needs a "
        f"kill point that is neither the start nor the last chunk: 3 chunks minimum, "
        f"which under {CHUNK_S:.0f}s/{OVERLAP_S:.0f}s geometry means > 210s of audio.",
    )
    media = make_mov(source_wav, work / "recording.mov") if as_mov else source_wav
    say(f"media   : {media.name} ({total / RATE:.0f}s, {len(starts)} chunks, starts={starts})")

    # -- 1. baseline -------------------------------------------------------
    base_out = work / "baseline.json"
    say("[1] baseline (--no-resume)")
    t0 = time.monotonic()
    wb = run_to_completion(media, base_out, work / "base.status",
                           work / "base.log", ["--no-resume"])
    base_chunks = chunk_lines(wb)
    say(f"    {time.monotonic() - t0:.0f}s, {base_chunks} chunk lines, sha {digest(base_out)[:12]}")
    check(base_chunks == len(starts),
          f"baseline decoded {base_chunks} chunks, geometry says {len(starts)}")
    check(final_status(wb, work / "base.status")["resumed_from_s"] == 0.0,
          "the baseline reports resuming from a checkpoint it was told to ignore")
    check(not (work / "baseline.json.ckpt").exists(),
          "the checkpoint outlived the run that completed")

    # -- 2. interrupted ----------------------------------------------------
    out = work / "resumed.json"
    ckpt = Path(str(out) + ".ckpt")
    kill_after = 1  # chunks; 1 <= kill_after <= len(starts) - 2
    say(f"[2] interrupted (SIGKILL the process group after chunk {kill_after})")

    w = launch(media, out, work / "int.status", work / "int.log", ["--no-resume"])
    say(f"    pid {w.proc.pid} pgid {w.pgid} marker {w.marker[:8]}")

    # The kill trigger is an OBSERVABLE, never a sleep. `sleep 45` is a race
    # against model load, ffmpeg extraction and a cold weights cache; on a slow
    # run it fires before any chunk lands and the gate proceeds to "resume"
    # a checkpoint that does not exist -- which is a full fresh run that then
    # compares equal to the baseline. Green, and nothing tested.
    def banked():
        c = read_ckpt(ckpt)
        return c if c and c["next_start"] >= starts[kill_after] else None

    banked_ckpt = wait_until(
        banked, FIRST_CHUNK_TIMEOUT_S,
        f"checkpoint to reach next_start >= {starts[kill_after]}", worker=w,
    )
    kill_and_prove_dead(w, ckpt)

    next_start = banked_ckpt["next_start"]
    say(f"    killed with next_start={next_start} ({next_start / RATE:.1f}s), "
        f"{len(banked_ckpt['tokens'])} tokens banked")

    # Mid-run, not empty and not complete. Both ends matter: next_start == 0
    # means nothing to resume, next_start == total means nothing left to do,
    # and either makes the next leg a no-op that still passes.
    check(next_start > 0, "checkpoint banked nothing")
    check(next_start < total, f"checkpoint is complete (next_start={next_start}, total={total})")
    check(bool(banked_ckpt["tokens"]), "checkpoint has a boundary but no tokens")
    check(banked_ckpt["fingerprint"]["media"] == str(media.resolve()),
          f"checkpoint keyed to {banked_ckpt['fingerprint']['media']}, not the source media "
          f"-- for a .mov that is a fingerprint that can never match again")
    check(not out.exists(), "a partial transcript was written as if it were complete")

    # -- 3. resume ---------------------------------------------------------
    extra = ["--no-resume"] if self_test else []
    say(f"[3] resume{' -- SELF TEST: --no-resume, the gate MUST fail below' if self_test else ''}")
    t0 = time.monotonic()
    wr = run_to_completion(media, out, work / "res.status", work / "res.log", extra)
    res_chunks = chunk_lines(wr)
    st = final_status(wr, work / "res.status")
    say(f"    {time.monotonic() - t0:.0f}s, {res_chunks} chunk lines, "
        f"resumed_from_s={st['resumed_from_s']}")

    # The assertion the parent harness was missing entirely. A resume that
    # silently restarts produces a correct transcript and passes any output
    # comparison; the ONLY things that distinguish it are these.
    expected_skipped = sum(1 for s in starts if s < next_start)
    check(
        res_chunks == len(starts) - expected_skipped,
        f"the resumed run decoded {res_chunks} chunks; a run that consumed the "
        f"checkpoint decodes {len(starts) - expected_skipped} "
        f"({len(starts)} - {expected_skipped} banked). It started over.",
    )
    check(
        abs(st["resumed_from_s"] - next_start / RATE) < 1e-6,
        f"status says resumed_from_s={st['resumed_from_s']}, checkpoint banked "
        f"{next_start / RATE} -- the checkpoint was not consumed",
    )
    check("resuming from" in wr.stderr_text(),
          "no 'resuming from' line: read_checkpoint returned None")

    # -- 4. equivalence ----------------------------------------------------
    say("[4] equivalence")
    for p in (base_out, out):
        check(p.exists() and p.stat().st_size > 0, f"{p} missing or empty")
        json.loads(p.read_text())  # a truncated transcript does not parse
    check(
        digest(base_out) == digest(out),
        f"resumed transcript differs from the uninterrupted one\n{structured_diff(base_out, out)}",
    )
    check(not ckpt.exists(), "the checkpoint outlived the run that completed")
    say(f"    byte-identical: {digest(out)[:12]}")

    # -- 5. stale checkpoint -----------------------------------------------
    if skip_stale:
        say("[5] stale-checkpoint rejection -- SKIPPED")
        return
    say("[5] stale-checkpoint rejection")
    for field, value in [("model_id", "mlx-community/not-the-model"),
                         ("chunk_s", 90.0),
                         ("media_mtime_ns", 1)]:
        stale_out = work / f"stale_{field}.json"
        stale_ckpt = Path(str(stale_out) + ".ckpt")
        # Take the real mid-run checkpoint we captured and poison one field.
        # read_checkpoint compares the stored fingerprint dict against the one
        # this run computes (checkpoint.py:140), so poisoning the stored copy is
        # equivalent to changing the run -- and does not cost a 2.4GB download.
        poisoned = json.loads(json.dumps(banked_ckpt))
        poisoned["fingerprint"][field] = value
        poisoned["tokens"][0]["text"] = " XYZZYSENTINEL"
        stale_ckpt.write_text(json.dumps(poisoned))

        ws = run_to_completion(media, stale_out, work / f"s_{field}.status",
                               work / f"s_{field}.log", [])
        sst = final_status(ws, work / f"s_{field}.status")
        check("resuming from" not in ws.stderr_text(),
              f"a checkpoint with a wrong {field} was accepted")
        check(sst["resumed_from_s"] == 0.0,
              f"a checkpoint with a wrong {field} was consumed")
        check("XYZZYSENTINEL" not in stale_out.read_text(),
              f"tokens from a checkpoint with a wrong {field} reached the transcript")
        check(digest(stale_out) == digest(base_out),
              f"rejecting a wrong-{field} checkpoint did not produce the correct "
              f"transcript\n{structured_diff(base_out, stale_out)}")
        say(f"    {field}: refused, full transcript, byte-identical to baseline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--media", type=Path, default=REPO / "scratch" / "clip225.wav")
    ap.add_argument("--work", type=Path, default=REPO / "scratch" / "gate_work")
    ap.add_argument("--mov", action="store_true", help="wrap the wav in a .mov first")
    ap.add_argument("--self-test", action="store_true",
                    help="disable resume on leg 3; the gate MUST report FAIL")
    ap.add_argument("--skip-stale", action="store_true")
    args = ap.parse_args()

    try:
        gate(args.media, args.work, args.mov, args.self_test, args.skip_stale)
    except GateFailure as exc:
        say(f"\nFAIL: {exc}")
        if args.self_test:
            say("\nSELF TEST PASSED: the gate detects a run that ignored its checkpoint.")
            return 0
        return 1
    if args.self_test:
        say("\nSELF TEST FAILED: the gate passed a run that ignored its checkpoint. "
            "It is not testing what it claims to test.")
        return 1
    say("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
