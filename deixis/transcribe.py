"""Transcribe a media file to a timestamped index, with live progress.

The transcript is deixis' index into the video, so this is the one step that
must not fail quietly. Progress is reported two ways at once: a live line on
the terminal, and a JSON status file that a detached run can be inspected
through. Background jobs are the normal case here -- an hour of audio is not
something you sit and watch -- and a run you cannot inspect is a run you cannot
trust.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MODEL",
    "CHUNK_S",
    "OVERLAP_S",
    "Progress",
    "render_bar",
    "transcribe",
    "main",
]

import json
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

# Imported as media_mod because the parameter it serves is named `media` and
# would shadow the module inside the function body.
from deixis import media as media_mod
from deixis.atomic import atomic_write_text

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# parakeet-mlx defaults chunk_duration to None, which feeds the whole file to
# Metal in one buffer. An hour of audio asks for ~14.5GB against a ~9.5GB max
# buffer and dies. Chunking is not optional at meeting length -- and it is also
# what makes chunk_callback fire, so progress reporting depends on it too.
CHUNK_S = 120.0
OVERLAP_S = 15.0


@dataclass
class Progress:
    audio_done_s: float
    audio_total_s: float
    elapsed_s: float
    # Audio already transcribed by an earlier run. Without it a resumed job
    # reports a fictional 300x, because it credits this run's clock with work a
    # previous one paid for -- and the ETA built on that speed is wrong by the
    # same factor, in the optimistic direction.
    resumed_from_s: float = 0.0

    @property
    def fraction(self) -> float:
        return self.audio_done_s / self.audio_total_s if self.audio_total_s else 0.0

    @property
    def speed(self) -> float:
        """Realtime multiple: seconds of audio per second of wall clock.

        Measured over this run's own work, so it stays a truthful estimate of
        what the remaining audio will cost.
        """
        done = self.audio_done_s - self.resumed_from_s
        return done / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def eta_s(self) -> float | None:
        if self.speed <= 0:
            return None
        return (self.audio_total_s - self.audio_done_s) / self.speed


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def render_bar(p: Progress, state: str, width: int = 24) -> str:
    filled = int(p.fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"{state:>10} [{bar}] {p.fraction * 100:3.0f}%  "
        f"{_clock(p.audio_done_s)}/{_clock(p.audio_total_s)} audio  "
        f"elapsed {_clock(p.elapsed_s)}  eta {_clock(p.eta_s)}  {p.speed:.1f}x"
    )


def _make_chunk_callback(
    rate: float,
    clock: Callable[[], float],
    emit: Callable[[Progress], None],
    resumed_from_s: float = 0.0,
) -> Callable[[float, float], None]:
    """Build the per-chunk progress callback.

    `current` and `full` arrive in SAMPLES, not seconds -- the sample counts
    parakeet-mlx passes its own callback, which deixis/chunking.py preserves.
    Upstream only ever feeds them to a ratio, so the units never mattered there.
    Dividing by `rate` is the whole point of this function, and a ratio-only
    assertion cannot see whether it happened, because the units cancel.
    """

    def chunk_callback(current: float, full: float) -> None:
        emit(Progress(current / rate, full / rate, clock(), resumed_from_s))

    return chunk_callback


def _with_speaker(sentence: dict, speaker: int) -> dict:
    """The same sentence with `speaker` inserted directly after `end`.

    Rebuilt rather than assigned into so the label reads before the token list
    instead of after it -- a sentence's tokens are most of its bytes, and a
    human scrolling the file should not have to cross them to find who spoke.
    Anchored on an existing key rather than on a literal key list, so adding a
    field to the payload above does not silently drop it here.
    """
    out: dict = {}
    for key, value in sentence.items():
        out[key] = value
        if key == "end":
            out["speaker"] = speaker
    return out


def _with_speakers(
    payload: dict, sentences: list[dict], labels: list[str], provenance: str
) -> dict:
    """The same payload, labelled, with the new keys up near the top.

    `speakers` is the legend for every `speaker` index below it and is read
    once; behind half a megabyte of sentences it would be the last thing an
    agent reaching the end of the file finds it needed at the start.
    """
    out: dict = {}
    for key, value in payload.items():
        out[key] = sentences if key == "sentences" else value
        if key == "model":
            out["speakers"] = labels
            # Its ABSENCE is load-bearing: without it "diarization was not run"
            # and "diarization ran and found one speaker" are the same document.
            out["diarization"] = provenance
    return out


def _label_speakers(
    payload: dict,
    audio: Path,
    out: Path,
    total_s: float,
    report: Callable[[Progress, str], None],
    require: bool,
) -> dict:
    """Diarize `audio` and rewrite `out` labelled, or leave both untouched.

    Called only after the unlabelled transcript is already on disk, which is
    what makes optionality structural rather than a matter of catching the
    right exceptions: every way this can fail leaves the correct, complete,
    unlabelled output exactly where it was.

    Returns the payload to hand back to the caller -- the labelled one when the
    pass ran, the one passed in when it did not.
    """
    from deixis import diarize as diarize_mod
    from deixis.merge import label_sentences

    started = time.monotonic()
    # Two events, not a bar. senko exposes no per-chunk callback, so there is
    # no intermediate progress to report and inventing some would be a lie.
    report(Progress(0.0, total_s, 0.0), "diarizing")
    try:
        result = diarize_mod.speaker_turns(audio)
    except diarize_mod.DiarizationUnavailable as exc:
        if require:
            raise
        # Named on stderr rather than swallowed: the transcript is correct, but
        # a user who asked for speaker labels and silently got none would have
        # no way to tell that from a recording with one speaker.
        print(f"diarization skipped: {exc}", file=sys.stderr)
        return payload

    speakers = label_sentences(payload["sentences"], result.turns)
    labelled = _with_speakers(
        payload,
        [_with_speaker(s, k) for s, k in zip(payload["sentences"], speakers)],
        result.labels,
        result.provenance,
    )
    # The second atomic write to the same path. The extra ~500KB buys there
    # being no instant in which `out` is absent or partial; holding the payload
    # to write once would put the whole ASR run behind this optional pass.
    atomic_write_text(out, json.dumps(labelled))
    report(Progress(total_s, total_s, time.monotonic() - started), "diarizing")
    return labelled


def transcribe(
    media: Path,
    out: Path,
    model_id: str = DEFAULT_MODEL,
    status_path: Path | None = None,
    on_progress: Callable[[Progress, str], None] | None = None,
    resume: bool = True,
    diarize: bool = True,
    require_diarize: bool = False,
) -> dict:
    """Transcribe `media`, writing a sentence+token timestamped JSON to `out`.

    `media` is any file ffmpeg can open -- the .mov straight off the screen
    recorder is the normal case. Audio is extracted to a temp file first,
    unless the input is already in the shape the model consumes.

    Returns the parsed result. `status_path` receives a JSON heartbeat during
    both phases so a detached run stays observable.

    An interrupted run leaves a checkpoint beside `out`, and the next run
    continues from its last completed chunk. `resume=False` ignores and removes
    any checkpoint and transcribes the whole file.

    Sentences are then labelled with who spoke them, which is a pass over an
    output that is already correct without it: any failure degrades to the
    unlabelled transcript and returns normally. `diarize=False` skips it and
    emits the unlabelled schema exactly; `require_diarize=True` makes a failure
    fatal for a caller who would rather have nothing than an unlabelled index.
    """
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import load_audio

    from deixis.checkpoint import (
        checkpoint_path_for,
        fingerprint,
        read_checkpoint,
        write_checkpoint,
    )
    from deixis.chunking import transcribe_chunked

    started = time.monotonic()
    # Load first: the extraction target rate is a property of this model's
    # preprocessor, not a constant we get to assume.
    model = from_pretrained(model_id)
    rate = model.preprocessor_config.sample_rate

    def write_status(p: Progress, state: str) -> None:
        if status_path is None:
            return
        payload = asdict(p) | {
            "state": state,
            "fraction": round(p.fraction, 4),
            "speed": round(p.speed, 2),
            "eta_s": p.eta_s,
        }
        # Not fsynced: a reader is protected by the rename alone, and a
        # heartbeat lost to a power cut costs nothing to regenerate.
        atomic_write_text(status_path, json.dumps(payload))

    def report(p: Progress, state: str) -> None:
        write_status(p, state)
        if on_progress:
            on_progress(p, state)

    stream = media_mod.probe(media)

    with tempfile.TemporaryDirectory(prefix="deixis-") as tmp:
        if media_mod.needs_conversion(stream, rate):
            # Per-phase clock. Extraction runs three orders of magnitude faster
            # than realtime and the transcription that follows around 20x;
            # sharing one elapsed would make both speeds meaningless.
            extract_started = time.monotonic()

            def on_extract(done_s: float) -> None:
                report(
                    Progress(done_s, stream.duration_s, time.monotonic() - extract_started),
                    "extracting",
                )

            audio = media_mod.extract_audio(
                media, Path(tmp) / "audio.wav", rate, on_progress=on_extract
            )
        else:
            audio = media

        audio_data = load_audio(audio, rate)

        ckpt_path = checkpoint_path_for(out)
        # Fingerprinted on `media`, never on `audio`: for a .mov those differ,
        # and `audio` is a temp wav with a fresh path and mtime on every run, so
        # a checkpoint keyed to it could never match a second time.
        fp = fingerprint(media, len(audio_data), model_id, CHUNK_S, OVERLAP_S)

        start_tokens: list = []
        skip_before = 0
        if resume:
            found = read_checkpoint(ckpt_path, fp)
            if found is not None:
                skip_before, start_tokens = found
                print(
                    f"resuming from {_clock(skip_before / rate)} "
                    f"({len(start_tokens)} tokens banked)",
                    file=sys.stderr,
                )
        else:
            ckpt_path.unlink(missing_ok=True)

        resumed_from_s = skip_before / rate

        # Per-phase clock, as above: transcription runs at a different order of
        # magnitude from extraction, so they cannot share an elapsed.
        transcribe_started = time.monotonic()
        chunk_callback = _make_chunk_callback(
            rate,
            lambda: time.monotonic() - transcribe_started,
            lambda p: report(p, "running"),
            resumed_from_s,
        )

        def on_chunk(done_through: int, next_start: int, total: int, merged: list) -> None:
            # Checkpointed with next_start, never done_through: chunks overlap,
            # so a chunk's end is past the following chunk's start and resuming
            # from it would skip a whole chunk of audio.
            #
            # Banked before it is reported, so an observer that sees 40% can
            # never be ahead of what a restart could actually recover.
            write_checkpoint(ckpt_path, fp, next_start, merged)
            chunk_callback(done_through, total)

        result = transcribe_chunked(
            model,
            audio_data,
            chunk_s=CHUNK_S,
            overlap_s=OVERLAP_S,
            start_tokens=start_tokens,
            skip_before=skip_before,
            on_chunk=on_chunk,
        )

        payload = {
            # The source the user handed us, never the temp wav -- this JSON is
            # an index into that file and has to keep pointing at it.
            "audio": str(media),
            "model": model_id,
            "text": result.text,
            "sentences": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "tokens": [{"t": t.start, "w": t.text} for t in s.tokens],
                }
                for s in result.sentences
            ],
        }
        # Atomic for the same reason as the heartbeat, and more: `out` is what
        # every downstream tool reads, and a truncated transcript does not
        # announce itself -- it merely looks short.
        atomic_write_text(out, json.dumps(payload))

        # The transcript is on disk, so the checkpoint has nothing left to
        # protect. Removed after the write, not before: a crash between the two
        # costs one redundant resume rather than the whole run.
        ckpt_path.unlink(missing_ok=True)

        # After the unlink, not before: the checkpoint protects ASR work, that
        # work is banked the moment the transcript is on disk, and a
        # diarization crash holding a stale checkpoint open would make the next
        # run resume audio it has already transcribed.
        #
        # `audio` and not `media`: media.py has already produced the 16 kHz
        # mono pcm_s16le wav senko wants, and it only exists until this `with`
        # block ends. A .mov handed straight to the diarizer is a second
        # normalization path to keep correct.
        if diarize:
            payload = _label_speakers(
                payload, audio, out, stream.duration_s, report, require_diarize
            )

        elapsed = time.monotonic() - started
        total = result.sentences[-1].end if result.sentences else 0.0
        report(Progress(total, total, elapsed, resumed_from_s), "done")
        return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe media to a timestamped index.")
    ap.add_argument("media", type=Path, help="video or audio file; a .mov is the normal case")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--status", type=Path, help="JSON heartbeat file for detached runs")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any checkpoint from an interrupted run and start over",
    )
    ap.add_argument(
        "--no-diarize",
        action="store_true",
        help="skip speaker labelling; emit the transcript alone",
    )
    ap.add_argument(
        "--require-diarize",
        action="store_true",
        help="fail the run if speaker labelling cannot run, instead of degrading",
    )
    args = ap.parse_args(argv)

    # \r only rewrites the line on a terminal; when piped, print one line per
    # chunk instead so a captured log stays readable rather than becoming one
    # enormous line of control characters.
    tty = sys.stderr.isatty()
    last_state = ""

    def show(p: Progress, state: str) -> None:
        # A phase change ends the rewritten line, so the finished extraction bar
        # stays on screen instead of being overwritten by transcription's 0%.
        nonlocal last_state
        if tty and last_state and state != last_state:
            print(file=sys.stderr)
        last_state = state
        end = "\r" if tty else "\n"
        print(render_bar(p, state), end=end, file=sys.stderr, flush=True)

    started = time.monotonic()
    try:
        result = transcribe(
            args.media,
            args.out,
            args.model,
            status_path=args.status,
            on_progress=show,
            resume=not args.no_resume,
            diarize=not args.no_diarize,
            require_diarize=args.require_diarize,
        )
    except Exception as exc:
        # Record and re-raise: a detached watcher polling the heartbeat has no
        # other way to distinguish "died" from "not started yet". The traceback
        # still reaches the terminal untouched.
        if args.status:
            # Atomic for the same reason as the heartbeat, and more so: the
            # watcher polling for exactly this document is in a tight read loop,
            # which makes it the reader most likely to land inside a torn write.
            atomic_write_text(
                args.status,
                json.dumps({"state": "failed", "error": f"{type(exc).__name__}: {exc}"}),
            )
        raise
    if tty:
        print(file=sys.stderr)
    elapsed = time.monotonic() - started
    total = result["sentences"][-1]["end"] if result["sentences"] else 0.0
    # No realtime multiple here any more. `total / elapsed` describes this run's
    # wall clock against the WHOLE file's duration, which on a resumed run
    # credits this process with work a previous one paid for -- an hour of audio
    # finished in the last two minutes reads as 30x. The live bar still reports
    # speed, correctly, because Progress.resumed_from_s lets it subtract the
    # banked audio; the summary has no such number to hand. Two plain durations
    # the reader can divide themselves beat one confident wrong figure.
    # A count, not a rate. The only number the summary can add here without
    # reintroducing the realtime multiple removed above, and it is guarded on
    # the key because a degraded run has no speakers to count.
    speakers = f" · {len(result['speakers'])} speakers" if "speakers" in result else ""
    print(
        f"done: {_clock(total)} audio in {_clock(elapsed)}{speakers} -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
