"""Transcribe a media file to a timestamped index, with live progress.

The transcript is dsj' index into the video, so this is the one step that
must not fail quietly. Progress is reported two ways at once: a live line on
the terminal, and a JSON status file that a detached run can be inspected
through. Background jobs are the normal case here -- an hour of audio is not
something you sit and watch -- and a run you cannot inspect is a run you cannot
trust.
"""

from __future__ import annotations

__all__ = [
    "CHUNK_S",
    "DEFAULT_MODEL",
    "DEFAULT_WHISPER_MODEL",
    "ENGINES",
    "OVERLAP_S",
    "Progress",
    "clock",
    "main",
    "render_bar",
    "transcribe",
]

import json
import logging
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Imported as media_mod because the parameter it serves is named `media` and
# would shadow the module inside the function body.
from dsj import media as media_mod
from dsj.asr import ENGINES, Transcription, get_engine
from dsj.atomic import atomic_write_text

# Names only -- both engine modules keep their backends lazy, so pulling
# these in costs nothing to a run that never asks for the engine, and it
# keeps `--help` able to print the defaults.
from dsj.parakeet import DEFAULT_MODEL
from dsj.whisper import DEFAULT_WHISPER_MODEL
from dsj.whisper import SAMPLE_RATE as WHISPER_SAMPLE_RATE

if TYPE_CHECKING:
    from dsj.alignment import AlignedToken

# The transcript is JSON, so its two nested shapes are plain dicts rather than
# dataclasses -- json.dumps is the only consumer here, and a schema class would
# have to be flattened right back. Naming them keeps the signatures below
# honest about which dict is which without pretending to more structure than
# the on-disk document has.
type Sentence = dict[str, Any]
type Payload = dict[str, Any]

# Module logger, not print: transcribe() is the import surface the frame-
# retrieval half will call, and a library that writes to stderr unasked is a
# library that cannot be embedded. main() attaches the stderr handler, so the
# CLI behaves exactly as before.
logger = logging.getLogger("dsj.suno")

# parakeet-mlx defaults chunk_duration to None, which feeds the whole file to
# Metal in one buffer. An hour of audio asks for ~14.5GB against a ~9.5GB max
# buffer and dies. Chunking is not optional at meeting length -- and it is also
# what makes chunk_callback fire, so progress reporting depends on it too.
CHUNK_S = 120.0
OVERLAP_S = 15.0


@dataclass
class Progress:
    """One sample of how far a transcription has got, and how fast."""

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
        """Share of the audio transcribed so far, 0.0 when the total is unknown."""
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
        """Seconds of wall clock left at the current speed, or None if not moving yet."""
        if self.speed <= 0:
            return None
        return (self.audio_total_s - self.audio_done_s) / self.speed


def clock(seconds: float | None) -> str:
    """Render a duration as m:ss, or h:mm:ss once it passes an hour.

    None is "--:--" and NOT "0:00": a run whose total duration ffprobe could
    not determine has no elapsed time to show, and printing zero would claim it
    finished instantly.
    """
    if seconds is None:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def render_bar(p: Progress, state: str, width: int = 24) -> str:
    """Render one progress line: phase, bar, audio clocks, speed and ETA."""
    filled = int(p.fraction * width)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"{state:>10} [{bar}] {p.fraction * 100:3.0f}%  "
        f"{clock(p.audio_done_s)}/{clock(p.audio_total_s)} audio  "
        f"elapsed {clock(p.elapsed_s)}  eta {clock(p.eta_s)}  {p.speed:.1f}x"
    )


def _make_chunk_callback(
    rate: float,
    clock: Callable[[], float],
    emit: Callable[[Progress], None],
    resumed_from_s: float = 0.0,
) -> Callable[[float, float], None]:
    """Build the per-chunk progress callback.

    `current` and `full` arrive in SAMPLES, not seconds -- the sample counts
    parakeet-mlx passes its own callback, which dsj/chunking.py preserves.
    Upstream only ever feeds them to a ratio, so the units never mattered there.
    Dividing by `rate` is the whole point of this function, and a ratio-only
    assertion cannot see whether it happened, because the units cancel.
    """

    def chunk_callback(current: float, full: float) -> None:
        emit(Progress(current / rate, full / rate, clock(), resumed_from_s))

    return chunk_callback


def _with_speaker(sentence: Sentence, speaker: int) -> Sentence:
    """The same sentence with `speaker` inserted directly after `end`.

    Rebuilt rather than assigned into so the label reads before the token list
    instead of after it -- a sentence's tokens are most of its bytes, and a
    human scrolling the file should not have to cross them to find who spoke.
    Anchored on an existing key rather than on a literal key list, so adding a
    field to the payload above does not silently drop it here.
    """
    out: Sentence = {}
    for key, value in sentence.items():
        out[key] = value
        if key == "end":
            out["speaker"] = speaker
    return out


def _with_speakers(
    payload: Payload, sentences: list[Sentence], labels: list[str], provenance: str
) -> Payload:
    """The same payload, labelled, with the new keys up near the top.

    `speakers` is the legend for every `speaker` index below it and is read
    once; behind half a megabyte of sentences it would be the last thing an
    agent reaching the end of the file finds it needed at the start.
    """
    out: Payload = {}
    for key, value in payload.items():
        out[key] = sentences if key == "sentences" else value
        if key == "model":
            out["speakers"] = labels
            # Its ABSENCE is load-bearing: without it "diarization was not run"
            # and "diarization ran and found one speaker" are the same document.
            out["diarization"] = provenance
    return out


def _label_speakers(
    payload: Payload,
    audio: Path,
    out: Path,
    total_s: float,
    report: Callable[[Progress, str], None],
    require: bool,
) -> Payload:
    """Diarize `audio` and rewrite `out` labelled, or leave both untouched.

    Called only after the unlabelled transcript is already on disk, which is
    what makes optionality structural rather than a matter of catching the
    right exceptions: every way this can fail leaves the correct, complete,
    unlabelled output exactly where it was.

    Returns the payload to hand back to the caller -- the labelled one when the
    pass ran, the one passed in when it did not.
    """
    from dsj import diarize as diarize_mod
    from dsj.merge import label_sentences

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
        logger.warning("diarization skipped: %s", exc)
        return payload

    speakers = label_sentences(payload["sentences"], result.turns)
    labelled = _with_speakers(
        payload,
        # strict=True: a length mismatch between sentences and their labels is a
        # merge bug, and silently truncating to the shorter one would hide it.
        [_with_speaker(s, k) for s, k in zip(payload["sentences"], speakers, strict=True)],
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
    model_id: str | None = None,
    status_path: Path | None = None,
    on_progress: Callable[[Progress, str], None] | None = None,
    resume: bool = True,
    diarize: bool = True,
    require_diarize: bool = False,
    engine: str = "parakeet",
    language: str | None = None,
    prompt: str | None = None,
) -> Payload:
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

    `engine` picks the ASR backend. "parakeet" is the default and everything
    above describes it. "whisper" exists for the languages parakeet does not
    have -- see dsj/whisper.py -- and differs in two ways worth knowing
    before you choose it: it owns its own window loop, so there is no
    checkpoint and no resume, and it reports no progress between start and
    finish. `language` and `prompt` are whisper's; parakeet takes neither.
    `model_id` defaults to whichever engine's model, so it is usually left
    alone.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}, expected one of {', '.join(ENGINES)}")
    if engine == "parakeet" and (language is not None or prompt is not None):
        raise ValueError(
            "--language and --prompt are whisper's; parakeet takes neither. "
            "Add --engine whisper, or drop them."
        )
    # Resolves the engine module and raises EngineUnavailable with the remedy
    # if its backend cannot import here. After this call, everything
    # engine-specific is an attribute of `eng_mod` -- this function never
    # imports a backend itself, which is what the AST boundary test enforces.
    spec, eng_mod = get_engine(engine)

    from dsj.checkpoint import (
        checkpoint_path_for,
        fingerprint,
        read_checkpoint,
        write_checkpoint,
    )
    from dsj.chunking import transcribe_chunked

    started = time.monotonic()
    # Load first: the extraction target rate is a property of the loaded
    # engine, not a constant we get to assume. The whisper engine skips the
    # load entirely -- its rate is fixed and its model loads inside its own
    # call, so an engine the user did not ask for never costs a model load.
    #
    # `spec.kind` IS the engine test from here down, read once: the chunk
    # branch below cannot run without `loaded`, and reading the engine string
    # a second time would let the two disagree.
    loaded: Any = None
    if spec.kind == "file":
        model_id = model_id or DEFAULT_WHISPER_MODEL
        rate = WHISPER_SAMPLE_RATE
    else:
        model_id = model_id or cast("str", eng_mod.DEFAULT_MODEL)
        loaded = eng_mod.load(model_id)
        rate = int(loaded.sample_rate)

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

    with tempfile.TemporaryDirectory(prefix="dsj-") as tmp:
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

        ckpt_path: Path | None = None
        resumed_from_s = 0.0
        if spec.kind == "file":
            from dsj.whisper import transcribe_whisper

            # No checkpoint, and so no resume: whisper owns its window loop and
            # exposes no per-window hook to bank one from. Said out loud rather
            # than left as a silently absent feature, because a resumable engine
            # and a non-resumable one look identical until the run is killed.
            if resume:
                logger.info("whisper writes no checkpoint; an interrupted run starts over")
            # One report, at 0%, then nothing until the end. mlx-whisper takes
            # no progress callback, and a bar that moved without evidence would
            # be a bar that lies.
            report(Progress(0.0, stream.duration_s, 0.0), "running")
            transcription = transcribe_whisper(
                audio, model_id=model_id, language=language, prompt=prompt
            )
        else:
            audio_data = loaded.load_audio(audio)

            ckpt_path = checkpoint_path_for(out)
            # Fingerprinted on `media`, never on `audio`: for a .mov those
            # differ, and `audio` is a temp wav with a fresh path and mtime on
            # every run, so a checkpoint keyed to it could never match a second
            # time.
            fp = fingerprint(
                media, len(audio_data), model_id, CHUNK_S, OVERLAP_S,
                engine_fields=cast("dict[str, str]", eng_mod.fingerprint_fields()),
            )

            start_tokens: list[AlignedToken] = []
            skip_before = 0
            if resume:
                found = read_checkpoint(ckpt_path, fp)
                if found is not None:
                    skip_before, start_tokens = found
                    logger.info(
                        "resuming from %s (%d tokens banked)",
                        clock(skip_before / rate),
                        len(start_tokens),
                    )
            else:
                ckpt_path.unlink(missing_ok=True)

            resumed_from_s = skip_before / rate

            # Per-phase clock, as above: transcription runs at a different order
            # of magnitude from extraction, so they cannot share an elapsed.
            transcribe_started = time.monotonic()
            chunk_callback = _make_chunk_callback(
                rate,
                lambda: time.monotonic() - transcribe_started,
                lambda p: report(p, "running"),
                resumed_from_s,
            )
            banked = ckpt_path

            def on_chunk(
                done_through: int, next_start: int, total: int, merged: list[AlignedToken]
            ) -> None:
                # Checkpointed with next_start, never done_through: chunks
                # overlap, so a chunk's end is past the following chunk's start
                # and resuming from it would skip a whole chunk of audio.
                #
                # Banked before it is reported, so an observer that sees 40% can
                # never be ahead of what a restart could actually recover.
                write_checkpoint(banked, fp, next_start, merged)
                chunk_callback(done_through, total)

            result = transcribe_chunked(
                loaded,
                audio_data,
                chunk_s=CHUNK_S,
                overlap_s=OVERLAP_S,
                start_tokens=start_tokens,
                skip_before=skip_before,
                on_chunk=on_chunk,
            )
            transcription = Transcription(
                text=result.text,
                sentences=[
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "tokens": [{"t": t.start, "w": t.text} for t in s.tokens],
                    }
                    for s in result.sentences
                ],
            )

        payload: Payload = {
            # The source the user handed us, never the temp wav -- this JSON is
            # an index into that file and has to keep pointing at it.
            "audio": str(media),
            # No separate engine field: the model id already names it, and
            # tests/test_suno.py pins this key set precisely so a downstream
            # reader can rely on it.
            "model": model_id,
            "text": transcription.text,
            "sentences": transcription.sentences,
        }
        # Atomic for the same reason as the heartbeat, and more: `out` is what
        # every downstream tool reads, and a truncated transcript does not
        # announce itself -- it merely looks short.
        atomic_write_text(out, json.dumps(payload))

        # The transcript is on disk, so the checkpoint has nothing left to
        # protect. Removed after the write, not before: a crash between the two
        # costs one redundant resume rather than the whole run.
        if ckpt_path is not None:
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
        total = transcription.sentences[-1]["end"] if transcription.sentences else 0.0
        report(Progress(total, total, elapsed, resumed_from_s), "done")
        return payload


def main(argv: list[str] | None = None) -> int:
    """Run the transcribe CLI.

    A shim onto `dsj.cli`, which owns every flag in the project so that the
    console script and `python -m dsj.suno` cannot drift apart. Kept
    as a function returning an int because that is the contract its tests --
    and any embedder -- already rely on.

    Returns:
        A process exit code.
    """
    from dsj.cli import run

    return run(["suno", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
