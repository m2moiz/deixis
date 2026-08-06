"""End to end: kill a run, start it again, get the same transcript.

Marked slow because it loads the ASR model. Registered but not deselected, like
the rest of the slow tests here -- a test carved out of the normal invocation is
a test nobody observes.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from deixis.checkpoint import checkpoint_path_for

if TYPE_CHECKING:
    from parakeet_mlx import BaseParakeet
    from parakeet_mlx.alignment import AlignedResult

    from deixis.transcribe import Progress

pytestmark = pytest.mark.slow


def test_a_killed_run_resumes_and_matches(chunked_audio_path: Path, tmp_path: Path) -> None:
    from deixis.transcribe import transcribe

    reference_out = tmp_path / "reference.json"
    # diarize=False: these are about ASR, extraction and resume. Running the
    # real diarizer here would add ~13s of CoreML model load per call and put
    # an unsupervised clustering result inside equality assertions that must
    # hold exactly. The pass has its own tests, and scratch/diarize_gate.py
    # runs it end to end on the real meeting.
    reference = transcribe(
        chunked_audio_path, reference_out, resume=False, diarize=False
    )

    out = tmp_path / "resumed.json"

    class Interrupt(Exception):
        pass

    seen = 0

    def die_after_two(p: Progress, state: str) -> None:
        nonlocal seen
        if state != "running":
            return
        seen += 1
        if seen == 2:
            raise Interrupt

    with pytest.raises(Interrupt):
        transcribe(chunked_audio_path, out, on_progress=die_after_two, diarize=False)

    ckpt = checkpoint_path_for(out)
    assert ckpt.exists(), "the interrupted run banked nothing"
    banked = json.loads(ckpt.read_text())
    assert banked["next_start"] > 0
    assert banked["tokens"]
    assert not out.exists(), "a partial transcript was written as if complete"

    resumed = transcribe(chunked_audio_path, out, diarize=False)

    assert resumed["sentences"] == reference["sentences"]
    assert resumed["text"] == reference["text"]
    assert json.loads(out.read_text()) == json.loads(reference_out.read_text())
    assert not ckpt.exists(), "the checkpoint outlived the run that completed"


def test_a_checkpoint_for_different_audio_is_ignored(
    chunked_audio_path: Path, tmp_path: Path
) -> None:
    from deixis.transcribe import transcribe

    out = tmp_path / "out.json"
    ckpt = checkpoint_path_for(out)
    ckpt.write_text(
        json.dumps(
            {
                "fingerprint": {
                    "schema": 1, "media": "/nowhere/else.wav", "media_size": 1,
                    "media_mtime_ns": 1, "total_samples": 1, "model_id": "x",
                    "parakeet_version": "0.0.0", "chunk_s": 1.0, "overlap_s": 1.0,
                },
                "next_start": 999_999_999,
                "tokens": [
                    {"id": 1, "text": " wrong", "start": 0.0,
                     "duration": 1.0, "confidence": 1.0}
                ],
            }
        )
    )

    result = transcribe(chunked_audio_path, out, diarize=False)
    assert "wrong" not in result["text"]
    assert result["sentences"], "the run produced nothing"


def test_no_resume_ignores_an_existing_checkpoint(
    chunked_audio_path: Path, tmp_path: Path
) -> None:
    from deixis.transcribe import transcribe

    out = tmp_path / "out.json"
    reference = transcribe(chunked_audio_path, out, resume=False, diarize=False)
    assert reference["sentences"]
    assert not checkpoint_path_for(out).exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_a_mov_resumes_even_though_its_audio_is_a_fresh_temp_wav_each_run(
    chunked_audio_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure a wav-only suite cannot see.

    A .mov is extracted to a new temp file on every run, so a fingerprint taken
    from the audio handed to the model never matches on the second run and the
    job silently restarts from zero -- while every test above stays green,
    because a conforming .wav IS the audio handed to the model.
    """
    from deixis.transcribe import transcribe

    mov = tmp_path / "recording.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10",
         "-i", str(chunked_audio_path), "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "48000", "-ac", "2", str(mov)],
        check=True,
    )

    out = tmp_path / "out.json"

    class Interrupt(Exception):
        pass

    seen = 0

    def die_after_two(p: Progress, state: str) -> None:
        nonlocal seen
        if state != "running":
            return
        seen += 1
        if seen == 2:
            raise Interrupt

    with pytest.raises(Interrupt):
        transcribe(mov, out, on_progress=die_after_two, diarize=False)

    ckpt = checkpoint_path_for(out)
    banked = json.loads(ckpt.read_text())
    assert banked["fingerprint"]["media"] == str(mov.resolve()), (
        "the checkpoint is keyed to the temp wav, so it can never match again"
    )
    first_next_start = banked["next_start"]
    assert first_next_start > 0

    # The second run extracts to a DIFFERENT temp wav. Resume must still fire,
    # and it must fire from the banked boundary rather than from zero.
    resumed_from: list[int] = []

    from deixis import chunking

    original = chunking.transcribe_chunked

    # audio_data and the kwargs are Any at the source too: the decoded samples
    # are an mlx array, a compiled extension with no stubs, so transcribe_chunked
    # itself types that parameter Any.
    def spy(model: BaseParakeet, audio_data: Any, **kwargs: Any) -> AlignedResult:
        skip_before: int = kwargs.get("skip_before", 0)
        resumed_from.append(skip_before)
        return original(model, audio_data, **kwargs)

    monkeypatch.setattr(chunking, "transcribe_chunked", spy)
    caplog.set_level(logging.INFO, logger="deixis.transcribe")
    transcribe(mov, out, diarize=False)

    assert resumed_from == [first_next_start], (
        f"resumed from {resumed_from}, expected the banked {first_next_start}"
    )
    # caplog, not capsys: the resume line is logger.info now, and only main()
    # attaches the stderr handler -- this test calls transcribe() in-process.
    # The assertion itself is load-bearing and stays: it is the only check that
    # the run announced a resume rather than silently starting over.
    assert "resuming from" in caplog.text
    assert not ckpt.exists()
    assert json.loads(out.read_text())["sentences"]
