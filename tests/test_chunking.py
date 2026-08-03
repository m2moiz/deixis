"""The chunk loop, re-driven so it can be resumed.

parakeet-mlx keeps `all_tokens` in a function-local and its chunk_callback
fires before the chunk is decoded (parakeet.py:185-186 precede :194), so there
is no way to seed or observe the accumulation from outside. The loop is
therefore ours; the merge is still theirs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from deixis.chunking import chunk_starts, transcribe_chunked
from deixis.transcribe import CHUNK_S, OVERLAP_S


def test_boundaries_match_the_librarys_stride() -> None:
    # parakeet.py:182 -- range(0, len(audio), chunk_samples - overlap_samples)
    rate = 16_000
    starts = chunk_starts(
        total_samples=360 * rate,
        chunk_samples=int(120.0 * rate),
        overlap_samples=int(15.0 * rate),
    )
    assert starts == [0, 1_680_000, 3_360_000, 5_040_000]
    assert [s / rate for s in starts] == [0.0, 105.0, 210.0, 315.0]


def test_a_file_shorter_than_one_chunk_still_yields_one_start() -> None:
    rate = 16_000
    assert chunk_starts(45 * rate, int(120.0 * rate), int(15.0 * rate)) == [0]


def test_a_chunk_end_is_past_the_following_chunk_start() -> None:
    # The trap this whole design works around: chunks overlap, so chunk 2 ends
    # at 3_600_000 while chunk 3 begins at 3_360_000. Resuming from an end
    # rather than a start would skip chunk 3 and silently drop 105s of audio.
    rate = 16_000
    chunk_samples = int(120.0 * rate)
    starts = chunk_starts(360 * rate, chunk_samples, int(15.0 * rate))

    second_end = starts[1] + chunk_samples
    assert second_end == 3_600_000
    assert starts[2] == 3_360_000
    assert second_end > starts[2]


def test_boundaries_are_independent_of_where_a_run_began() -> None:
    # This is what makes resume exact: a restarted run recomputes the same
    # boundaries from the same three numbers, with no memory of the first run.
    rate = 16_000
    args = (360 * rate, int(120.0 * rate), int(15.0 * rate))
    assert chunk_starts(*args) == chunk_starts(*args)


# --- Model-backed. These are the measurements the whole feature rests on. -----

@pytest.fixture(scope="module")
def model(model_id: str):
    from parakeet_mlx import from_pretrained

    return from_pretrained(model_id)


def _load(model, path: Path):
    from parakeet_mlx.audio import load_audio

    # No dtype argument, matching BaseParakeet.transcribe, which also takes
    # load_audio's mx.bfloat16 default (parakeet.py:166). Passing a different
    # one here would make the equivalence test compare two different decodes.
    return load_audio(path, model.preprocessor_config.sample_rate)


def _shape(result) -> list[dict]:
    """A comparable rendering: the same fields transcribe() writes to disk."""
    return [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "tokens": [dataclasses.asdict(t) for t in s.tokens],
        }
        for s in result.sentences
    ]


@pytest.mark.slow
def test_our_loop_reproduces_the_librarys_transcribe(model, chunked_audio_path) -> None:
    # The guard on the coupling described in chunking.py's docstring. If
    # upstream changes its stride, offsets, or merge order, this fails loudly
    # instead of the transcript quietly changing.
    theirs = model.transcribe(
        chunked_audio_path, chunk_duration=CHUNK_S, overlap_duration=OVERLAP_S
    )
    ours = transcribe_chunked(
        model, _load(model, chunked_audio_path), chunk_s=CHUNK_S, overlap_s=OVERLAP_S
    )

    assert ours.text == theirs.text
    assert _shape(ours) == _shape(theirs)


@pytest.mark.slow
def test_transcription_is_deterministic_across_runs(model, chunked_audio_path) -> None:
    # Resume can only be byte-identical if the underlying decode is. Measured
    # here separately so a failure is attributed to MLX/Metal rather than to
    # the resume logic.
    audio = _load(model, chunked_audio_path)
    a = transcribe_chunked(model, audio, chunk_s=CHUNK_S, overlap_s=OVERLAP_S)
    b = transcribe_chunked(model, audio, chunk_s=CHUNK_S, overlap_s=OVERLAP_S)

    assert a.text == b.text
    assert _shape(a) == _shape(b)


@pytest.mark.slow
def test_resuming_mid_file_gives_the_uninterrupted_result(model, chunked_audio_path) -> None:
    audio = _load(model, chunked_audio_path)
    uninterrupted = transcribe_chunked(model, audio, chunk_s=CHUNK_S, overlap_s=OVERLAP_S)

    class Interrupt(Exception):
        pass

    banked: list = []
    chunks_seen = 0
    next_start = 0

    def stop_after_two(done_through, following, total, merged):
        nonlocal chunks_seen, banked, next_start
        chunks_seen += 1
        banked = list(merged)
        next_start = following
        if chunks_seen == 2:
            raise Interrupt

    with pytest.raises(Interrupt):
        transcribe_chunked(
            model, audio, chunk_s=CHUNK_S, overlap_s=OVERLAP_S, on_chunk=stop_after_two
        )

    assert chunks_seen == 2
    assert banked, "nothing was banked before the interrupt"

    # The chunk reported as next must be the third boundary, not the second
    # chunk's end -- those differ by the overlap, and confusing them silently
    # drops a chunk.
    rate = model.preprocessor_config.sample_rate
    starts = chunk_starts(len(audio), int(CHUNK_S * rate), int(OVERLAP_S * rate))
    assert next_start == starts[2]

    resumed = transcribe_chunked(
        model,
        audio,
        chunk_s=CHUNK_S,
        overlap_s=OVERLAP_S,
        start_tokens=banked,
        skip_before=next_start,
    )

    assert resumed.text == uninterrupted.text
    assert _shape(resumed) == _shape(uninterrupted)
