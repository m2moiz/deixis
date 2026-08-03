"""The chunk loop, re-driven so it can be resumed.

parakeet-mlx keeps `all_tokens` in a function-local and its chunk_callback
fires before the chunk is decoded (parakeet.py:185-186 precede :194), so there
is no way to seed or observe the accumulation from outside. The loop is
therefore ours; the merge is still theirs.
"""

from __future__ import annotations

from deixis.chunking import chunk_starts


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
