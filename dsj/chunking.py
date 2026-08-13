"""The resumable chunk loop, generic over any engine that decodes statelessly.

Born as a re-drive of parakeet-mlx's own loop (its transcribe() accumulates
merged tokens in a function-local that can be neither seeded nor observed, so
resume required reproducing it). Since the extraction it is engine-free: the
loop's real job -- boundaries, offsets, overlap merging, checkpoint hooks --
was never speech recognition, and any ChunkEngine (dsj.asr) can drive it. The
engine's contract is one method: decode a slice of samples into dsj tokens
timed from the slice's own start.

The overlap merge is dsj's own (dsj.alignment, vendored verbatim from
parakeet-mlx 0.5.2). The equivalence test in tests/test_chunking.py proves the
whole loop faithful against upstream's transcribe() while parakeet-mlx is
installed; tests/test_alignment.py exercises the merge math directly with no
model at all.
"""

from __future__ import annotations

__all__ = [
    "chunk_starts",
    "transcribe_chunked",
]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dsj.alignment import (
    AlignedResult,
    AlignedToken,
    SentenceConfig,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)

if TYPE_CHECKING:
    from dsj.asr import ChunkEngine


def chunk_starts(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[int]:
    """Sample offsets each chunk begins at.

    A pure function of three numbers, which is exactly why resume can be exact:
    a restarted run lands on the same boundaries with no memory of the first.
    """
    return list(range(0, total_samples, chunk_samples - overlap_samples))


def transcribe_chunked(
    engine: ChunkEngine,
    audio_data: Any,
    *,
    chunk_s: float,
    overlap_s: float,
    start_tokens: list[AlignedToken] | None = None,
    skip_before: int = 0,
    on_chunk: Callable[[int, int, int, list[AlignedToken]], None] | None = None,
    sentence: SentenceConfig | None = None,
) -> AlignedResult:
    """Transcribe already-loaded audio, chunk by chunk, resumably.

    `start_tokens` and `skip_before` come from a checkpoint: the tokens merged
    so far, and the first chunk *start* not yet accounted for.

    `on_chunk(done_through, next_start, total, merged)` fires after each chunk
    merges -- after, not before, so the number it reports is work that actually
    happened. It reports `next_start` separately from `done_through` because
    chunks overlap: a chunk's end is past the following chunk's start, so
    resuming from an end would skip one whole chunk.
    """
    rate = engine.sample_rate

    # Computed exactly as parakeet-mlx does, so int() truncation lands the same
    # way on geometries that are not a whole number of samples.
    chunk_samples = int(chunk_s * rate)
    overlap_samples = int(overlap_s * rate)

    total = len(audio_data)
    all_tokens: list[AlignedToken] = list(start_tokens or [])
    starts = chunk_starts(total, chunk_samples, overlap_samples)

    for i, start in enumerate(starts):
        end = min(start + chunk_samples, total)

        if end - start < engine.min_chunk_samples:
            break  # the engine's guard against a zero-length feature window

        if start < skip_before:
            continue  # already merged into start_tokens by an earlier run

        # A ChunkEngine decodes each chunk from a fresh decoder state -- no
        # hidden state crosses this call -- so a chunk's tokens depend on
        # nothing but its own audio. That is what makes resuming mid-file give
        # the same answer as never having stopped, and it is the property the
        # protocol's docstring demands of every implementation.
        #
        # The engine times tokens from the chunk's own start; the offset is
        # applied here, in construction. Same float arithmetic as the original
        # in-place walk: __post_init__ computes end = (start + offset) +
        # duration, which is what `token.start += offset; token.end =
        # token.start + duration` did. The golden checkpoint holds this to the
        # byte.
        offset = start / rate
        chunk_tokens = [
            AlignedToken(
                id=token.id,
                text=token.text,
                start=token.start + offset,
                duration=token.duration,
                confidence=token.confidence,
            )
            for token in engine.decode(audio_data[start:end])
        ]

        if all_tokens:
            try:
                all_tokens = merge_longest_contiguous(
                    all_tokens, chunk_tokens, overlap_duration=overlap_s
                )
            except RuntimeError:
                all_tokens = merge_longest_common_subsequence(
                    all_tokens, chunk_tokens, overlap_duration=overlap_s
                )
        else:
            all_tokens = chunk_tokens

        if on_chunk is not None:
            # The following boundary, or the end of the audio if this was the
            # last chunk -- never this chunk's own end, which overlaps it.
            next_start = starts[i + 1] if i + 1 < len(starts) else total
            on_chunk(end, next_start, total, all_tokens)

    return sentences_to_result(tokens_to_sentences(all_tokens, sentence))
