"""Re-drive parakeet-mlx's chunk loop so a run can be resumed.

BaseParakeet.transcribe accumulates merged tokens in a function-local, and its
chunk_callback both fires before the chunk is decoded and receives only sample
counts. Neither seeding nor observing that accumulation is possible from
outside, so the loop is reproduced here.

The overlap merge is NOT reproduced. This module calls the library's own
merge_longest_contiguous and merge_longest_common_subsequence, in the same
order and with the same arguments, so the hard part stays upstream. What that
buys in correctness it costs in coupling: this loop mirrors parakeet-mlx 0.5.2
(parakeet.py:164-221) and will drift silently if upstream changes. The
equivalence test in tests/test_chunking.py is what makes that drift loud, and
pyproject pins the version rather than floating it.
"""

from __future__ import annotations

__all__ = [
    "chunk_starts",
    "transcribe_chunked",
]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from parakeet_mlx import DecodingConfig
from parakeet_mlx.alignment import (
    AlignedResult,
    AlignedToken,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)
from parakeet_mlx.audio import get_logmel

if TYPE_CHECKING:
    from parakeet_mlx import BaseParakeet


def chunk_starts(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[int]:
    """Sample offsets each chunk begins at.

    A pure function of three numbers, which is exactly why resume can be exact:
    a restarted run lands on the same boundaries with no memory of the first.
    """
    return list(range(0, total_samples, chunk_samples - overlap_samples))


def transcribe_chunked(
    model: BaseParakeet,
    audio_data: Any,
    *,
    chunk_s: float,
    overlap_s: float,
    start_tokens: list[AlignedToken] | None = None,
    skip_before: int = 0,
    on_chunk: Callable[[int, int, int, list[AlignedToken]], None] | None = None,
    decoding_config: DecodingConfig | None = None,
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
    cfg = decoding_config or DecodingConfig()
    rate = model.preprocessor_config.sample_rate

    # Computed exactly as the library does, so int() truncation lands the same
    # way on geometries that are not a whole number of samples.
    chunk_samples = int(chunk_s * rate)
    overlap_samples = int(overlap_s * rate)

    total = len(audio_data)
    all_tokens: list[AlignedToken] = list(start_tokens or [])
    starts = chunk_starts(total, chunk_samples, overlap_samples)

    for i, start in enumerate(starts):
        end = min(start + chunk_samples, total)

        if end - start < model.preprocessor_config.hop_length:
            break  # upstream's guard against a zero-length log mel

        if start < skip_before:
            continue  # already merged into start_tokens by an earlier run

        chunk_mel = get_logmel(audio_data[start:end], model.preprocessor_config)
        chunk_result = model.generate(chunk_mel, decoding_config=cfg)[0]

        # generate() decodes each chunk from a fresh decoder state -- it passes
        # neither last_token nor hidden_state -- so a chunk's tokens depend on
        # nothing but its own audio. That is what makes resuming mid-file give
        # the same answer as never having stopped.
        offset = start / rate
        for sentence in chunk_result.sentences:
            for token in sentence.tokens:
                token.start += offset
                token.end = token.start + token.duration

        if all_tokens:
            try:
                all_tokens = merge_longest_contiguous(
                    all_tokens, chunk_result.tokens, overlap_duration=overlap_s
                )
            except RuntimeError:
                all_tokens = merge_longest_common_subsequence(
                    all_tokens, chunk_result.tokens, overlap_duration=overlap_s
                )
        else:
            all_tokens = chunk_result.tokens

        if on_chunk is not None:
            # The following boundary, or the end of the audio if this was the
            # last chunk -- never this chunk's own end, which overlaps it.
            next_start = starts[i + 1] if i + 1 < len(starts) else total
            on_chunk(end, next_start, total, all_tokens)

    return sentences_to_result(tokens_to_sentences(all_tokens, cfg.sentence))
