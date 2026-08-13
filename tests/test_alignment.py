"""Direct tests for dsj.alignment -- no model, no slow marker.

Until the extraction, the merge and sentence functions were only exercised
through the real parakeet model in slow-marked tests, so the fast lane said
nothing about them. These run on synthetic tokens and pin the behaviours the
rest of dsj is built on: the end recompute, the punctuation split, the merge's
dedup-by-id, its midpoint fallback, and the RuntimeError that chunking.py uses
as control flow.
"""

from __future__ import annotations

import math

import pytest

from dsj.alignment import (
    AlignedSentence,
    AlignedToken,
    SentenceConfig,
    merge_longest_common_subsequence,
    merge_longest_contiguous,
    sentences_to_result,
    tokens_to_sentences,
)


def tok(id_: int, text: str, start: float, duration: float = 0.2) -> AlignedToken:
    return AlignedToken(id=id_, text=text, start=start, duration=duration)


# ---------------------------------------------------------------- dataclasses


def test_token_end_is_recomputed_from_start_plus_duration() -> None:
    t = AlignedToken(id=1, text=" x", start=119.96, duration=0.301, end=999.0)
    # Whatever the caller passes for `end` is overwritten -- it is derived
    # state, and the checkpoint file relies on exactly this arithmetic.
    assert t.end == 120.261


def test_sentence_sorts_tokens_and_aggregates() -> None:
    late = tok(2, " world", 1.0)
    early = tok(1, " hello", 0.0)
    s = AlignedSentence(text=" hello world", tokens=[late, early])
    assert [t.id for t in s.tokens] == [1, 2]
    assert s.start == 0.0
    assert s.end == late.end
    assert s.duration == s.end - s.start


def test_sentence_confidence_is_geometric_mean() -> None:
    a = AlignedToken(id=1, text=" a", start=0.0, duration=0.1, confidence=0.5)
    b = AlignedToken(id=2, text=" b", start=0.2, duration=0.1, confidence=0.5)
    s = AlignedSentence(text=" a b", tokens=[a, b])
    assert s.confidence == pytest.approx(0.5, abs=1e-6)
    # Not the arithmetic mean: one zero-ish token should crater the aggregate.
    c = AlignedToken(id=3, text=" c", start=0.4, duration=0.1, confidence=1e-9)
    s2 = AlignedSentence(text=" a c", tokens=[a, c])
    assert s2.confidence < 0.01
    assert not math.isnan(s2.confidence)


def test_result_strips_text_and_flattens_tokens() -> None:
    s1 = AlignedSentence(text=" one.", tokens=[tok(1, " one.", 0.0)])
    s2 = AlignedSentence(text=" two.", tokens=[tok(2, " two.", 1.0)])
    r = sentences_to_result([s1, s2])
    assert r.text == "one. two."
    assert [t.id for t in r.tokens] == [1, 2]


# ---------------------------------------------------------- sentence assembly


def test_splits_on_sentence_final_period() -> None:
    tokens = [
        tok(1, " it", 0.0),
        tok(2, " works.", 0.3),
        tok(3, " next", 0.9),
        tok(4, " one", 1.2),
    ]
    sentences = tokens_to_sentences(tokens)
    assert [s.text for s in sentences] == [" it works.", " next one"]


def test_period_inside_a_word_does_not_split() -> None:
    # "v0.5" -- the period is mid-token and the following token has no leading
    # space, which is the upstream heuristic for "not a sentence end".
    tokens = [tok(1, " v0.", 0.0), tok(2, "5", 0.3), tok(3, " ships.", 0.6)]
    sentences = tokens_to_sentences(tokens)
    assert len(sentences) == 1
    assert sentences[0].text == " v0.5 ships."


def test_silence_gap_splits_when_configured() -> None:
    tokens = [tok(1, " before", 0.0), tok(2, " after", 5.0)]
    assert len(tokens_to_sentences(tokens)) == 1
    split = tokens_to_sentences(tokens, SentenceConfig(silence_gap=2.0))
    assert [s.text for s in split] == [" before", " after"]


def test_trailing_tokens_flush_into_a_final_sentence() -> None:
    tokens = [tok(1, " no", 0.0), tok(2, " punctuation", 0.3)]
    sentences = tokens_to_sentences(tokens)
    assert len(sentences) == 1
    assert sentences[0].text == " no punctuation"


# ------------------------------------------------------------------- merging


def _chunk(ids: list[int], t0: float, step: float = 0.5) -> list[AlignedToken]:
    return [tok(i, f" w{i}", t0 + n * step, 0.4) for n, i in enumerate(ids)]


def test_disjoint_chunks_concatenate() -> None:
    a = _chunk([1, 2, 3], 0.0)
    b = _chunk([4, 5, 6], 10.0)
    for merge in (merge_longest_contiguous, merge_longest_common_subsequence):
        assert [t.id for t in merge(a, b, overlap_duration=15.0)] == [1, 2, 3, 4, 5, 6]


def test_empty_side_returns_the_other() -> None:
    b = _chunk([1], 0.0)
    for merge in (merge_longest_contiguous, merge_longest_common_subsequence):
        assert merge([], b, overlap_duration=15.0) == b
        assert merge(b, [], overlap_duration=15.0) == b


def test_overlap_dedupes_by_id_and_time() -> None:
    # Chunk A covers ids 1-6; chunk B re-decodes 4-6 at the same times, then
    # continues 7-9. The merged stream must contain each id exactly once.
    a = _chunk([1, 2, 3, 4, 5, 6], 0.0)
    b = _chunk([4, 5, 6, 7, 8, 9], 1.5)
    for merge in (merge_longest_contiguous, merge_longest_common_subsequence):
        merged = merge(a, b, overlap_duration=1.5)
        assert [t.id for t in merged] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_contiguous_merge_raises_when_pairs_are_scarce() -> None:
    # Overlapping in time, but B re-decoded the region to entirely different
    # ids -- no pair matches. The bare RuntimeError IS the contract:
    # chunking.py catches exactly this type to fall back to the LCS merge.
    a = _chunk([1, 2, 3, 4], 0.0)
    b = _chunk([9, 8, 7, 6], 1.0)
    with pytest.raises(RuntimeError):
        merge_longest_contiguous(a, b, overlap_duration=2.0)
    # The LCS fallback handles the same input without raising.
    merged = merge_longest_common_subsequence(a, b, overlap_duration=2.0)
    assert merged, "fallback must still produce a stream"


def test_thin_overlap_falls_back_to_midpoint_cutoff() -> None:
    # Fewer than two tokens on one side of the overlap: both merges cut at the
    # midpoint between A's end and B's start rather than pair-matching.
    a = _chunk([1, 2, 3], 0.0)
    b = [tok(9, " straggler", a[-1].end - 0.05, 0.4), *_chunk([10, 11], 5.0)]
    for merge in (merge_longest_contiguous, merge_longest_common_subsequence):
        merged = merge(a, b, overlap_duration=0.2)
        ids = [t.id for t in merged]
        assert ids == sorted(set(ids), key=ids.index), "no duplicate ids"
        assert set(ids) <= {1, 2, 3, 9, 10, 11}
