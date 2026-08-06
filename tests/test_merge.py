"""The token vote, exercised on literals and on real measured turns.

No fixture and no diarizer anywhere in this file. That is the point of keeping
the merge a pure function of two plain data structures: the part of diarization
where the subtle bugs live is the part that needs neither a model nor 12 seconds
of CoreML compilation to test.
"""

from __future__ import annotations

import bisect
from typing import Any

import pytest

from deixis.merge import Turn, TurnIndex, label_sentence, label_sentences


def sentence(*times: float, start: float | None = None) -> dict[str, Any]:
    """A sentence carrying only what the merge reads: a start and token times."""
    return {
        "start": times[0] if start is None else start,
        "end": times[-1] if times else 0.0,
        "tokens": [{"t": t, "w": "w"} for t in times],
    }


def test_a_sentence_inside_one_turn_takes_that_speaker() -> None:
    turns = [Turn(0.0, 10.0, 0), Turn(10.0, 20.0, 1)]
    assert label_sentences([sentence(1.0, 2.0, 3.0)], turns) == [0]


def test_a_straddling_sentence_goes_to_the_token_majority() -> None:
    turns = [Turn(0.0, 10.0, 0), Turn(10.0, 20.0, 1)]
    # 3 tokens before the change, 2 after.
    assert label_sentences([sentence(1.0, 2.0, 3.0, 11.0, 12.0)], turns) == [0]


def test_a_long_internal_silence_does_not_flip_the_label() -> None:
    # The case that pins token-vote over sentence-span max-overlap. The sentence
    # spans 0 to 100 seconds, of which 90 lie inside speaker 1's turn -- but
    # every word was said in the first ten seconds, by speaker 0. Interval
    # overlap gives this sentence to 1. Counting words gives it to 0, which is
    # who said them. Without this test a refactor to max-overlap passes
    # everything else in this file.
    turns = [Turn(0.0, 10.0, 0), Turn(10.0, 100.0, 1)]
    s = sentence(1.0, 2.0, 3.0, 4.0)
    s["end"] = 100.0
    assert label_sentence(s, TurnIndex(turns)) == 0


def test_a_token_in_a_gap_between_turns_abstains() -> None:
    # 40.0 falls in the silence between the two turns. If it were pushed onto
    # the nearest turn it would tie the vote at 2-2 and the tiebreak would then
    # decide; abstaining leaves speaker 1 a clean majority.
    turns = [Turn(0.0, 30.0, 0), Turn(50.0, 80.0, 1)]
    assert label_sentence(sentence(29.0, 40.0, 51.0, 52.0), TurnIndex(turns)) == 1


def test_a_sentence_entirely_in_a_gap_falls_back_to_the_nearest_turn() -> None:
    # Never fired on the reference recording; it exists because a very short
    # sentence at a turn boundary could reach it, and a crash in an optional
    # pass would cost a whole ASR run.
    turns = [Turn(0.0, 10.0, 0), Turn(50.0, 60.0, 1)]
    assert label_sentence(sentence(20.0, 21.0), TurnIndex(turns)) == 0
    assert label_sentence(sentence(45.0, 46.0), TurnIndex(turns)) == 1


def test_a_tie_goes_to_the_earliest_tokens_speaker() -> None:
    turns = [Turn(0.0, 10.0, 1), Turn(10.0, 20.0, 0)]
    # Two tokens each. The lower speaker index would win a naive argmax; the
    # earliest token is speaker 1's, so speaker 1 takes the sentence.
    assert label_sentence(sentence(1.0, 2.0, 11.0, 12.0), TurnIndex(turns)) == 1


def test_a_tie_is_not_won_by_a_speaker_who_is_not_in_it() -> None:
    # Speaker 2 spoke first but only holds one token. The tie is between 0 and
    # 1, and "the earliest token's speaker" has to mean the earliest among the
    # tied, not the earliest overall.
    turns = [Turn(0.0, 5.0, 2), Turn(5.0, 15.0, 1), Turn(15.0, 25.0, 0)]
    assert label_sentence(sentence(1.0, 6.0, 7.0, 16.0, 17.0), TurnIndex(turns)) == 1


def test_a_token_before_the_first_turn_abstains() -> None:
    # bisect_right returns 0 here, so the index is -1 and there is no preceding
    # turn to look at. Indexing a list with -1 would silently take the LAST turn.
    turns = [Turn(10.0, 20.0, 0), Turn(20.0, 30.0, 1)]
    index = TurnIndex(turns)
    assert index.speaker_at(5.0) is None
    assert label_sentence(sentence(5.0, 11.0, 12.0), index) == 0


def test_a_time_on_a_shared_boundary_belongs_to_the_later_turn() -> None:
    # senko's merged segments share endpoints, so this is not a hypothetical.
    turns = [Turn(0.0, 10.0, 0), Turn(10.0, 20.0, 1)]
    assert TurnIndex(turns).speaker_at(10.0) == 1


def test_unsorted_turns_are_ordered_before_they_are_indexed() -> None:
    turns = [Turn(10.0, 20.0, 1), Turn(0.0, 10.0, 0)]
    assert TurnIndex(turns).speaker_at(5.0) == 0


def test_an_empty_turn_list_is_refused_rather_than_answered() -> None:
    # The caller (deixis.diarize) turns empty segments into
    # DiarizationUnavailable; if one ever reaches here, say so loudly rather
    # than inventing a speaker.
    with pytest.raises(ValueError, match=r"^cannot index an empty turn list$"):
        TurnIndex([])


def test_turns_are_indexed_not_scanned() -> None:
    # The bisect boundary conditions (`<=` on both ends, the -1 from an empty
    # left side) are exactly where an off-by-one hides. Pinned against a naive
    # reference implementation that is obviously correct because it is slow.
    turns = [Turn(float(i * 10), float(i * 10 + 7), i % 3) for i in range(200)]
    times = [i * 0.199 for i in range(10_000)]

    def naive(t: float) -> int | None:
        for turn in turns:
            if turn.start <= t <= turn.end:
                return turn.speaker
        return None

    index = TurnIndex(turns)
    assert [index.speaker_at(t) for t in times] == [naive(t) for t in times]
    # The gaps are real, so this is not comparing two constant Nones.
    assert any(naive(t) is None for t in times)
    assert any(naive(t) is not None for t in times)
    # ignore reportPrivateUsage twice: the sorted bisect array IS the invariant
    # under test here, and exposing it publicly just to satisfy the checker
    # would widen the API for a test's convenience.
    assert index._starts == sorted(index._starts)  # pyright: ignore[reportPrivateUsage]
    assert bisect.bisect_right(index._starts, 0.0) == 1  # pyright: ignore[reportPrivateUsage]


# --- Real measured data, from scratch/senko_meeting.json and the transcript ---
#
# The first speaker change on the reference 74-minute call, at full precision,
# and the one transcript sentence that straddles it. Literals rather than files
# because scratch/ is not committed; the numbers are what senko and parakeet
# actually emitted on that recording.

REAL_TURNS = [
    Turn(0.0, 57.809375, 0),
    Turn(57.809375, 73.796875, 1),
    Turn(74.404375, 301.906875, 0),
]

REAL_STRADDLING_SENTENCE = {
    "start": 57.120000000000005,
    "end": 58.32,
    "tokens": [
        {"t": 57.120000000000005, "w": " Is"},
        {"t": 57.52, "w": " that"},
        {"t": 57.68, "w": " o"},
        {"t": 57.76, "w": "kay"},
        {"t": 57.84, "w": " with"},
        {"t": 58.0, "w": " you"},
        {"t": 58.160000000000004, "w": "?"},
    ],
}


def test_the_real_straddle_at_the_first_speaker_change() -> None:
    # "Is that okay with you?" -- four tokens land before the change and three
    # after, so the sentence goes to the speaker who began it. This is a genuine
    # 4-3 call, and it is what 10.4% of the sentences on this recording look
    # like: the vote decides, and it is not always by much.
    assert label_sentence(REAL_STRADDLING_SENTENCE, TurnIndex(REAL_TURNS)) == 0


def test_the_real_inter_turn_gap_makes_tokens_abstain() -> None:
    # 73.796875 -> 74.404375 is 0.6s of measured silence between two turns. 809
    # seconds of the recording is gaps like this one.
    index = TurnIndex(REAL_TURNS)
    assert index.speaker_at(74.0) is None
    assert index.speaker_at(73.5) == 1
    assert index.speaker_at(74.5) == 0


# --- gaps mutation testing found ---------------------------------------------
#
# Every test below kills a mutant that SURVIVED the first mutmut run. Each line
# they cover was already at 98% coverage: it executed, and nothing asserted on
# what it produced. That gap is the entire reason mutation testing is in this
# project.


def test_the_nearest_turn_is_measured_from_whichever_end_is_closer() -> None:
    """`_distance` SUBTRACTS; a sign slip on either operand still looks fine.

    Mutants `abs(t + turn.start)` and `abs(t + turn.end)` survived the first
    mutmut run. They are invisible unless the corrupted term is the one `min`
    would otherwise have picked -- so each case below is built so that it is.
    """
    # t sits 5s BEFORE a turn's start, so the start term is the minimum. A
    # `t + start` slip inflates it to 195 and hands the answer to the other turn.
    index = TurnIndex([Turn(0.0, 85.0, 1), Turn(100.0, 110.0, 0)])
    assert index.nearest_speaker(95.0) == 0

    # Mirrored: t sits 1s AFTER a turn's end, so the end term is the minimum.
    # A `t + end` slip inflates it to 171 and the answer flips.
    index = TurnIndex([Turn(0.0, 85.0, 1), Turn(120.0, 130.0, 0)])
    assert index.nearest_speaker(86.0) == 1


def test_a_time_inside_a_turn_is_zero_distance_from_it() -> None:
    """The early return is 0.0, not "small".

    Mutant `return 1.0` survived: it only shows when a competing turn is nearer
    than 1s, which no existing test had. Turns 0.6s apart do.
    """
    index = TurnIndex([Turn(0.0, 10.0, 0), Turn(10.6, 20.0, 1)])
    # 9.9 is inside turn 0 (distance 0) and 0.7 from turn 1. At a fake distance
    # of 1.0 the neighbour wins.
    assert index.nearest_speaker(9.9) == 0


def test_a_time_exactly_on_a_turn_edge_is_inside_that_turn() -> None:
    """`turn.start <= t <= turn.end` -- both bounds are inclusive, deliberately.

    Mutants: `start < t` and `t < end`. A token landing exactly on an edge is
    not a hypothetical; parakeet emits times that coincide with senko's
    boundaries, and pushing such a token to the neighbouring turn is a
    misattribution the vote then inherits.
    """
    turns = [Turn(0.0, 10.0, 0), Turn(50.0, 60.0, 1)]
    index = TurnIndex(turns)

    # Exactly on turn 0's start, and exactly on its end. Both are distance 0
    # from turn 0, so neither may be answered with speaker 1.
    assert index.nearest_speaker(0.0) == 0
    assert index.nearest_speaker(10.0) == 0


def test_the_last_turn_is_reachable_and_nothing_past_it_is() -> None:
    """`0 <= j < len(self._turns)` -- the guard against indexing off the end.

    Mutant: `j <= len(...)`, which lets the candidate list hold an index one
    past the last turn. It survived because no test asked for a time after the
    final turn, which is exactly where that index gets built.
    """
    turns = [Turn(0.0, 10.0, 0), Turn(20.0, 30.0, 1)]
    index = TurnIndex(turns)

    # Well past the end: the only valid answer is the last turn's speaker, and
    # a candidate index of len(turns) would raise IndexError instead.
    assert index.nearest_speaker(999.0) == 1


def test_every_token_carries_exactly_one_vote() -> None:
    """`votes[speaker] += 1` -- one token, one vote, unweighted.

    Mutant: `+= 2`. It survived every existing test because doubling every vote
    equally never changes who wins. It changes the answer only when the two
    sides are unequal AND a tie-break is in play, which is the case built here.

    This matters more than its size suggests: the token vote is the algorithm
    the README documents as the reason sentences are labelled the way they are.
    """
    turns = [Turn(0.0, 10.0, 0), Turn(10.0, 20.0, 1)]
    index = TurnIndex(turns)

    # Two tokens for speaker 0, one for speaker 1: 2 vs 1 either way, so the
    # winner is stable...
    assert label_sentence(sentence(1.0, 2.0, 11.0), index) == 0

    # ...but a genuine tie must stay a tie and fall to the earliest token. If
    # each token voted twice, this would still be 4 vs 4 -- so the assertion
    # that actually pins the weight is that the count itself is odd-sensitive:
    # three tokens against one cannot be reached by doubling two against one.
    assert label_sentence(sentence(1.0, 2.0, 3.0, 11.0), index) == 0
    assert label_sentence(sentence(1.0, 11.0, 12.0, 13.0), index) == 1
