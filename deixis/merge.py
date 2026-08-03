"""Decide who spoke each sentence, by counting token votes against turns.

Diarization emits speaker turns as intervals over the recording; the transcript
has sentences with their own start/end and a start time on every token. The two
segmentations do not line up. parakeet breaks on punctuation and pauses, senko
breaks on voice, and one "sentence" on the reference recording runs 108 seconds
and crosses several turns.

Tokens vote rather than the sentence's span being intersected with the turns,
because a span includes the silences inside it. A speaker who happened to be
talking during a pause in the middle of a sentence can hold more of that span
than the speaker who said the words, and interval overlap would hand them the
sentence. Counting tokens can only ever count words that were actually said.

The two agree on all but 6 of 664 sentences on the reference recording, so this
is a small correctness win, not a large one. It is kept because the failure mode
of interval overlap scales with how interleaved a conversation is, and a
two-person call with long uninterrupted stretches is the easy case, not the
adversarial one.

Nothing here imports senko or loads a model: it is a function of two plain data
structures, which is what makes it testable without a diarizer.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple


class Turn(NamedTuple):
    """One speaker holding the floor, from `start` to `end` seconds.

    `speaker` is an index into a labels list, not an identity. Diarization
    clusters are arbitrary and per-file; an integer looks like the artifact it
    is, where "SPEAKER_01" invites a reader to believe otherwise.
    """

    start: float
    end: float
    speaker: int


class TurnIndex:
    """Answer "who held the floor at time t" in log time.

    Built once per transcript and queried once per token -- 14,394 tokens
    against 220 turns on the reference recording, so a linear scan per token
    would be three million interval comparisons for an answer bisect gives for
    free.
    """

    def __init__(self, turns: Sequence[Turn]) -> None:
        if not turns:
            raise ValueError("cannot index an empty turn list")
        # Sorted here rather than demanded of the caller: it is one pass over a
        # couple of hundred items, and every query below depends on the order
        # holding. A caller that sorted already pays almost nothing.
        self._turns = sorted(Turn(*t) for t in turns)
        self._starts = [t.start for t in self._turns]

    def speaker_at(self, t: float) -> int | None:
        """Who was speaking at `t`, or None if nobody was.

        None is the answer for a time in the silence between two turns, and for
        a time before the first turn begins. Both are real: 219 of 14,394 tokens
        on the reference recording land in inter-turn silence.

        A time falling exactly on a shared boundary belongs to the later turn,
        which is what bisect_right gives and is as good as either choice.
        """
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0:
            return None
        turn = self._turns[i]
        return turn.speaker if turn.start <= t <= turn.end else None

    def nearest_speaker(self, t: float) -> int:
        """Who was speaking closest to `t`, whether or not anyone was at it.

        Only the turn before `t` and the turn after it can be the nearest, so
        there are two candidates however long the recording is.
        """
        i = bisect.bisect_right(self._starts, t) - 1
        candidates = [j for j in (i, i + 1) if 0 <= j < len(self._turns)]
        nearest = min(candidates, key=lambda j: _distance(self._turns[j], t))
        return self._turns[nearest].speaker


def _distance(turn: Turn, t: float) -> float:
    if turn.start <= t <= turn.end:
        return 0.0
    return min(abs(t - turn.start), abs(t - turn.end))


def label_sentence(sentence: Mapping, index: TurnIndex) -> int:
    """The speaker index for one sentence, by majority of its tokens.

    `sentence` is the dict transcribe() writes: a "start" and a "tokens" list of
    {"t": seconds, "w": text}. Token end times are not in the transcript -- they
    were dropped for file size -- so each token votes once by its start, not
    weighted by how long it took to say.
    """
    votes: Counter[int] = Counter()
    first_vote: dict[int, float] = {}
    for token in sentence["tokens"]:
        speaker = index.speaker_at(token["t"])
        # A token in a gap abstains rather than being pushed onto the nearest
        # turn. It is 1.5% of tokens and they are ambiguous by construction,
        # while the rest of the same sentence is not.
        if speaker is None:
            continue
        votes[speaker] += 1
        first_vote.setdefault(speaker, token["t"])

    if not votes:
        # Every token fell in a gap. Never observed on the reference recording,
        # but a very short sentence at a turn boundary could reach it, and a
        # crash here would take down a pass that is meant to be optional.
        return index.nearest_speaker(sentence["start"])

    most = max(votes.values())
    tied = [speaker for speaker, count in votes.items() if count == most]
    # The earliest token breaks a tie: deterministic, and it favours whoever
    # started the sentence, which is the reading a person would give. Exactly
    # one of 664 sentences needed this on the reference recording. Written as a
    # min over the tied speakers rather than "the first token's speaker" so a
    # third speaker who spoke first but lost cannot take the sentence.
    return min(tied, key=lambda speaker: first_vote[speaker])


def label_sentences(sentences: Sequence[Mapping], turns: Sequence[Turn]) -> list[int]:
    """A speaker index per sentence, in the order the sentences came in."""
    index = TurnIndex(turns)
    return [label_sentence(s, index) for s in sentences]
