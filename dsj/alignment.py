"""dsj's own token vocabulary: a word with a timestamp, and the merge math.

Ported verbatim from parakeet-mlx 0.5.2's alignment.py (Apache-2.0,
github.com/senstella/parakeet-mlx). Nothing in that file touches MLX -- it is
pure Python plus one numpy call -- yet importing it made resume and chunk
stitching require Apple hardware, which is the single reason dsj could not
install on anything else. Owning the copy is the extraction's first move
(docs/android-port-design.md).

Four deliberate deviations from upstream, and only four:

  - The two merge functions are annotated `list[AlignedToken]`. Upstream
    builds a bare list, which forced a cast at every call site in chunking.py.
  - tokens_to_sentences defaults `config` to None instead of a shared
    SentenceConfig() instance; same behaviour, no mutable default.
  - Loop variable `l` is renamed `jj` (lint); the algorithm is untouched.
  - Empty accumulators carry type annotations for strict checking.

Everything else is byte-faithful on purpose, including two load-bearing
choices a cleanup would be tempted to "fix":

  - AlignedToken.__post_init__ recomputes `end = start + duration`. The
    checkpoint file omits `end` and relies on exactly this arithmetic;
    tests/test_checkpoint_golden.py pins it to the float result, not the idea.
  - merge_longest_contiguous raises bare RuntimeError when too few token
    pairs line up. chunking.py catches exactly that type to fall back to the
    LCS merge -- the exception type is control flow, not an error report.

The upstream-equivalence test in tests/test_chunking.py compares this copy's
behaviour against the installed parakeet-mlx while both exist; once
parakeet-mlx is an optional extra, its absence legitimately skips that one
comparison and these unit tests carry the load (tests/test_alignment.py).
"""

from __future__ import annotations

__all__ = [
    "AlignedResult",
    "AlignedSentence",
    "AlignedToken",
    "SentenceConfig",
    "merge_longest_common_subsequence",
    "merge_longest_contiguous",
    "sentences_to_result",
    "tokens_to_sentences",
]

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedToken:
    id: int
    text: str
    start: float
    duration: float
    confidence: float = 1.0  # confidence score (0.0 to 1.0)
    end: float = 0.0  # temporary

    def __post_init__(self) -> None:
        self.end = self.start + self.duration


@dataclass
class AlignedSentence:
    text: str
    tokens: list[AlignedToken]
    start: float = 0.0  # temporary
    end: float = 0.0  # temporary
    duration: float = 0.0  # temporary
    confidence: float = 1.0  # aggregate confidence score

    def __post_init__(self) -> None:
        self.tokens = list(sorted(self.tokens, key=lambda x: x.start))
        self.start = self.tokens[0].start
        self.end = self.tokens[-1].end
        self.duration = self.end - self.start
        # Compute geometric mean of token confidences
        confidences = np.array([t.confidence for t in self.tokens])
        self.confidence = float(np.exp(np.mean(np.log(confidences + 1e-10))))


@dataclass
class AlignedResult:
    text: str
    sentences: list[AlignedSentence]

    def __post_init__(self) -> None:
        self.text = self.text.strip()

    @property
    def tokens(self) -> list[AlignedToken]:
        return [token for sentence in self.sentences for token in sentence.tokens]


@dataclass
class SentenceConfig:
    max_words: int | None = None
    silence_gap: float | None = None
    max_duration: float | None = None


def tokens_to_sentences(
    tokens: list[AlignedToken], config: SentenceConfig | None = None
) -> list[AlignedSentence]:
    # Upstream's signature defaults to a shared SentenceConfig() instance; a
    # None default is the same behaviour without the mutable-default lint.
    if config is None:
        config = SentenceConfig()
    sentences: list[AlignedSentence] = []
    current_tokens: list[AlignedToken] = []

    for idx, token in enumerate(tokens):
        current_tokens.append(token)

        is_punctuation = (
            # hacky, will fix
            "!" in token.text
            or "?" in token.text
            or "。" in token.text
            or "？" in token.text
            or "！" in token.text
            or (
                "." in token.text
                and (idx == len(tokens) - 1 or " " in tokens[idx + 1].text)
            )
        )
        is_word_limit = (
            (config.max_words is not None)
            and (idx != len(tokens) - 1)
            and (
                len([x for x in current_tokens if " " in x.text])
                + (1 if " " in tokens[idx + 1].text else 0)
                > config.max_words
            )
        )
        is_long_silence = (
            (config.silence_gap is not None)
            and (idx != len(tokens) - 1)
            and (tokens[idx + 1].start - token.end >= config.silence_gap)
        )
        is_over_duration = (config.max_duration is not None) and (
            token.end - current_tokens[0].start >= config.max_duration
        )

        if is_punctuation or is_word_limit or is_long_silence or is_over_duration:
            sentence_text = "".join(t.text for t in current_tokens)
            sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
            sentences.append(sentence)

            current_tokens = []

    if current_tokens:
        sentence_text = "".join(t.text for t in current_tokens)
        sentence = AlignedSentence(text=sentence_text, tokens=current_tokens)
        sentences.append(sentence)

    return sentences


def sentences_to_result(sentences: list[AlignedSentence]) -> AlignedResult:
    return AlignedResult("".join(sentence.text for sentence in sentences), sentences)


def merge_longest_contiguous(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
) -> list[AlignedToken]:
    if not a or not b:
        return b if not a else a

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    enough_pairs = len(overlap_a) // 2

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    best_contiguous: list[tuple[int, int]] = []
    for i in range(len(overlap_a)):
        for j in range(len(overlap_b)):
            if (
                overlap_a[i].id == overlap_b[j].id
                and abs(overlap_a[i].start - overlap_b[j].start) < overlap_duration / 2
            ):
                current: list[tuple[int, int]] = []
                k, jj = i, j
                while (
                    k < len(overlap_a)
                    and jj < len(overlap_b)
                    and overlap_a[k].id == overlap_b[jj].id
                    and abs(overlap_a[k].start - overlap_b[jj].start)
                    < overlap_duration / 2
                ):
                    current.append((k, jj))
                    k += 1
                    jj += 1

                if len(current) > len(best_contiguous):
                    best_contiguous = current

    if len(best_contiguous) >= enough_pairs:
        a_start_idx = len(a) - len(overlap_a)
        lcs_indices_a = [a_start_idx + pair[0] for pair in best_contiguous]
        lcs_indices_b = [pair[1] for pair in best_contiguous]

        result: list[AlignedToken] = []
        result.extend(a[: lcs_indices_a[0]])

        for i in range(len(best_contiguous)):
            idx_a = lcs_indices_a[i]
            idx_b = lcs_indices_b[i]

            result.append(a[idx_a])

            if i < len(best_contiguous) - 1:
                next_idx_a = lcs_indices_a[i + 1]
                next_idx_b = lcs_indices_b[i + 1]

                gap_tokens_a = a[idx_a + 1 : next_idx_a]
                gap_tokens_b = b[idx_b + 1 : next_idx_b]

                if len(gap_tokens_b) > len(gap_tokens_a):
                    result.extend(gap_tokens_b)
                else:
                    result.extend(gap_tokens_a)

        result.extend(b[lcs_indices_b[-1] + 1 :])
        return result
    else:
        raise RuntimeError(f"No pairs exceeding {enough_pairs}")


def merge_longest_common_subsequence(
    a: list[AlignedToken],
    b: list[AlignedToken],
    *,
    overlap_duration: float,
) -> list[AlignedToken]:
    if not a or not b:
        return b if not a else a

    a_end_time = a[-1].end
    b_start_time = b[0].start

    if a_end_time <= b_start_time:
        return a + b

    overlap_a = [token for token in a if token.end > b_start_time - overlap_duration]
    overlap_b = [token for token in b if token.start < a_end_time + overlap_duration]

    if len(overlap_a) < 2 or len(overlap_b) < 2:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    dp = [[0 for _ in range(len(overlap_b) + 1)] for _ in range(len(overlap_a) + 1)]

    for i in range(1, len(overlap_a) + 1):
        for j in range(1, len(overlap_b) + 1):
            if (
                overlap_a[i - 1].id == overlap_b[j - 1].id
                and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
                < overlap_duration / 2
            ):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_pairs: list[tuple[int, int]] = []
    i, j = len(overlap_a), len(overlap_b)

    while i > 0 and j > 0:
        if (
            overlap_a[i - 1].id == overlap_b[j - 1].id
            and abs(overlap_a[i - 1].start - overlap_b[j - 1].start)
            < overlap_duration / 2
        ):
            lcs_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_pairs.reverse()

    if not lcs_pairs:
        cutoff_time = (a_end_time + b_start_time) / 2
        return [t for t in a if t.end <= cutoff_time] + [
            t for t in b if t.start >= cutoff_time
        ]

    a_start_idx = len(a) - len(overlap_a)
    lcs_indices_a = [a_start_idx + pair[0] for pair in lcs_pairs]
    lcs_indices_b = [pair[1] for pair in lcs_pairs]

    result: list[AlignedToken] = []

    result.extend(a[: lcs_indices_a[0]])

    for i in range(len(lcs_pairs)):
        idx_a = lcs_indices_a[i]
        idx_b = lcs_indices_b[i]

        result.append(a[idx_a])

        if i < len(lcs_pairs) - 1:
            next_idx_a = lcs_indices_a[i + 1]
            next_idx_b = lcs_indices_b[i + 1]

            gap_tokens_a = a[idx_a + 1 : next_idx_a]
            gap_tokens_b = b[idx_b + 1 : next_idx_b]

            if len(gap_tokens_b) > len(gap_tokens_a):
                result.extend(gap_tokens_b)
            else:
                result.extend(gap_tokens_a)

    result.extend(b[lcs_indices_b[-1] + 1 :])

    return result
