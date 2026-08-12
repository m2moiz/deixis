"""whisper, for the audio parakeet cannot read.

`parakeet-tdt-0.6b-v3` is the default engine and stays the default: it runs at
~13x realtime and does not invent words over silence. What it does not have is
Urdu -- its 25 languages are European, and `ur` is not among the model card's
tags. A voice note that mixes Urdu and English comes back as nothing usable.

whisper-large-v3-turbo does read it, at a cost: measured on 116s of Urdu speech
it took 84.7s (~1.4x realtime) against parakeet's ~13x. That is fine for a voice
note and would not be for an hour of lecture, which is why this is a flag and
not a replacement.

ROMAN URDU IS A PROMPT, NOT A SETTING. whisper transcribes Urdu in Urdu script
by default. Seeding the decoder with a Roman Urdu `initial_prompt` makes it emit
Roman instead, and it carries across windows through whisper's own
condition-on-previous-text: over the same 116s, 275 of 277 words came back in
Latin. The two that did not were single words inside otherwise-Roman sentences.

That trick is model-specific, and the difference is not subtle. The full
whisper-large-v3 ignores the prompt completely -- 280 of 280 words in Urdu
script, in 218s rather than 85s. Hence the default here is turbo, and changing
it means re-measuring rather than assuming.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_WHISPER_MODEL",
    "INSTALL_HINT",
    "ROMAN_URDU_PROMPT",
    "SAMPLE_RATE",
    "WhisperUnavailable",
    "transcribe_whisper",
]

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from jaano.asr import Transcription

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

# whisper's own front end resamples to 16 kHz mono whatever it is handed, so
# this is not a preference: handing it anything else means paying for a second
# resample of an hour of audio inside a library that will not report progress.
SAMPLE_RATE = 16000

INSTALL_HINT = (
    'uv tool install "jaano[whisper] @ git+https://github.com/m2moiz/jaano"'
    " (or `uv sync --extra whisper` from a clone)"
)

# Not a magic incantation -- a worked example of the output wanted, which is
# what an initial_prompt is for. It carries a mid-sentence English word on
# purpose ("duty free", "okay"), because the target is code-switched speech and
# a prompt of pure Urdu biases against leaving English in Latin where it was
# said in English.
ROMAN_URDU_PROMPT = (
    "Yeh Roman Urdu transcript hai. Mujhe maloom nahin tha ke aap ne mehsoos kiya ya nahin. "
    "Is mulk mein zyada tar cheezein duty free hain, okay?"
)


class WhisperUnavailable(RuntimeError):
    """The whisper engine was asked for and mlx-whisper is not installed."""


def transcribe_whisper(
    audio: Path,
    *,
    model_id: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
    prompt: str | None = None,
) -> Transcription:
    """Transcribe `audio` end to end with whisper.

    Unchunked, unlike the parakeet path: whisper does its own 30-second windows
    and threads each window's text into the next as a prompt, which is exactly
    the mechanism the Roman Urdu bias rides on. Cutting the file up here would
    break that continuity to buy a resume that voice-note-length audio does not
    need.

    Args:
        audio: A file ffmpeg can open. 16 kHz mono costs least; anything else
            is resampled inside whisper.
        model_id: An mlx-community whisper repo.
        language: ISO code, or None to let whisper detect it. Naming it saves
            the detection pass and stops a code-switched clip being detected as
            English.
        prompt: Seeds the decoder. `ROMAN_URDU_PROMPT` is the measured one.

    Returns:
        The full text and the payload's sentences, one per whisper segment.

    Raises:
        WhisperUnavailable: if mlx-whisper is not installed.
    """
    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise WhisperUnavailable(
            f"the whisper engine needs mlx-whisper, which is an optional extra. "
            f"Install it with `{INSTALL_HINT}`, or use the default "
            f"`--engine parakeet`."
        ) from exc

    # mlx-whisper annotates its parameters but returns
    # `dict[str, str | list[Unknown]]`, so every value read out of the result
    # is partially unknown at the read. Restating the call once here stops that
    # spreading through the loop below. The suppression is on the member access
    # itself, which has no expression to annotate.
    transcribe = cast(
        "Callable[..., dict[str, Any]]",
        mlx_whisper.transcribe,  # pyright: ignore[reportUnknownMemberType]
    )
    result = transcribe(
        str(audio),
        path_or_hf_repo=model_id,
        language=language,
        initial_prompt=prompt,
        # The whole point of choosing whisper here. merge.py's speaker vote is
        # per token, and without this whisper returns segment bounds only.
        word_timestamps=True,
        # None, and NOT False. mlx-whisper reads this backwards from the way it
        # looks: `disable=verbose is not False`, so verbose=False is the value
        # that SHOWS its tqdm bar, and only None silences it. Observed -- the
        # first run of this function printed an 11,580-frame bar over jaano's
        # own line. jaano owns this terminal row, and a detached run reads
        # --status rather than stderr.
        verbose=None,
    )

    segments = cast("list[dict[str, Any]]", result.get("segments") or [])
    sentences: list[dict[str, Any]] = []
    for segment in segments:
        words = cast("list[dict[str, Any]]", segment.get("words") or [])
        sentences.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment["text"]).strip(),
                # The word text is kept exactly as whisper emits it, leading
                # space and all, which is how parakeet's tokens arrive too. A
                # reader joining tokens gets the sentence back either way.
                "tokens": [{"t": float(w["start"]), "w": str(w["word"])} for w in words],
            }
        )
    return Transcription(text=str(result.get("text", "")).strip(), sentences=sentences)
