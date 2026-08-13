"""The whisper engine: the mapping, the call it makes, and one real run.

mlx-whisper is stubbed in every fast test here -- not to avoid asserting on a
stub, but because what these tests are about IS the boundary: which keyword
arguments dsj sends, and what it does with the dict that comes back. The one
test that runs the real thing is slow-marked and runs whisper-tiny against
speech `say` synthesizes, so nothing binary is committed.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from dsj import whisper as whisper_mod
from dsj.suno import transcribe
from dsj.whisper import ROMAN_URDU_PROMPT, WhisperUnavailable, transcribe_whisper


def _stub_mlx_whisper(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any], seen: dict[str, Any] | None = None
) -> None:
    """Put a fake `mlx_whisper` in sys.modules for the duration of one test.

    transcribe_whisper imports it inside the function body, so the import runs
    on every call and reads sys.modules fresh -- there is no
    dsj.whisper.mlx_whisper attribute to patch.
    """
    module = ModuleType("mlx_whisper")

    def fake_transcribe(audio: str, **kwargs: Any) -> dict[str, Any]:
        if seen is not None:
            seen.update({"audio": audio, **kwargs})
        return result

    module.transcribe = fake_transcribe  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)


def _result(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "text": "  Mujhe maloom nahin. Aap kaise hain?  ",
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": " Mujhe maloom nahin.",
                "words": [
                    {"word": " Mujhe", "start": 0.0, "end": 0.5},
                    {"word": " maloom", "start": 0.5, "end": 1.0},
                    {"word": " nahin.", "start": 1.0, "end": 1.5},
                ],
            },
            {
                "start": 1.5,
                "end": 2.5,
                "text": " Aap kaise hain?",
                "words": [{"word": " Aap", "start": 1.5, "end": 2.5}],
            },
        ],
    }
    return base | over


def test_segments_become_the_payloads_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_mlx_whisper(monkeypatch, _result())
    got = transcribe_whisper(Path("a.wav"))

    assert got.text == "Mujhe maloom nahin. Aap kaise hain?"
    assert [s["start"] for s in got.sentences] == [0.0, 1.5]
    assert got.sentences[0]["text"] == "Mujhe maloom nahin."
    # The token list is what merge.py votes over, and `t` is the only field it
    # reads. The word text keeps its leading space, as parakeet's tokens do.
    assert got.sentences[0]["tokens"] == [
        {"t": 0.0, "w": " Mujhe"},
        {"t": 0.5, "w": " maloom"},
        {"t": 1.0, "w": " nahin."},
    ]


def test_numpy_times_are_narrowed_to_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Word times arrive as np.float64, which json.dumps refuses.

    Not a hypothetical: a real whisper-tiny run returns
    `{'word': ' The', 'start': np.float64(0.0), ...}`. Without the float()
    at the boundary the transcript raises at the write, an hour of ASR after
    the last point where anything could be salvaged.
    """
    result = _result(
        segments=[
            {
                "start": np.float64(0.0),
                "end": np.float64(1.0),
                "text": " hello",
                "words": [{"word": " hello", "start": np.float64(0.0), "end": np.float64(1.0)}],
            }
        ]
    )
    _stub_mlx_whisper(monkeypatch, result)
    got = transcribe_whisper(Path("a.wav"))

    assert type(got.sentences[0]["tokens"][0]["t"]) is float
    json.dumps(got.sentences)  # the assertion that matters: it serialises


def test_the_call_asks_for_words_and_silences_the_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two arguments dsj cannot get wrong, pinned against upstream drift.

    `word_timestamps=True` is the reason to choose whisper here at all -- the
    speaker vote is per token. `verbose=None` reads backwards: mlx-whisper
    disables its tqdm on `verbose is not False`, so False is the value that
    PRINTS an 11,580-frame bar over dsj's own progress line.
    """
    seen: dict[str, Any] = {}
    _stub_mlx_whisper(monkeypatch, _result(), seen)
    transcribe_whisper(Path("a.wav"), model_id="m", language="ur", prompt=ROMAN_URDU_PROMPT)

    assert seen["word_timestamps"] is True
    assert seen["verbose"] is None
    assert seen["path_or_hf_repo"] == "m"
    assert seen["language"] == "ur"
    assert seen["initial_prompt"] == ROMAN_URDU_PROMPT


def test_the_missing_extra_names_both_install_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None in sys.modules is what an absent module raises through."""
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with pytest.raises(WhisperUnavailable) as exc:
        transcribe_whisper(Path("a.wav"))

    # Both forms, because `uv sync` is a no-op for someone who installed the
    # tool rather than the project, and the reverse.
    assert "uv tool install" in str(exc.value)
    assert "uv sync --extra whisper" in str(exc.value)


@pytest.mark.usefixtures("already_extracted_media")
def test_the_whisper_engine_writes_the_schema_and_leaves_no_checkpoint(
    monkeypatch: pytest.MonkeyPatch, fake_media: Path, tmp_path: Path
) -> None:
    """The transcript a whisper run leaves behind is the same document.

    Downstream -- dekho, and any agent reading the index -- must not be able to
    tell which engine wrote it apart from the model id. The checkpoint check is
    the other half: whisper has no chunk loop of dsj's to bank, so a file
    beside the output would be a stale one nothing could ever resume from.
    """
    _stub_mlx_whisper(monkeypatch, _result())
    out = tmp_path / "out.json"

    payload = transcribe(fake_media, out, engine="whisper", diarize=False)

    assert set(payload) == {"audio", "model", "text", "sentences"}
    assert payload["model"] == whisper_mod.DEFAULT_WHISPER_MODEL
    assert json.loads(out.read_text())["sentences"][0]["tokens"][0] == {"t": 0.0, "w": " Mujhe"}
    assert list(tmp_path.glob("*.checkpoint*")) == []


def test_an_unknown_engine_is_refused_by_name(fake_media: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown engine 'wshiper'"):
        transcribe(fake_media, tmp_path / "o.json", engine="wshiper")


def test_parakeet_refuses_the_flags_that_are_not_its(fake_media: Path, tmp_path: Path) -> None:
    """Silently ignoring --prompt would be the worst of the three options.

    A prompt is the whole mechanism behind Roman Urdu output. Accepting one and
    dropping it means a run that looks configured and is not.
    """
    with pytest.raises(ValueError, match="whisper's; parakeet takes neither"):
        transcribe(fake_media, tmp_path / "o.json", language="ur")


def test_roman_urdu_sets_the_engine_the_language_and_the_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One flag, because the three that make it work are not guessable."""
    import dsj.suno as suno_mod

    seen: dict[str, object] = {}

    def fake(media: Path, out: Path, model: str = "", **kw: object) -> dict[str, object]:
        seen.update({"model": model, **kw})
        return {"sentences": []}

    monkeypatch.setattr(suno_mod, "transcribe", fake)
    out = tmp_path / "o.json"
    assert suno_mod.main(["v.mov", "-o", str(out), "--roman-urdu"]) == 0

    assert seen["engine"] == "whisper"
    assert seen["language"] == "ur"
    assert seen["prompt"] == ROMAN_URDU_PROMPT
    # The model default follows the engine, so --roman-urdu alone must not send
    # parakeet's repo id to whisper.
    assert seen["model"] == whisper_mod.DEFAULT_WHISPER_MODEL


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None or shutil.which("say") is None,
    reason="needs the whisper extra and macOS `say`",
)
def test_a_real_whisper_run_returns_words_with_times(tmp_path: Path) -> None:
    """The stubs above assert the wiring; this asserts the wiring was right.

    whisper-tiny rather than the default turbo, and three seconds of `say`
    rather than a committed clip: the point is that a real model returns the
    shape dsj maps, not that a small model is accurate.
    """
    wav = tmp_path / "said.wav"
    subprocess.run(
        ["say", "-o", str(wav), "--data-format=LEI16@16000", "The quick brown fox."],
        check=True,
    )

    got = transcribe_whisper(wav, model_id="mlx-community/whisper-tiny", language="en")

    # Two words, not the sentence: tiny heard "the quick brown socks." on the
    # first run of this test. Its accuracy is not what is on trial -- that
    # something real came back, in the shape dsj maps, is.
    assert "quick brown" in got.text.lower()
    tokens = [t for s in got.sentences for t in s["tokens"]]
    assert len(tokens) >= 4
    assert all(isinstance(t["t"], float) for t in tokens)
    # Monotonic, because a token list out of order silently mislabels speakers:
    # merge.py bisects it.
    assert [t["t"] for t in tokens] == sorted(t["t"] for t in tokens)
