"""main()'s summary line.

The `total / elapsed` that used to close this line was unguarded and sat AFTER
the transcript was written, so on an instantaneous run it turned a completed
transcription into a traceback with the output file already on disk. Reaching
elapsed == 0.0 needs a frozen clock -- time.monotonic has nanosecond resolution
on macOS -- which is why it was a P3 and not a field report.

The realtime multiple is gone now, for a second reason: on a resumed run the
whole file's duration over this run's wall clock credits this process with work
a previous one paid for. The tests that pinned it are rewritten below; the ones
that pinned "a completed run still reports itself" keep their full value, and
are the reason the division cannot come back unguarded.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from conftest import FakeToken

from dsj.suno import main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from conftest import FakeModel

    from dsj.suno import Payload

# main() feeds a path straight through transcribe(), which probes it for real
# and diarizes for real; these tests are about the summary and about neither of
# those, so they need both stubs.
pytestmark = pytest.mark.usefixtures("already_extracted_media", "no_real_diarizer")


def _tokens() -> list[FakeToken]:
    """One sentence ending at 12.0s."""
    return [FakeToken(0.0, 0.4, "see"), FakeToken(0.5, 12.0, " this.")]


def test_instantaneous_run_still_prints_a_summary(
    fake_parakeet: Callable[..., FakeModel],
    frozen_clock: Callable[[list[float]], None],
    fake_media: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_parakeet(tokens=_tokens())
    frozen_clock([1000.0])  # every reading identical -> elapsed == 0.0
    out = tmp_path / "out.json"

    code = main([str(fake_media), "-o", str(out)])

    assert code == 0
    assert out.exists()
    assert json.loads(out.read_text())["sentences"][0]["end"] == 12.0
    assert "0:12 audio" in capsys.readouterr().err


def test_empty_transcript_still_prints_a_summary(
    fake_parakeet: Callable[..., FakeModel],
    frozen_clock: Callable[[list[float]], None],
    fake_media: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No sentences means total == 0.0; the summary must say so, not crash."""
    fake_parakeet(tokens=[])
    frozen_clock([1000.0, 1002.0])
    out = tmp_path / "out.json"

    code = main([str(fake_media), "-o", str(out)])

    assert code == 0
    assert out.exists()
    assert "0:00 audio" in capsys.readouterr().err


def test_summary_reports_both_durations_and_no_multiple(
    fake_parakeet: Callable[..., FakeModel],
    frozen_clock: Callable[[list[float]], None],
    fake_media: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """12s of audio in 2s of wall clock.

    It does NOT say 6.0x. On a resumed run that figure would describe this
    process's clock against the whole file, most of which an earlier run
    transcribed -- an hour finished in two minutes would read as 30x. Two plain
    durations the reader can divide themselves beat one confident wrong number.
    """
    fake_parakeet(tokens=_tokens())
    frozen_clock([1000.0, 1002.0])
    out = tmp_path / "out.json"

    main([str(fake_media), "-o", str(out)])

    err = capsys.readouterr().err
    assert "done: 0:12 audio in 0:02 ->" in err
    assert "realtime" not in err


def test_the_live_bar_still_reports_speed(
    fake_parakeet: Callable[..., FakeModel],
    fake_media: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dropping the multiple from the summary must not drop it from the bar.

    The bar has the number the summary lacks: Progress.resumed_from_s lets it
    measure speed over this run's own work, so it stays truthful on a resume.
    """
    fake_parakeet(tokens=_tokens())

    main([str(fake_media), "-o", str(tmp_path / "out.json")])

    err = capsys.readouterr().err
    assert "x" in err.split("\n")[0]
    assert "running" in err


def test_no_resume_is_passed_through(
    fake_parakeet: Callable[..., FakeModel],
    fake_media: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag is only useful if it reaches transcribe()."""
    import dsj.suno as transcribe_mod

    fake_parakeet(tokens=_tokens())
    seen: list[bool] = []
    real = transcribe_mod.transcribe

    # Any on the passthrough: the spy forwards transcribe()'s whole signature
    # untouched, so naming its parameters here would only duplicate it.
    def spy(*args: Any, **kwargs: Any) -> Payload:
        seen.append(kwargs["resume"])
        return real(*args, **kwargs)

    monkeypatch.setattr(transcribe_mod, "transcribe", spy)

    main([str(fake_media), "-o", str(tmp_path / "a.json"), "--no-resume"])
    main([str(fake_media), "-o", str(tmp_path / "b.json")])

    assert seen == [False, True]


def test_the_diarize_flags_are_passed_through(
    fake_parakeet: Callable[..., FakeModel],
    fake_media: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two booleans, not a tri-state: does it run, and is failure fatal.

    A flag argparse accepts and main() then drops is worse than no flag: the
    user gets the behaviour they asked to avoid, silently and with exit 0.
    """
    import dsj.suno as transcribe_mod

    fake_parakeet(tokens=_tokens())
    seen: list[tuple[bool, bool]] = []
    real = transcribe_mod.transcribe

    def spy(*args: Any, **kwargs: Any) -> Payload:
        seen.append((kwargs["diarize"], kwargs["require_diarize"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(transcribe_mod, "transcribe", spy)

    main([str(fake_media), "-o", str(tmp_path / "a.json"), "--no-diarize"])
    main([str(fake_media), "-o", str(tmp_path / "b.json")])

    assert seen == [(False, False), (True, False)]


def test_the_summary_counts_speakers_only_when_there_are_some(
    fake_parakeet: Callable[..., FakeModel],
    fake_media: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_turns: Callable[..., list[Path]],
) -> None:
    """The one number the summary can gain without bringing back a rate.

    `total / elapsed` is deliberately absent from this line (see the module
    docstring); a count read out of the payload cannot divide by zero and
    cannot credit this run with a previous one's work.
    """
    from dsj.merge import Turn

    fake_parakeet(tokens=_tokens())
    fake_turns(turns=[Turn(0.0, 20.0, 0), Turn(20.0, 30.0, 1)],
               labels=["SPEAKER_01", "SPEAKER_02"])

    main([str(fake_media), "-o", str(tmp_path / "out.json")])
    assert "2 speakers ->" in capsys.readouterr().err

    # Degraded: no speakers key, so nothing to count and nothing claimed.
    main([str(fake_media), "-o", str(tmp_path / "plain.json"), "--no-diarize"])
    err = capsys.readouterr().err
    assert "speakers" not in err
    assert "0:12 audio" in err
