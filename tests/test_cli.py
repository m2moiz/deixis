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

import json

import pytest
from conftest import FakeToken

from deixis.transcribe import main

# main() feeds a path straight through transcribe(), which probes it for real;
# these tests are about the summary and not about ffmpeg, so they need the stub.
pytestmark = pytest.mark.usefixtures("already_extracted_media")


def _tokens() -> list[FakeToken]:
    """One sentence ending at 12.0s."""
    return [FakeToken(0.0, 0.4, "see"), FakeToken(0.5, 12.0, " this.")]


def test_instantaneous_run_still_prints_a_summary(
    fake_parakeet, frozen_clock, fake_media, tmp_path, capsys
):
    fake_parakeet(tokens=_tokens())
    frozen_clock([1000.0])  # every reading identical -> elapsed == 0.0
    out = tmp_path / "out.json"

    code = main([str(fake_media), "-o", str(out)])

    assert code == 0
    assert out.exists()
    assert json.loads(out.read_text())["sentences"][0]["end"] == 12.0
    assert "0:12 audio" in capsys.readouterr().err


def test_empty_transcript_still_prints_a_summary(
    fake_parakeet, frozen_clock, fake_media, tmp_path, capsys
):
    """No sentences means total == 0.0; the summary must say so, not crash."""
    fake_parakeet(tokens=[])
    frozen_clock([1000.0, 1002.0])
    out = tmp_path / "out.json"

    code = main([str(fake_media), "-o", str(out)])

    assert code == 0
    assert out.exists()
    assert "0:00 audio" in capsys.readouterr().err


def test_summary_reports_both_durations_and_no_multiple(
    fake_parakeet, frozen_clock, fake_media, tmp_path, capsys
):
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
    fake_parakeet, fake_media, tmp_path, capsys
):
    """Dropping the multiple from the summary must not drop it from the bar.

    The bar has the number the summary lacks: Progress.resumed_from_s lets it
    measure speed over this run's own work, so it stays truthful on a resume.
    """
    fake_parakeet(tokens=_tokens())

    main([str(fake_media), "-o", str(tmp_path / "out.json")])

    err = capsys.readouterr().err
    assert "x" in err.split("\n")[0]
    assert "running" in err


def test_no_resume_is_passed_through(fake_parakeet, fake_media, tmp_path, monkeypatch):
    """The flag is only useful if it reaches transcribe()."""
    import deixis.transcribe as transcribe_mod

    fake_parakeet(tokens=_tokens())
    seen: list[bool] = []
    real = transcribe_mod.transcribe

    def spy(*args, **kwargs):
        seen.append(kwargs["resume"])
        return real(*args, **kwargs)

    monkeypatch.setattr(transcribe_mod, "transcribe", spy)

    main([str(fake_media), "-o", str(tmp_path / "a.json"), "--no-resume"])
    main([str(fake_media), "-o", str(tmp_path / "b.json")])

    assert seen == [False, True]
