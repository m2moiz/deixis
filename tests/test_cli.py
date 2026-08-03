"""main()'s summary line.

The `total / elapsed` in the summary is unguarded and sits AFTER out.write_text,
so on an instantaneous run it turns a completed transcription into a traceback
with the output file already on disk. Reaching elapsed == 0.0 needs a frozen
clock -- time.monotonic has nanosecond resolution on macOS -- which is why this
is a P3 and not a field report.
"""

import json

import pytest
from conftest import FakeSentence, FakeToken

from deixis.transcribe import main

# main() feeds a path straight through transcribe(), which probes it for real;
# these tests pass paths that do not exist, so they need the media stub.
pytestmark = pytest.mark.usefixtures("already_extracted_media")


def _sentences() -> list[FakeSentence]:
    return [
        FakeSentence(
            start=0.0,
            end=12.0,
            text="see this column here",
            tokens=[FakeToken(start=0.0, end=0.4, text="see")],
        )
    ]


def test_instantaneous_run_still_prints_a_summary(
    fake_parakeet, frozen_clock, tmp_path, capsys
):
    fake_parakeet(sentences=_sentences())
    frozen_clock([1000.0])  # every reading identical -> elapsed == 0.0
    out = tmp_path / "out.json"

    code = main([str(tmp_path / "in.wav"), "-o", str(out)])

    assert code == 0
    assert out.exists()
    assert json.loads(out.read_text())["sentences"][0]["end"] == 12.0
    assert "0:12 audio" in capsys.readouterr().err


def test_empty_transcript_still_prints_a_summary(
    fake_parakeet, frozen_clock, tmp_path, capsys
):
    """No sentences means total == 0.0; the summary must say so, not crash."""
    fake_parakeet(sentences=[])
    frozen_clock([1000.0, 1002.0])
    out = tmp_path / "out.json"

    code = main([str(tmp_path / "in.wav"), "-o", str(out)])

    assert code == 0
    assert out.exists()
    assert "0:00 audio" in capsys.readouterr().err


def test_summary_reports_the_realtime_multiple(
    fake_parakeet, frozen_clock, tmp_path, capsys
):
    """12s of audio in 2s of wall clock is 6.0x."""
    fake_parakeet(sentences=_sentences())
    frozen_clock([1000.0, 1002.0])
    out = tmp_path / "out.json"

    main([str(tmp_path / "in.wav"), "-o", str(out)])

    assert "6.0x realtime" in capsys.readouterr().err
