"""transcribe() wiring, exercised against a fake model.

The unit tests in test_chunk_callback.py prove the conversion is correct. This
module proves transcribe() actually hands the model's sample rate to it -- the
half a unit test cannot see.
"""

import json

import pytest
from conftest import FakeSentence, FakeToken

from deixis.transcribe import CHUNK_S, OVERLAP_S, Progress, transcribe

RATE = 16_000


def _sentences() -> list[FakeSentence]:
    return [
        FakeSentence(
            start=0.0,
            end=750.0,
            text="see this column here",
            tokens=[FakeToken(start=0.0, end=0.4, text="see")],
        )
    ]


def test_transcribe_reports_progress_in_seconds(fake_parakeet, tmp_path):
    """The end-to-end version of the units regression."""
    fake_parakeet(
        sample_rate=RATE,
        sentences=_sentences(),
        chunk_positions=[(750.0 * RATE, 4427.028 * RATE)],
    )
    seen: list[Progress] = []

    transcribe(
        tmp_path / "in.wav",
        tmp_path / "out.json",
        on_progress=lambda p, state: seen.append(p) if state == "running" else None,
    )

    assert seen[-1].audio_done_s == pytest.approx(750.0)
    assert seen[-1].audio_total_s == pytest.approx(4427.028)


def test_transcribe_uses_the_models_sample_rate(fake_parakeet, tmp_path):
    """A model at 8kHz must halve the seconds, not reuse a hardcoded 16000."""
    fake_parakeet(
        sample_rate=8_000,
        sentences=_sentences(),
        chunk_positions=[(10.0 * 8_000, 60.0 * 8_000)],
    )
    seen: list[Progress] = []

    transcribe(
        tmp_path / "in.wav",
        tmp_path / "out.json",
        on_progress=lambda p, state: seen.append(p) if state == "running" else None,
    )

    assert seen[-1].audio_done_s == pytest.approx(10.0)
    assert seen[-1].audio_total_s == pytest.approx(60.0)


def test_transcribe_always_chunks(fake_parakeet, tmp_path):
    """chunk_duration=None feeds an hour of audio to Metal in one buffer and dies,
    and it is also what makes chunk_callback fire at all."""
    model = fake_parakeet(sentences=_sentences())

    transcribe(tmp_path / "in.wav", tmp_path / "out.json")

    assert model.calls[0]["chunk_duration"] == CHUNK_S
    assert model.calls[0]["overlap_duration"] == OVERLAP_S


def test_transcribe_writes_the_timestamped_index(fake_parakeet, tmp_path):
    fake_parakeet(sentences=_sentences())
    out = tmp_path / "out.json"

    payload = transcribe(tmp_path / "in.wav", out)

    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert on_disk["sentences"][0]["start"] == 0.0
    assert on_disk["sentences"][0]["end"] == 750.0
    assert on_disk["sentences"][0]["tokens"] == [{"t": 0.0, "w": "see"}]


def test_transcribe_on_an_empty_result_still_writes_a_file(fake_parakeet, tmp_path):
    """No sentences must not mean no output -- the file is the deliverable."""
    fake_parakeet(sentences=[])
    out = tmp_path / "out.json"

    payload = transcribe(tmp_path / "in.wav", out)

    assert out.exists()
    assert payload["sentences"] == []


def test_status_file_carries_seconds_and_a_state(fake_parakeet, tmp_path):
    """A detached run is inspected through this file; it is the only window in."""
    fake_parakeet(
        sample_rate=RATE,
        sentences=_sentences(),
        chunk_positions=[(750.0 * RATE, 4427.028 * RATE)],
    )
    status = tmp_path / "status.json"
    seen: list[dict] = []

    # Reading from inside on_progress pins the production fan-out order: emit
    # writes the status file first, then calls on_progress.
    def capture(_p, state):
        if state == "running":
            seen.append(json.loads(status.read_text()))

    transcribe(
        tmp_path / "in.wav",
        tmp_path / "out.json",
        status_path=status,
        on_progress=capture,
    )

    running = seen[-1]
    assert running["state"] == "running"
    assert running["audio_done_s"] == pytest.approx(750.0)
    assert running["audio_total_s"] == pytest.approx(4427.028)
    assert running["fraction"] == pytest.approx(0.1694, abs=1e-4)


def test_status_file_ends_in_the_done_state(fake_parakeet, tmp_path):
    fake_parakeet(
        sample_rate=RATE,
        sentences=_sentences(),
        chunk_positions=[(750.0 * RATE, 4427.028 * RATE)],
    )
    status = tmp_path / "status.json"

    transcribe(tmp_path / "in.wav", tmp_path / "out.json", status_path=status)

    final = json.loads(status.read_text())
    assert final["state"] == "done"
    assert final["audio_done_s"] == pytest.approx(750.0)
    assert final["fraction"] == 1.0


def test_no_status_path_writes_nothing(fake_parakeet, tmp_path):
    fake_parakeet(
        sentences=_sentences(),
        chunk_positions=[(750.0 * RATE, 4427.028 * RATE)],
    )

    transcribe(tmp_path / "in.wav", tmp_path / "out.json")

    assert list(tmp_path.iterdir()) == [tmp_path / "out.json"]
