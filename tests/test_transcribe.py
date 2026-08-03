"""transcribe() wiring, exercised against a fake model.

The unit tests in test_chunk_callback.py prove the sample-to-second conversion
is correct. This module proves transcribe() actually hands the model's sample
rate to it, and that the chunk loop, the checkpoint and the heartbeat are wired
to each other -- the half a unit test cannot see.

Only the decode is faked. The chunk boundaries, the offsets and the merge are
the real ones from deixis/chunking.py.
"""

import json
from pathlib import Path

import pytest
from conftest import FakeToken

from deixis.checkpoint import checkpoint_path_for
from deixis.transcribe import CHUNK_S, OVERLAP_S, Progress, transcribe

# transcribe() probes its input for real; these tests care about the chunk loop
# and not about ffmpeg, so they need the stub that used to be autouse.
pytestmark = pytest.mark.usefixtures("already_extracted_media")

RATE = 16_000


def _tokens() -> list[FakeToken]:
    """One sentence ending at 750.0s -- the trailing '.' is what closes it."""
    return [
        FakeToken(0.0, 0.4, "see"),
        FakeToken(0.5, 1.0, " this"),
        FakeToken(1.1, 1.6, " column"),
        FakeToken(1.7, 750.0, " here."),
    ]


def test_transcribe_reports_progress_in_seconds(fake_parakeet, fake_media, tmp_path):
    """The end-to-end version of the units regression.

    A 74-minute file at 16 kHz is 70,832,448 samples; reported as seconds that
    is 2.2 years of audio, and the shipped bug did exactly that.
    """
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=4427.028)
    seen: list[Progress] = []

    transcribe(
        fake_media,
        tmp_path / "out.json",
        on_progress=lambda p, state: seen.append(p) if state == "running" else None,
    )

    assert seen[0].audio_done_s == pytest.approx(CHUNK_S)
    assert seen[0].audio_total_s == pytest.approx(4427.028)
    assert seen[-1].audio_done_s == pytest.approx(4427.028)


def test_transcribe_uses_the_models_sample_rate(fake_parakeet, fake_media, tmp_path):
    """A model at 8kHz must halve the seconds, not reuse a hardcoded 16000."""
    fake_parakeet(sample_rate=8_000, tokens=[], audio_s=60.0)
    seen: list[Progress] = []

    transcribe(
        fake_media,
        tmp_path / "out.json",
        on_progress=lambda p, state: seen.append(p) if state == "running" else None,
    )

    assert seen[-1].audio_done_s == pytest.approx(60.0)
    assert seen[-1].audio_total_s == pytest.approx(60.0)


def test_transcribe_always_chunks(fake_parakeet, fake_media, tmp_path):
    """Feeding an hour of audio to Metal in one buffer asks ~14.5GB and dies.

    Asserted on the audio actually handed to the decoder, chunk by chunk: four
    chunks for 360s under the 120s/15s geometry, the first a full chunk long and
    the last the 45s remainder.
    """
    model = fake_parakeet(sample_rate=RATE, tokens=[], audio_s=360.0)

    transcribe(fake_media, tmp_path / "out.json")

    assert [len(m) for m in model.mels] == [
        int(CHUNK_S * RATE),
        int(CHUNK_S * RATE),
        int(CHUNK_S * RATE),
        360 * RATE - int(315.0 * RATE),
    ]
    # The stride is chunk minus overlap, so consecutive chunks start 105s apart.
    assert model.mels[1].start - model.mels[0].start == int((CHUNK_S - OVERLAP_S) * RATE)


def test_transcribe_writes_the_timestamped_index(fake_parakeet, fake_media, tmp_path):
    fake_parakeet(tokens=_tokens())
    out = tmp_path / "out.json"

    payload = transcribe(fake_media, out)

    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert on_disk["sentences"][0]["start"] == 0.0
    assert on_disk["sentences"][0]["end"] == 750.0
    assert on_disk["sentences"][0]["text"] == "see this column here."
    assert on_disk["sentences"][0]["tokens"][0] == {"t": 0.0, "w": "see"}
    assert on_disk["audio"] == str(fake_media)


def test_transcribe_on_an_empty_result_still_writes_a_file(
    fake_parakeet, fake_media, tmp_path
):
    """No sentences must not mean no output -- the file is the deliverable."""
    fake_parakeet(tokens=[])
    out = tmp_path / "out.json"

    payload = transcribe(fake_media, out)

    assert out.exists()
    assert payload["sentences"] == []


def test_status_file_carries_seconds_and_a_state(fake_parakeet, fake_media, tmp_path):
    """A detached run is inspected through this file; it is the only window in."""
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=4427.028)
    status = tmp_path / "status.json"
    seen: list[dict] = []

    # Reading from inside on_progress pins the production fan-out order: emit
    # writes the status file first, then calls on_progress.
    def capture(_p, state):
        if state == "running":
            seen.append(json.loads(status.read_text()))

    transcribe(fake_media, tmp_path / "out.json", status_path=status, on_progress=capture)

    first = seen[0]
    assert first["state"] == "running"
    assert first["audio_done_s"] == pytest.approx(CHUNK_S)
    assert first["audio_total_s"] == pytest.approx(4427.028)
    assert first["fraction"] == pytest.approx(0.0271, abs=1e-4)


def test_status_file_ends_in_the_done_state(fake_parakeet, fake_media, tmp_path):
    fake_parakeet(sample_rate=RATE, tokens=_tokens())
    status = tmp_path / "status.json"

    transcribe(fake_media, tmp_path / "out.json", status_path=status)

    final = json.loads(status.read_text())
    assert final["state"] == "done"
    assert final["audio_done_s"] == pytest.approx(750.0)
    assert final["fraction"] == 1.0


def test_no_status_path_writes_nothing(fake_parakeet, fake_media, tmp_path):
    fake_parakeet(tokens=_tokens())

    transcribe(fake_media, tmp_path / "out.json")

    assert set(tmp_path.iterdir()) == {fake_media, tmp_path / "out.json"}


def test_a_completed_run_leaves_no_checkpoint(fake_parakeet, fake_media, tmp_path):
    """The checkpoint exists to be outlived. One left behind would be replayed
    by the next run over audio it no longer describes."""
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=360.0)
    out = tmp_path / "out.json"

    transcribe(fake_media, out)

    assert not checkpoint_path_for(out).exists()


def test_the_checkpoint_is_banked_once_per_chunk_before_progress_is_reported(
    fake_parakeet, fake_media, tmp_path
):
    """An observer must never be able to outrun what a restart could recover.

    Reading the checkpoint from inside on_progress pins the order: if the
    report came first, a watcher at 40% could restart and find nothing banked.
    """
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=360.0)
    out = tmp_path / "out.json"
    ckpt = checkpoint_path_for(out)
    banked: list[int] = []

    def capture(_p, state):
        if state == "running":
            banked.append(json.loads(ckpt.read_text())["next_start"])

    transcribe(fake_media, out, on_progress=capture)

    # Four chunks starting at 0, 105s, 210s, 315s; each banks the FOLLOWING
    # boundary, and the last banks the end of the audio rather than a boundary
    # past it.
    assert banked == [
        int(105.0 * RATE),
        int(210.0 * RATE),
        int(315.0 * RATE),
        360 * RATE,
    ]


def test_an_interrupted_run_leaves_a_checkpoint_and_no_transcript(
    fake_parakeet, fake_media, tmp_path
):
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=360.0)
    out = tmp_path / "out.json"

    class Interrupt(Exception):
        pass

    seen = 0

    def die_after_two(_p, state):
        nonlocal seen
        if state != "running":
            return
        seen += 1
        if seen == 2:
            raise Interrupt

    with pytest.raises(Interrupt):
        transcribe(fake_media, out, on_progress=die_after_two)

    assert not out.exists(), "a partial transcript was written as if complete"
    banked = json.loads(checkpoint_path_for(out).read_text())
    assert banked["next_start"] == int(210.0 * RATE)


def test_no_resume_removes_a_checkpoint_it_will_not_use(
    fake_parakeet, fake_media, tmp_path
):
    fake_parakeet(sample_rate=RATE, tokens=_tokens())
    out = tmp_path / "out.json"
    ckpt = checkpoint_path_for(out)
    ckpt.write_text("{}")

    transcribe(fake_media, out, resume=False)

    assert not ckpt.exists()


def _spy_on_atomic_write(monkeypatch) -> list[Path]:
    """Record every path routed through the atomic writer, and really write it."""
    import deixis.transcribe as transcribe_mod

    seen: list[Path] = []
    real = transcribe_mod.atomic_write_text

    def spy(path, text, **kwargs):
        seen.append(path)
        real(path, text, **kwargs)

    monkeypatch.setattr(transcribe_mod, "atomic_write_text", spy)
    return seen


def test_the_heartbeat_is_written_through_the_atomic_writer(
    fake_parakeet, fake_media, tmp_path, monkeypatch
):
    """Guards the call site, not the writer.

    tests/test_atomic.py proves atomic_write_text is atomic; it says nothing
    about whether transcribe() still calls it. A merge that resolves this line
    back to status_path.write_text reintroduces the torn read with the whole
    suite green, so the wiring needs its own assertion.
    """
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=360.0)
    status = tmp_path / "status.json"
    seen = _spy_on_atomic_write(monkeypatch)

    transcribe(fake_media, tmp_path / "out.json", status_path=status)

    assert seen.count(status) >= 2, "every heartbeat must go through the atomic writer"
    assert json.loads(status.read_text())["state"] == "done"


def test_the_transcript_is_written_through_the_atomic_writer(
    fake_parakeet, fake_media, tmp_path, monkeypatch
):
    """`out` is what every downstream tool reads, and a truncated transcript
    does not announce itself -- it merely looks short."""
    fake_parakeet(sample_rate=RATE, tokens=_tokens())
    out = tmp_path / "out.json"
    seen = _spy_on_atomic_write(monkeypatch)

    transcribe(fake_media, out)

    assert out in seen


def test_the_failure_status_is_written_through_the_atomic_writer(
    fake_parakeet, fake_media, tmp_path, monkeypatch
):
    """main()'s except-handler writes the one document a watcher polls hardest.

    A watcher distinguishing "died" from "not started yet" reads this file in a
    tight loop, so it is the reader most likely to land inside a torn write.
    """
    import deixis.transcribe as transcribe_mod

    status = tmp_path / "status.json"
    seen = _spy_on_atomic_write(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(transcribe_mod, "transcribe", boom)

    with pytest.raises(RuntimeError):
        transcribe_mod.main(
            [str(fake_media), "-o", str(tmp_path / "out.json"), "--status", str(status)]
        )

    assert seen == [status]
    failed = json.loads(status.read_text())
    assert failed["state"] == "failed"
    assert "model exploded" in failed["error"]
