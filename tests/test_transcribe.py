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
from deixis.diarize import DiarizationUnavailable
from deixis.merge import Turn
from deixis.transcribe import CHUNK_S, OVERLAP_S, Progress, transcribe

# transcribe() probes its input for real; these tests care about the chunk loop
# and not about ffmpeg, so they need the stub that used to be autouse.
pytestmark = pytest.mark.usefixtures("already_extracted_media", "no_real_diarizer")

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


# --- speaker labels ------------------------------------------------------


def _one_speaker(**kw):
    """The fake diarization every labelled test below uses unless it says more.

    One turn spanning the whole of _tokens()'s single 750s sentence, so the
    token vote has an unambiguous answer and the tests can be about the wiring.
    """
    return dict(turns=[Turn(0.0, 800.0, 0)], labels=["SPEAKER_01"], **kw)


def test_sentences_carry_a_speaker_index_into_the_speakers_list(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    fake_parakeet(tokens=_tokens())
    fake_turns(**_one_speaker())
    out = tmp_path / "out.json"

    payload = transcribe(fake_media, out)

    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert on_disk["speakers"] == ["SPEAKER_01"]
    assert on_disk["diarization"] == "senko 0.0.0-fake"
    for sentence in on_disk["sentences"]:
        assert 0 <= sentence["speaker"] < len(on_disk["speakers"])


def test_the_labelled_schema_only_adds_keys(fake_parakeet, fake_media, tmp_path, fake_turns):
    """Additive or nothing. Every downstream reader of the old shape still works.

    Pinned against the unlabelled run in the same test rather than against a
    literal key list, so this cannot drift out of agreement with the payload.
    """
    fake_parakeet(tokens=_tokens())
    fake_turns(**_one_speaker())

    plain = transcribe(fake_media, tmp_path / "plain.json", diarize=False)
    labelled = transcribe(fake_media, tmp_path / "labelled.json")

    assert set(labelled) - set(plain) == {"speakers", "diarization"}
    assert set(plain) - set(labelled) == set()
    for before, after in zip(plain["sentences"], labelled["sentences"]):
        assert set(after) - set(before) == {"speaker"}
        assert {k: after[k] for k in before} == before


def test_no_diarize_output_is_the_old_schema_exactly(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    """The byte-identity guard. --no-diarize must produce today's file.

    `fake_turns` is installed and asserted never called: "the keys are absent"
    would also hold for a pass that ran and failed, and those are different
    bugs.
    """
    fake_parakeet(tokens=_tokens())
    calls = fake_turns(**_one_speaker())
    out = tmp_path / "out.json"

    payload = transcribe(fake_media, out, diarize=False)

    assert calls == []
    assert set(payload) == {"audio", "model", "text", "sentences"}
    assert set(payload["sentences"][0]) == {"start", "end", "text", "tokens"}


def test_diarization_failure_leaves_a_complete_unlabelled_transcript(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    """An optional pass may not cost the hour of ASR that ran before it."""
    fake_parakeet(tokens=_tokens())
    fake_turns(raises=DiarizationUnavailable("senko is not installed"))
    out = tmp_path / "out.json"

    payload = transcribe(fake_media, out)

    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert on_disk["sentences"][0]["text"] == "see this column here."
    assert "speaker" not in on_disk["sentences"][0]
    assert "speakers" not in on_disk
    # The absence of this key is how a reader tells "not run" from "ran and
    # found one speaker". A run that failed must not claim provenance.
    assert "diarization" not in on_disk


def test_require_diarize_makes_the_failure_fatal(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    fake_parakeet(tokens=_tokens())
    fake_turns(raises=DiarizationUnavailable("senko is not installed"))
    out = tmp_path / "out.json"

    with pytest.raises(DiarizationUnavailable):
        transcribe(fake_media, out, require_diarize=True)

    # Still written: the caller asked for labels or nothing, but the ASR work
    # is banked on disk either way and re-running it would cost the hour.
    assert "speakers" not in json.loads(out.read_text())


def test_a_bug_in_diarization_is_not_swallowed(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    """Same narrowness as the boundary itself: only DiarizationUnavailable
    degrades. A TypeError here is a bug in deixis and must be loud."""
    fake_parakeet(tokens=_tokens())
    fake_turns(raises=TypeError("unsupported operand"))

    with pytest.raises(TypeError):
        transcribe(fake_media, tmp_path / "out.json")


def test_the_transcript_is_written_before_diarization_runs(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    """§4's ordering, which is what makes optionality structural.

    The fake reads `out` from inside the diarization call. A refactor that
    holds the payload and writes once at the end breaks this and nothing else.
    """
    fake_parakeet(tokens=_tokens())
    out = tmp_path / "out.json"
    seen: list[dict] = []

    fake_turns(**_one_speaker(then=lambda wav: seen.append(json.loads(out.read_text()))))

    transcribe(fake_media, out)

    assert seen, "diarization never ran"
    assert seen[0]["sentences"][0]["text"] == "see this column here."
    assert "speaker" not in seen[0]["sentences"][0]


def test_the_checkpoint_is_gone_before_diarization_runs(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    """The checkpoint protects ASR, and ASR is banked once `out` exists.

    Left in place across this pass, a diarization crash would strand it, and
    the next run would resume audio it has already transcribed.
    """
    fake_parakeet(sample_rate=RATE, tokens=[], audio_s=360.0)
    out = tmp_path / "out.json"
    ckpt = checkpoint_path_for(out)
    seen: list[bool] = []

    fake_turns(**_one_speaker(then=lambda wav: seen.append(ckpt.exists())))

    transcribe(fake_media, out)

    assert seen == [False]


def test_the_diarizing_state_is_reported_between_running_and_done(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    fake_parakeet(sample_rate=RATE, tokens=_tokens(), audio_s=360.0)
    fake_turns(**_one_speaker())
    states: list[str] = []

    transcribe(
        fake_media,
        tmp_path / "out.json",
        on_progress=lambda p, state: states.append(state),
    )

    assert "diarizing" in states
    assert states.index("running") < states.index("diarizing")
    assert states.index("diarizing") < states.index("done")
    # Two frames, not a bar: senko emits nothing from inside, so the honest
    # report is a start and an end rather than invented intermediate progress.
    assert states.count("diarizing") == 2


def test_the_diarizing_heartbeat_reaches_the_status_file(
    fake_parakeet, fake_media, tmp_path, fake_turns
):
    """A detached run is inspected through this file, and this is the phase a
    watcher would otherwise see as a stall between "running" and "done"."""
    fake_parakeet(sample_rate=RATE, tokens=_tokens(), audio_s=360.0)
    status = tmp_path / "status.json"
    seen: list[dict] = []

    fake_turns(**_one_speaker(then=lambda wav: seen.append(json.loads(status.read_text()))))

    transcribe(fake_media, tmp_path / "out.json", status_path=status)

    assert seen[0]["state"] == "diarizing"


def test_the_labelled_transcript_is_written_through_the_atomic_writer(
    fake_parakeet, fake_media, tmp_path, fake_turns, monkeypatch
):
    """Two atomic writes to the same path, so there is no instant in which the
    transcript is absent or partial. A merge that resolves either back to
    write_text reintroduces the torn read with the whole suite green."""
    fake_parakeet(tokens=_tokens())
    out = tmp_path / "out.json"
    fake_turns(**_one_speaker())
    seen = _spy_on_atomic_write(monkeypatch)

    transcribe(fake_media, out)

    assert seen.count(out) == 2


def test_the_diarizer_is_handed_the_extracted_wav_not_the_source(
    fake_parakeet, fake_media, tmp_path, fake_turns, monkeypatch
):
    """senko wants a wav and the source is normally a .mov.

    media.py already produces the 16 kHz mono pcm_s16le the ASR pass consumes,
    and it lives only until the temp directory is torn down -- so the pass has
    to run inside that window, against that file.
    """
    from deixis import media

    fake_parakeet(tokens=_tokens())
    extracted: list[Path] = []
    monkeypatch.setattr(media, "needs_conversion", lambda stream, rate: True)

    def fake_extract(source, dest, rate, on_progress=None):
        dest.write_bytes(b"RIFF")
        extracted.append(dest)
        return dest

    monkeypatch.setattr(media, "extract_audio", fake_extract)
    calls = fake_turns(**_one_speaker())

    transcribe(fake_media, tmp_path / "out.json")

    assert calls == extracted
    assert calls[0] != fake_media
