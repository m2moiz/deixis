"""The `jaano` command: does each verb reach the thing it names?

Small surface, but the one that decides whether any of this is usable. The
measurement that justifies frame retrieval (docs/do-marks-help.md) was only
possible because a Python snippet was hand-written into an agent's prompt; these
tests exist so the supported path cannot quietly stop working.

The dispatch tests assert ROUTING, not behaviour -- transcribe and mark are
covered by their own suites, and re-running real ASR here would buy nothing for
two minutes of wall clock.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from jaano.cli import main

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Two colours, 4 seconds each, so a seek can be checked for landing right."""
    parts: list[Path] = []
    for i, colour in enumerate(("0x101010", "0xE0E0E0")):
        part = tmp_path / f"p{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"color=c={colour}:s=320x240:d=4:r=10", "-pix_fmt", "yuv420p", str(part)],
            check=True,
        )
        parts.append(part)
    listing = tmp_path / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    dest = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(dest)],
        check=True,
    )
    return dest


def _dimensions(image: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(image)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _mean_grey(image: Path) -> float:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(image),
         "-vf", "scale=4:4,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return sum(out) / len(out)


def test_a_bare_invocation_explains_itself_and_fails() -> None:
    """Exit 2, not 0.

    Forgetting the verb is a caller error, and a shell script checking the exit
    code should be told about it.
    """
    assert main([]) == 2


def test_help_is_not_an_error() -> None:
    assert main(["--help"]) == 0
    assert main(["help"]) == 0


def test_an_unknown_verb_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["wat"]) == 2
    # Substring, not the whole line: Typer renders errors in a box that wraps
    # to the terminal width, so an exact match would fail on a narrow terminal
    # and pass on a wide one.
    assert "No such command" in capsys.readouterr().err


def test_frame_writes_the_image_and_prints_its_path(
    video: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "f.jpg"
    assert main(["dikhao", str(video), "1.0", "-o", str(dest)]) == 0
    assert dest.exists()
    # The path alone on stdout, so a caller can use it directly.
    assert capsys.readouterr().out.strip() == str(dest)


def test_frame_seeks_to_the_second_asked_for(video: Path, tmp_path: Path) -> None:
    """The fixture is near-black for 0-4s and near-white after.

    A seek that landed in the wrong half is invisible to "a file was written".
    """
    early = tmp_path / "early.png"
    late = tmp_path / "late.png"
    main(["dikhao", str(video), "1.0", "-o", str(early)])
    main(["dikhao", str(video), "6.0", "-o", str(late)])
    assert _mean_grey(early) < 64 < 192 < _mean_grey(late)


def test_frame_scales_to_the_requested_width(video: Path, tmp_path: Path) -> None:
    """Aspect preserved, height even.

    `scale=W:-2`, not `-1`: the latter can produce an odd height that some
    encoders refuse outright.
    """
    dest = tmp_path / "f.jpg"
    main(["dikhao", str(video), "1.0", "-o", str(dest), "--width", "160"])
    assert _dimensions(dest) == (160, 120)


def test_width_zero_keeps_the_source_resolution(video: Path, tmp_path: Path) -> None:
    dest = tmp_path / "f.jpg"
    main(["dikhao", str(video), "1.0", "-o", str(dest), "--width", "0"])
    assert _dimensions(dest) == (320, 240)


def test_a_bad_timestamp_surfaces_rather_than_writing_nothing(
    video: Path, tmp_path: Path
) -> None:
    from jaano.media import MediaError

    with pytest.raises(MediaError, match="past the end"):
        main(["dikhao", str(video), "9999", "-o", str(tmp_path / "f.jpg")])


def test_the_module_entry_point_reaches_the_mark_command(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python -m jaano.dekho` must still work, and must go through Typer.

    The direction of this reversed when the CLI moved to Typer: frames.main is
    now a shim ONTO the app rather than something the dispatcher calls. What
    matters either way is that the flags parse to the same values.
    """
    import jaano.dekho as frames_mod

    seen: dict[str, object] = {}

    def fake(media: Path, transcript: Path, out: Path, **kw: object) -> dict[str, object]:
        seen.update({"media": media, "transcript": transcript, "out": out, **kw})
        return {"marks": [], "marks_meta": {"frames_sampled": 0}}

    monkeypatch.setattr(frames_mod, "mark_video", fake)
    assert frames_mod.main([str(video), "-t", "t.json", "--budget", "7", "--min-gap", "0"]) == 0
    assert seen["media"] == video
    assert seen["transcript"] == Path("t.json")
    assert seen["out"] == Path("t.json"), "no -o means overwrite the transcript"
    assert seen["budget"] == 7
    assert seen["min_gap_s"] == 0.0
    # Untouched flags must arrive as the defaults frames.py documents, not as
    # the zero sentinels the Typer signature uses to detect "not given".
    assert seen["fps"] == frames_mod.DEFAULT_FPS
    assert seen["delta"] == frames_mod.DEFAULT_DELTA


def test_mark_writes_elsewhere_when_told_to(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-o must beat the transcript path, not be discarded in favour of it.

    The no--o case above passes whether or not the option is honoured, so on
    its own it pins nothing -- a mutant that ignored -o survived it.
    """
    import jaano.dekho as frames_mod

    seen: dict[str, object] = {}

    def fake(media: Path, transcript: Path, out: Path, **kw: object) -> dict[str, object]:
        seen.update({"transcript": transcript, "out": out})
        return {"marks": [], "marks_meta": {"frames_sampled": 0}}

    monkeypatch.setattr(frames_mod, "mark_video", fake)
    assert frames_mod.main([str(video), "-t", "t.json", "-o", "elsewhere.json"]) == 0
    assert seen["transcript"] == Path("t.json")
    assert seen["out"] == Path("elsewhere.json")


def test_the_module_entry_point_reaches_the_transcribe_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jaano.suno as transcribe_mod

    seen: dict[str, object] = {}

    def fake(media: Path, out: Path, model: str = "", **kw: object) -> dict[str, object]:
        seen.update({"media": media, "out": out, "model": model, **kw})
        return {"sentences": []}

    monkeypatch.setattr(transcribe_mod, "transcribe", fake)
    out = tmp_path / "out.json"
    assert transcribe_mod.main(["v.mov", "-o", str(out), "--no-diarize"]) == 0
    assert seen["media"] == Path("v.mov")
    assert seen["out"] == out
    assert seen["diarize"] is False
    assert seen["resume"] is True
    assert seen["model"] == transcribe_mod.DEFAULT_MODEL


def test_the_installed_console_script_works() -> None:
    """The entry point itself, not just the function behind it.

    A `[project.scripts]` typo is invisible to every test that imports main()
    directly -- and the console script is the whole point of this file.
    """
    proc = subprocess.run(
        ["uv", "run", "jaano", "--help"], capture_output=True, text=True, check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode == 0, proc.stderr
    for verb in ("transcribe", "mark", "frame"):
        assert verb in proc.stdout, proc.stdout


# --------------------------------------------------------------------------
# Containers the mark pass can read, the frame pass must be able to open
# --------------------------------------------------------------------------


@pytest.fixture
def transport_stream(video: Path, tmp_path: Path) -> Path:
    """The same content remuxed to MPEG-TS, which carries no global index."""
    dest = tmp_path / "v.ts"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c", "copy",
         "-f", "mpegts", str(dest)],
        check=True,
    )
    return dest


def test_a_frame_can_be_pulled_from_a_container_with_no_index(
    transport_stream: Path, tmp_path: Path
) -> None:
    """An index whose entries cannot be opened is worse than no index.

    MPEG-TS has no global header, so `-ss` BEFORE `-i` seeks to nothing and
    ffmpeg writes no frame -- at a timestamp the file plainly contains. Measured
    before the fix: extract_tile_grid read all 6 seconds of a .ts that
    extract_frame could not open at t=3, and the error blamed the timestamp.
    """
    dest = tmp_path / "f.jpg"
    assert main(["dikhao", str(transport_stream), "6.0", "-o", str(dest)]) == 0
    assert dest.exists()
    assert _mean_grey(dest) > 192, "must land in the second, near-white half"


def test_a_missing_output_directory_says_so(video: Path, tmp_path: Path) -> None:
    """Not 'a timestamp past the end' -- that hint used to be unconditional.

    Writing frames into a scratch directory it forgot to create is the mistake
    a calling agent actually makes, and it was being pointed at the wrong
    variable.
    """
    from jaano.media import MediaError

    with pytest.raises(MediaError, match="does not exist"):
        main(["dikhao", str(video), "1.0", "-o", str(tmp_path / "nope" / "f.jpg")])


def test_past_the_end_still_says_past_the_end(video: Path, tmp_path: Path) -> None:
    """The hint is now conditional, so prove it still fires when it should."""
    from jaano.media import MediaError

    with pytest.raises(MediaError, match="past the end of this"):
        main(["dikhao", str(video), "9999", "-o", str(tmp_path / "f.jpg")])
